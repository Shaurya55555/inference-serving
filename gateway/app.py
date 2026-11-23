"""FastAPI gateway in front of one or more OpenAI-compatible providers (llama.cpp's llama-server, Groq, Gemini, ...).

Adds: exact-match LRU response cache (deterministic requests only), merging of identical in-flight requests,
multi-provider failover (priority order, retries with backoff, per-provider rate limits, circuit breakers) and /metrics.

UPSTREAM=http://127.0.0.1:8080 uvicorn gateway.app:app --port 8000                 # one local provider, as before
PROVIDERS_FILE=providers.json uvicorn gateway.app:app --port 8000                  # local first, cloud as fallback
Tuning env vars: FAILOVER=0|1, BREAKER=0|1, MAX_ATTEMPTS=n, BREAKER_THRESHOLD, BREAKER_COOLDOWN, DEADLINE_S.
"""
import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .providers import load_providers
from .router import AllProvidersFailed, ClientError, FailoverRouter, RouterConfig

CACHE_SIZE = int(os.getenv("CACHE_SIZE", "512"))

app = FastAPI(title="inference-gateway")
client = httpx.AsyncClient(timeout=300)


def flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


def build_router() -> FailoverRouter:
    cfg = RouterConfig(
        failover=flag("FAILOVER", True), breaker=flag("BREAKER", True),
        max_attempts=int(os.environ["MAX_ATTEMPTS"]) if os.getenv("MAX_ATTEMPTS") else None,
        breaker_threshold=int(os.getenv("BREAKER_THRESHOLD", "5")),
        breaker_cooldown=float(os.getenv("BREAKER_COOLDOWN", "10")),
        deadline_s=float(os.getenv("DEADLINE_S", "60")))
    return FailoverRouter(load_providers(), cfg, client=client)


router = build_router()
cache: OrderedDict[str, dict] = OrderedDict()
inflight: dict[str, asyncio.Future] = {}
stats = {"requests": 0, "cache_hits": 0, "merged": 0, "upstream_calls": 0, "completion_tokens": 0}


class Chat(BaseModel):
    messages: list[dict] = Field(min_length=1)
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0, le=2)


def key_of(req: Chat) -> str:
    return hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()


async def call_upstream(req: Chat) -> dict:
    """Send the request through the failover router. The provider that answered is returned as `served_by`."""
    stats["upstream_calls"] += 1
    try:
        data, route = await router.complete(req.model_dump())
    except ClientError as e:
        raise HTTPException(e.status, e.body)
    except AllProvidersFailed as e:
        raise HTTPException(503, {"error": "all providers failed", "tried": e.tried})
    return {**data, "served_by": route.provider}


@app.post("/v1/chat/completions")
async def chat(req: Chat):
    stats["requests"] += 1
    t0 = time.time()
    k = key_of(req)
    deterministic = req.temperature == 0
    if deterministic and k in cache:
        cache.move_to_end(k)
        stats["cache_hits"] += 1
        return {**cache[k], "gateway": {"cache": "hit", "ms": round((time.time() - t0) * 1000, 2)}}
    if deterministic and k in inflight:  # identical request already running: wait for it
        stats["merged"] += 1
        data = await inflight[k]
        return {**data, "gateway": {"cache": "merged", "ms": round((time.time() - t0) * 1000, 2)}}
    fut = asyncio.get_running_loop().create_future()
    if deterministic:
        inflight[k] = fut
    try:
        data = await call_upstream(req)
        stats["completion_tokens"] += data.get("usage", {}).get("completion_tokens", 0)
        if deterministic:
            cache[k] = data
            if len(cache) > CACHE_SIZE:
                cache.popitem(last=False)
        fut.set_result(data)
        return {**data, "gateway": {"cache": "miss", "ms": round((time.time() - t0) * 1000, 2)}}
    except Exception as e:
        fut.set_exception(e)
        fut.exception()  # mark retrieved so un-awaited futures do not warn
        raise
    finally:
        inflight.pop(k, None)


@app.get("/metrics")
def metrics():
    s = dict(stats)
    s["cache_hit_rate"] = round(s["cache_hits"] / max(1, s["requests"]), 3)
    s["cache_entries"] = len(cache)
    s["providers"] = router.snapshot()
    return s
