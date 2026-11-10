"""FastAPI gateway in front of llama.cpp's llama-server (OpenAI-compatible).

Adds: exact-match LRU response cache (deterministic requests only), merging of identical
in-flight requests, a concurrency cap, and /metrics.

UPSTREAM=http://127.0.0.1:8080 uvicorn gateway.app:app --port 8000
"""
import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

UPSTREAM = os.getenv("UPSTREAM", "http://127.0.0.1:8080")
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "512"))
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "8"))

app = FastAPI(title="inference-gateway")
client = httpx.AsyncClient(timeout=300)
sem = asyncio.Semaphore(MAX_INFLIGHT)
cache: OrderedDict[str, dict] = OrderedDict()
inflight: dict[str, asyncio.Future] = {}
stats = {"requests": 0, "cache_hits": 0, "merged": 0, "upstream_calls": 0, "completion_tokens": 0}


class Chat(BaseModel):
    messages: list[dict]
    max_tokens: int = 128
    temperature: float = 0.0


def key_of(req: Chat) -> str:
    return hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()


async def call_upstream(req: Chat) -> dict:
    async with sem:
        stats["upstream_calls"] += 1
        r = await client.post(f"{UPSTREAM}/v1/chat/completions", json=req.model_dump())
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()


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
    return s
