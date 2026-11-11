import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway import app as g  # noqa: E402


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    g.cache.clear()
    g.inflight.clear()
    for k in g.stats:
        g.stats[k] = 0
    calls = {"n": 0}

    async def fake_upstream(req):
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 3}}

    monkeypatch.setattr(g, "call_upstream", fake_upstream)
    return calls


def body(text="hi", temperature=0.0):
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 8, "temperature": temperature}


def test_second_identical_request_is_a_cache_hit(fresh):
    c = TestClient(g.app)
    assert c.post("/v1/chat/completions", json=body()).json()["gateway"]["cache"] == "miss"
    assert c.post("/v1/chat/completions", json=body()).json()["gateway"]["cache"] == "hit"
    assert fresh["n"] == 1
    assert c.get("/metrics").json()["cache_hit_rate"] == 0.5


def test_sampled_requests_are_never_cached(fresh):
    c = TestClient(g.app)
    c.post("/v1/chat/completions", json=body(temperature=0.8))
    c.post("/v1/chat/completions", json=body(temperature=0.8))
    assert fresh["n"] == 2 and not g.cache


def test_different_prompts_do_not_collide(fresh):
    c = TestClient(g.app)
    c.post("/v1/chat/completions", json=body("a"))
    c.post("/v1/chat/completions", json=body("b"))
    assert fresh["n"] == 2 and len(g.cache) == 2


def test_lru_eviction(monkeypatch, fresh):
    monkeypatch.setattr(g, "CACHE_SIZE", 2)
    c = TestClient(g.app)
    for t in ["a", "b", "c"]:
        c.post("/v1/chat/completions", json=body(t))
    assert len(g.cache) == 2


def test_concurrent_identical_requests_merge(fresh):
    async def go():
        req = g.Chat(**body("same"))
        return await asyncio.gather(*[g.chat(req) for _ in range(5)])

    out = asyncio.run(go())
    assert fresh["n"] == 1
    assert sorted(o["gateway"]["cache"] for o in out) == ["merged"] * 4 + ["miss"]
