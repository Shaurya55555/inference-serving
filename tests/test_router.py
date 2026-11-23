"""Failure scenarios for the failover router. Providers are scripted fakes; the clock and sleep are fake too."""
import asyncio
import json

import httpx
import pytest

from gateway import providers as P
from gateway.router import AllProvidersFailed, ClientError, FailoverRouter, RouterConfig
from gateway.providers import Provider, Result

OK = Result(200, {"choices": [{"message": {"content": "hi"}}], "usage": {"completion_tokens": 2}}, {})
BODY = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8, "temperature": 0}


def status(code, retry_after=None):
    return Result(code, {"error": "x"}, {"retry-after": str(retry_after)} if retry_after is not None else {})


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class World:
    """Scripted providers. behaviour[name] is a callable(n_call) returning a Result or raising."""

    def __init__(self, **behaviour):
        self.behaviour, self.calls, self.sleeps, self.clock = behaviour, [], [], Clock()

    async def call(self, p, body):
        self.calls.append(p.name)
        n = self.calls.count(p.name)
        out = self.behaviour[p.name]
        out = out(n) if callable(out) else out
        if isinstance(out, Exception):
            raise out
        return out

    async def sleep(self, s):
        self.sleeps.append(s)
        self.clock.t += s

    def router(self, provs, **cfg):
        return FailoverRouter(provs, RouterConfig(**cfg), call=self.call, clock=self.clock, sleep=self.sleep,
                              rand=lambda: 0.5, client=httpx.AsyncClient())


def prov(name, **kw):
    return Provider(name=name, base_url="http://x", **kw)


def run(coro):
    return asyncio.run(coro)


def two(**kw):
    return [prov("local", **kw), prov("cloud", **kw)]


# ---------------------------------------------------------------- happy path and retries
def test_healthy_primary_serves_the_request():
    w = World(local=OK, cloud=OK)
    data, info = run(w.router(two()).complete(BODY))
    assert info.provider == "local" and info.attempts == 1 and info.failovers == 0 and w.calls == ["local"]


def test_transient_503_is_retried_on_the_same_provider():
    w = World(local=lambda n: status(503) if n == 1 else OK, cloud=OK)
    _, info = run(w.router(two()).complete(BODY))
    assert info.provider == "local" and info.attempts == 2 and len(w.sleeps) == 1 and "cloud" not in w.calls


def test_retries_exhausted_then_fail_over():
    w = World(local=status(503), cloud=OK)
    _, info = run(w.router(two()).complete(BODY))
    assert info.provider == "cloud" and info.failovers == 1 and w.calls == ["local", "local", "cloud"]


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("t"), httpx.ConnectError("c"), status(500), status(502),
                                     status(504), Result(200, {"nope": 1}, {})])
def test_every_retryable_failure_ends_on_the_fallback(failure):
    w = World(local=failure, cloud=OK)
    _, info = run(w.router(two()).complete(BODY))
    assert info.provider == "cloud"


def test_short_retry_after_is_waited_out():
    w = World(local=lambda n: status(429, 1) if n == 1 else OK, cloud=OK)
    _, info = run(w.router(two()).complete(BODY))
    assert info.provider == "local" and w.sleeps == [1.0]


def test_long_retry_after_fails_over_without_waiting():
    w = World(local=status(429, 60), cloud=OK)
    _, info = run(w.router(two()).complete(BODY))
    assert info.provider == "cloud" and w.sleeps == [] and w.calls == ["local", "cloud"]


@pytest.mark.parametrize("code", [401, 403, 404])
def test_auth_and_config_errors_fail_over_without_retrying(code):
    w = World(local=status(code), cloud=OK)
    _, info = run(w.router(two()).complete(BODY))
    assert info.provider == "cloud" and w.calls == ["local", "cloud"]


@pytest.mark.parametrize("code", [400, 413, 422])
def test_client_errors_are_returned_and_not_sent_elsewhere(code):
    w = World(local=status(code), cloud=OK)
    with pytest.raises(ClientError) as e:
        run(w.router(two()).complete(BODY))
    assert e.value.status == code and w.calls == ["local"]


def test_all_providers_failing_raises_with_a_per_provider_report():
    w = World(local=status(503), cloud=httpx.ReadTimeout("t"))
    with pytest.raises(AllProvidersFailed) as e:
        run(w.router(two()).complete(BODY))
    assert [t["provider"] for t in e.value.tried] == ["local", "cloud"]
    assert "HTTP 503" in e.value.tried[0]["outcome"] and "timeout" in e.value.tried[1]["outcome"]


def test_failover_disabled_only_ever_uses_the_first_provider():
    w = World(local=status(503), cloud=OK)
    with pytest.raises(AllProvidersFailed):
        run(w.router(two(), failover=False).complete(BODY))
    assert "cloud" not in w.calls


def test_max_attempts_override_applies_to_every_provider():
    w = World(local=status(503), cloud=OK)
    run(w.router(two(), max_attempts=1).complete(BODY))
    assert w.calls == ["local", "cloud"]


def test_request_deadline_stops_further_attempts():
    w = World(local=status(503), cloud=status(503))
    r = w.router(two(max_attempts=5), deadline_s=0.3, backoff_base=1.0, breaker=False)
    with pytest.raises(AllProvidersFailed) as e:
        run(r.complete(BODY))
    assert len(w.calls) < 10 and any("deadline" in t["outcome"] for t in e.value.tried)


# ---------------------------------------------------------------- circuit breaker behaviour through the router
def test_open_circuit_skips_the_dead_provider_without_calling_it():
    w = World(local=status(503), cloud=OK)
    r = w.router(two(max_attempts=1), breaker_threshold=3, breaker_cooldown=10)
    for _ in range(3):
        run(r.complete(BODY))
    assert r.breakers["local"].state == "open"
    before = w.calls.count("local")
    _, info = run(r.complete(BODY))
    assert info.provider == "cloud" and w.calls.count("local") == before
    assert r.stats["local"].skipped_circuit_open == 1


def test_circuit_probes_after_cooldown_and_closes_when_the_provider_recovers():
    state = {"up": False}
    w = World(local=lambda n: OK if state["up"] else status(503), cloud=OK)
    r = w.router(two(max_attempts=1), breaker_threshold=2, breaker_cooldown=10)
    for _ in range(2):
        run(r.complete(BODY))
    assert r.breakers["local"].state == "open"
    state["up"] = True
    _, info = run(r.complete(BODY))
    assert info.provider == "cloud"  # still cooling down
    w.clock.t += 11
    _, info = run(r.complete(BODY))
    assert info.provider == "local" and r.breakers["local"].state == "closed"


def test_failed_probe_keeps_the_circuit_open():
    w = World(local=status(503), cloud=OK)
    r = w.router(two(max_attempts=1), breaker_threshold=2, breaker_cooldown=10)
    for _ in range(2):
        run(r.complete(BODY))
    w.clock.t += 11
    before = w.calls.count("local")
    run(r.complete(BODY))
    assert w.calls.count("local") == before + 1 and r.breakers["local"].state == "open"


def test_breaker_disabled_keeps_hitting_a_dead_provider():
    w = World(local=status(503), cloud=OK)
    r = w.router(two(max_attempts=1), breaker=False)
    for _ in range(10):
        run(r.complete(BODY))
    assert w.calls.count("local") == 10


# ---------------------------------------------------------------- rate limiting
def test_rate_limited_provider_is_skipped_for_the_next_one():
    w = World(local=OK, cloud=OK)
    provs = [prov("local", rpm=60), prov("cloud")]  # burst capacity = max(1, 60/6) = 10 tokens
    r = w.router(provs)
    served = [run(r.complete(BODY))[1].provider for _ in range(12)]
    assert served[:10] == ["local"] * 10 and served[10:] == ["cloud", "cloud"]
    assert r.stats["local"].skipped_rate_limited == 2


def test_rate_limit_recovers_over_time():
    w = World(local=OK, cloud=OK)
    r = w.router([prov("local", rpm=6), prov("cloud")])  # burst 1, one token per 10 s
    assert run(r.complete(BODY))[1].provider == "local"
    assert run(r.complete(BODY))[1].provider == "cloud"
    w.clock.t += 10
    assert run(r.complete(BODY))[1].provider == "local"


def test_refused_probe_does_not_leave_the_breaker_stuck_half_open():
    w = World(local=status(503), cloud=OK)
    r = w.router([prov("local", rpm=60, max_attempts=1), prov("cloud")], breaker_threshold=2, breaker_cooldown=10)
    for _ in range(2):
        run(r.complete(BODY))
    r.buckets["local"].tokens = 0  # no rate-limit tokens when the cooldown ends
    w.clock.t += 11
    run(r.complete(BODY))
    assert r.breakers["local"].probe_inflight is False


# ---------------------------------------------------------------- concurrency and metrics
def test_outage_under_concurrent_load_all_requests_succeed_via_fallback():
    w = World(local=httpx.ConnectError("down"), cloud=OK)
    r = w.router(two(), breaker_threshold=3)

    async def go():
        return await asyncio.gather(*[r.complete(BODY) for _ in range(40)])

    out = run(go())
    assert len(out) == 40 and all(info.provider == "cloud" for _, info in out)
    assert r.breakers["local"].state == "open"
    assert r.stats["local"].attempts < 40  # the breaker stopped most of the traffic to the dead provider


def test_snapshot_reports_counters_and_circuit_state():
    w = World(local=status(503), cloud=OK)
    r = w.router(two(max_attempts=1), breaker_threshold=2)
    for _ in range(3):
        run(r.complete(BODY))
    snap = r.snapshot()
    assert snap["local"]["circuit"] == "open" and snap["local"]["failures"] == 2
    assert snap["cloud"]["successes"] == 3 and snap["local"]["skipped_circuit_open"] == 1


# ---------------------------------------------------------------- the real HTTP layer (httpx MockTransport)
def test_http_call_sends_key_and_model_and_key_never_appears_in_reports(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(503, json={"error": "busy"})

    monkeypatch.setenv("TEST_KEY_ENV", "sk-secret-123")
    p = Provider(name="cloud", base_url="https://api.example.com/openai/v1", model="m1", api_key_env="TEST_KEY_ENV")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = FailoverRouter([p], RouterConfig(max_attempts=1), client=client)
    with pytest.raises(AllProvidersFailed) as e:
        run(r.complete(BODY))
    assert seen["url"] == "https://api.example.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret-123" and seen["payload"]["model"] == "m1"
    assert "sk-secret" not in json.dumps(e.value.tried) and "sk-secret" not in json.dumps(r.snapshot())


def test_min_max_tokens_raises_a_small_budget_but_never_lowers_a_large_one():
    sent = []

    def handler(request: httpx.Request):
        sent.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json=OK.data)

    p = Provider(name="reasoner", base_url="http://x", min_max_tokens=256)
    r = FailoverRouter([p], client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    run(r.complete({**BODY, "max_tokens": 48}))
    run(r.complete({**BODY, "max_tokens": 1000}))
    assert sent == [256, 1000]


def test_local_provider_url_gets_the_v1_prefix():
    assert prov("l").chat_url() == "http://x/v1/chat/completions"
    assert Provider(name="g", base_url="https://api.groq.com/openai/v1").chat_url().endswith("/v1/chat/completions")
    assert Provider(name="m", base_url="https://generativelanguage.googleapis.com/v1beta/openai").chat_url() \
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_non_json_error_body_is_survivable():
    def handler(request):
        return httpx.Response(502, content=b"<html>bad gateway</html>")

    r = FailoverRouter([prov("a", max_attempts=1)], client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(AllProvidersFailed):
        run(r.complete(BODY))


# ---------------------------------------------------------------- configuration
def test_load_providers_from_file_and_default(tmp_path, monkeypatch):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"name": "a", "base_url": "http://a"}, {"name": "b", "base_url": "http://b", "enabled": False},
                             {"name": "c", "base_url": "http://c", "rpm": 20}]))
    assert [p.name for p in P.load_providers(str(f))] == ["a", "c"]
    assert P.load_providers(None, upstream="http://u")[0].base_url == "http://u"


def test_duplicate_provider_names_and_empty_file_are_rejected(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"name": "a", "base_url": "x"}, {"name": "a", "base_url": "y"}]))
    with pytest.raises(ValueError, match="unique"):
        P.load_providers(str(f))
    f.write_text("[]")
    with pytest.raises(ValueError, match="non-empty"):
        P.load_providers(str(f))
