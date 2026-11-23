"""Failover router: tries providers in priority order with retries, backoff, per-provider rate limits and circuit breakers.

For each provider, in order:
  1. skip it if its circuit is open (cheap, no network)
  2. skip it if its rate limiter has no token (the request goes to the next provider instead of queueing)
  3. try up to `max_attempts` times; retry only retryable failures, waiting with jittered exponential backoff
  4. a failure that is not retryable here (auth error, rate limit with a long Retry-After, retries used up) moves on

Outcome classes for one attempt:
  success       2xx with a JSON body that has `choices`         -> return
  client_error  400, 413, 422 (the request itself is bad)        -> return the error as is, no retry, no failover
  retryable     429, 500, 502, 503, 504, timeout, connection error, 2xx with an unusable body
  provider      401, 403, 404 and any other status               -> the provider is misconfigured: no retry, fail over

Everything that decides behaviour (clock, sleep, randomness, the HTTP call) is injected, so the failure scenarios in
tests/test_router.py run in milliseconds with no network.
"""
import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

from .providers import Provider, Result, http_call
from .resilience import CircuitBreaker, TokenBucket, backoff_delay

CLIENT_ERRORS = {400, 413, 422}
RETRYABLE = {429, 500, 502, 503, 504}


class ClientError(Exception):
    """The provider rejected the request itself; trying elsewhere would not help."""

    def __init__(self, status: int, body: str):
        super().__init__(body)
        self.status, self.body = status, body


class AllProvidersFailed(Exception):
    def __init__(self, tried: list[dict]):
        super().__init__("all providers failed")
        self.tried = tried


@dataclass
class ProviderStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    skipped_circuit_open: int = 0
    skipped_rate_limited: int = 0
    last_error: str = ""


@dataclass
class RouteInfo:
    provider: str
    attempts: int
    failovers: int
    tried: list[dict] = field(default_factory=list)


@dataclass
class RouterConfig:
    failover: bool = True              # False: only the first provider is ever used
    breaker: bool = True
    max_attempts: int | None = None    # overrides every provider's own max_attempts when set
    breaker_threshold: int = 5
    breaker_cooldown: float = 10.0
    deadline_s: float = 60.0           # stop starting new attempts after this long
    backoff_base: float = 0.25
    backoff_cap: float = 4.0
    max_retry_after: float = 2.0       # a longer Retry-After means "fail over", not "wait"


class FailoverRouter:
    def __init__(self, providers: list[Provider], config: RouterConfig | None = None, *, call=None,
                 client: httpx.AsyncClient | None = None, clock=time.monotonic, sleep=asyncio.sleep,
                 rand=random.random):
        if not providers:
            raise ValueError("at least one provider is required")
        self.cfg = config or RouterConfig()
        self.providers = providers if self.cfg.failover else providers[:1]
        self.clock, self.sleep, self.rand = clock, sleep, rand
        self.client = client or httpx.AsyncClient()
        self.call = call or (lambda p, body: http_call(self.client, p, body))
        self.buckets = {p.name: TokenBucket(p.rpm, clock=clock) for p in self.providers}
        self.breakers = {p.name: CircuitBreaker(self.cfg.breaker_threshold, self.cfg.breaker_cooldown, clock,
                                                enabled=self.cfg.breaker) for p in self.providers}
        self.sems = {p.name: asyncio.Semaphore(p.max_concurrency) for p in self.providers}
        self.stats = {p.name: ProviderStats() for p in self.providers}

    # ------------------------------------------------------------------ one attempt
    async def _attempt(self, p: Provider, body: dict):
        """Returns ('success', data) | ('client_error', (status, text)) | ('retryable', reason, retry_after)
        | ('provider', reason)."""
        st = self.stats[p.name]
        st.attempts += 1
        try:
            async with self.sems[p.name]:
                res: Result = await self.call(p, body)
        except httpx.TimeoutException:
            return "retryable", "timeout", None
        except httpx.TransportError as e:
            return "retryable", f"connection error ({type(e).__name__})", None
        if 200 <= res.status < 300:
            if res.data and res.data.get("choices"):
                return "success", res.data
            return "retryable", "unusable response body", None
        if res.status in CLIENT_ERRORS:
            return "client_error", (res.status, str(res.data)[:300])
        if res.status in RETRYABLE:
            ra = res.headers.get("retry-after") or res.headers.get("Retry-After")
            try:
                ra = float(ra) if ra is not None else None
            except ValueError:
                ra = None
            return "retryable", f"HTTP {res.status}", ra
        return "provider", f"HTTP {res.status}", None

    # ------------------------------------------------------------------ routing
    async def complete(self, body: dict) -> tuple[dict, RouteInfo]:
        start = self.clock()
        tried: list[dict] = []
        total_attempts = 0
        failovers = 0
        for p in self.providers:
            st, br, bucket = self.stats[p.name], self.breakers[p.name], self.buckets[p.name]
            if not br.allow():
                st.skipped_circuit_open += 1
                tried.append({"provider": p.name, "outcome": "skipped: circuit open"})
                failovers += 1
                continue
            attempts = self.cfg.max_attempts or p.max_attempts
            last_reason = ""
            outcome = "failed"
            for attempt in range(1, attempts + 1):
                if self.clock() - start > self.cfg.deadline_s:
                    last_reason, outcome = "request deadline exceeded", "deadline"
                    break
                if not bucket.try_acquire():
                    st.skipped_rate_limited += 1
                    br.release_probe()
                    last_reason, outcome = "rate limit reached", "skipped: rate limited"
                    break
                total_attempts += 1
                if attempt > 1:
                    st.retries += 1
                kind, *rest = await self._attempt(p, body)
                if kind == "success":
                    st.successes += 1
                    br.record_success()
                    tried.append({"provider": p.name, "outcome": "ok", "attempts": attempt})
                    return rest[0], RouteInfo(p.name, total_attempts, failovers, tried)
                if kind == "client_error":
                    br.release_probe()
                    raise ClientError(*rest[0])
                st.failures += 1
                last_reason = rest[0]
                st.last_error = last_reason
                br.record_failure()
                if kind == "provider":
                    outcome = f"failed: {last_reason}"
                    break
                # retryable: back off and try again on this provider, unless it just asked us to wait too long
                outcome = f"failed: {last_reason}"
                if attempt < attempts and br.allow():
                    delay = backoff_delay(attempt, self.cfg.backoff_base, self.cfg.backoff_cap, rest[1],
                                          self.cfg.max_retry_after, self.rand)
                    if delay is None:  # Retry-After is longer than we are willing to wait
                        outcome = f"failed: {last_reason}, Retry-After too long"
                        break
                    await self.sleep(delay)
                else:
                    break
            tried.append({"provider": p.name, "outcome": outcome if outcome != "failed" else last_reason})
            failovers += 1
        raise AllProvidersFailed(tried)

    # ------------------------------------------------------------------ introspection
    def snapshot(self) -> dict:
        out = {}
        for p in self.providers:
            s, b = self.stats[p.name], self.breakers[p.name]
            out[p.name] = {"circuit": b.state, "circuit_opened": b.opened_count, "attempts": s.attempts,
                           "successes": s.successes, "failures": s.failures, "retries": s.retries,
                           "skipped_circuit_open": s.skipped_circuit_open,
                           "skipped_rate_limited": s.skipped_rate_limited, "last_error": s.last_error}
        return out
