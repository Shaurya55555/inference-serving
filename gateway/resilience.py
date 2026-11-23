"""Resilience primitives for the gateway. Pure logic with an injectable clock, so tests need no real waiting.

TokenBucket     per-provider rate limit (requests per minute) with a small burst allowance
CircuitBreaker  closed -> open after N consecutive failures -> half-open after a cooldown (one probe) -> closed
backoff_delay   exponential backoff with full jitter, or the provider's Retry-After when it is short enough
"""
import random
import time

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class TokenBucket:
    """`rpm` requests per minute. `burst` tokens are available at once (default: rpm / 6, at least 1)."""

    def __init__(self, rpm: float | None, burst: float | None = None, clock=time.monotonic):
        self.rpm, self.clock = rpm, clock
        self.capacity = None if not rpm else (burst if burst is not None else max(1.0, rpm / 6))
        self.tokens = self.capacity
        self.last = clock()

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rpm / 60.0)
        self.last = now

    def try_acquire(self) -> bool:
        if self.rpm is None or self.rpm <= 0:
            return True  # unlimited
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def seconds_until_token(self) -> float:
        if self.rpm is None or self.rpm <= 0:
            return 0.0
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) * 60.0 / self.rpm


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; lets exactly one probe through after `cooldown` seconds."""

    def __init__(self, threshold: int = 5, cooldown: float = 10.0, clock=time.monotonic, enabled: bool = True):
        self.threshold, self.cooldown, self.clock, self.enabled = threshold, cooldown, clock, enabled
        self.state, self.failures, self.opened_at, self.probe_inflight = CLOSED, 0, 0.0, False
        self.opened_count = 0

    def allow(self) -> bool:
        if not self.enabled or self.state == CLOSED:
            return True
        if self.state == OPEN:
            if self.clock() - self.opened_at < self.cooldown:
                return False
            self.state, self.probe_inflight = HALF_OPEN, False
        # half-open: a single probe request at a time
        if self.probe_inflight:
            return False
        self.probe_inflight = True
        return True

    def record_success(self):
        self.failures, self.probe_inflight = 0, False
        self.state = CLOSED

    def record_failure(self):
        if not self.enabled:
            return
        self.probe_inflight = False
        self.failures += 1
        if self.state == HALF_OPEN or self.failures >= self.threshold:
            if self.state != OPEN:
                self.opened_count += 1
            self.state, self.opened_at = OPEN, self.clock()

    def release_probe(self):
        """The probe was not actually sent (for example the rate limiter refused it): free the half-open slot."""
        self.probe_inflight = False


def backoff_delay(attempt: int, base: float = 0.25, cap: float = 4.0, retry_after: float | None = None,
                  max_retry_after: float = 2.0, rand=random.random) -> float | None:
    """Seconds to wait before retry number `attempt` (1-based).

    If the provider sent Retry-After: return it when it is at most `max_retry_after`, otherwise None, which tells the
    caller not to wait that long on this request and to fail over instead.
    """
    if retry_after is not None:
        return retry_after if retry_after <= max_retry_after else None
    return rand() * min(cap, base * (2 ** (attempt - 1)))
