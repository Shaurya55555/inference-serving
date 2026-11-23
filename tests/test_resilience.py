"""Token bucket, circuit breaker and backoff, driven by a fake clock (no real waiting)."""
from gateway.resilience import CLOSED, HALF_OPEN, OPEN, CircuitBreaker, TokenBucket, backoff_delay


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


# ---------------------------------------------------------------- token bucket
def test_unlimited_bucket_always_allows():
    b = TokenBucket(None)
    assert all(b.try_acquire() for _ in range(1000))


def test_bucket_allows_a_burst_then_refuses():
    c = Clock()
    b = TokenBucket(60, burst=3, clock=c)  # 1 request per second, burst of 3
    assert [b.try_acquire() for _ in range(5)] == [True, True, True, False, False]


def test_bucket_refills_with_time_and_never_exceeds_capacity():
    c = Clock()
    b = TokenBucket(60, burst=2, clock=c)
    assert b.try_acquire() and b.try_acquire() and not b.try_acquire()
    c.advance(1.0)
    assert b.try_acquire() and not b.try_acquire()
    c.advance(3600)
    assert [b.try_acquire() for _ in range(3)] == [True, True, False]


def test_seconds_until_token():
    c = Clock()
    b = TokenBucket(30, burst=1, clock=c)  # one token every 2 seconds
    assert b.try_acquire() and not b.try_acquire()
    assert abs(b.seconds_until_token() - 2.0) < 1e-6
    c.advance(1.0)
    assert abs(b.seconds_until_token() - 1.0) < 1e-6


# ---------------------------------------------------------------- circuit breaker
def breaker(**kw):
    c = Clock()
    return c, CircuitBreaker(threshold=3, cooldown=10, clock=c, **kw)


def test_opens_after_consecutive_failures_only():
    c, b = breaker()
    b.record_failure(); b.record_failure()
    b.record_success()  # a success resets the streak
    b.record_failure(); b.record_failure()
    assert b.state == CLOSED and b.allow()
    b.record_failure()
    assert b.state == OPEN and not b.allow() and b.opened_count == 1


def test_stays_open_during_cooldown_then_allows_one_probe():
    c, b = breaker()
    for _ in range(3):
        b.record_failure()
    c.advance(9.9)
    assert not b.allow()
    c.advance(0.2)
    assert b.allow() and b.state == HALF_OPEN
    assert not b.allow()  # only one probe at a time


def test_successful_probe_closes_the_circuit():
    c, b = breaker()
    for _ in range(3):
        b.record_failure()
    c.advance(11)
    assert b.allow()
    b.record_success()
    assert b.state == CLOSED and b.allow() and b.allow()


def test_failed_probe_reopens_and_restarts_the_cooldown():
    c, b = breaker()
    for _ in range(3):
        b.record_failure()
    c.advance(11)
    assert b.allow()
    b.record_failure()
    assert b.state == OPEN and not b.allow()
    c.advance(9)
    assert not b.allow()
    c.advance(2)
    assert b.allow()


def test_released_probe_frees_the_half_open_slot():
    c, b = breaker()
    for _ in range(3):
        b.record_failure()
    c.advance(11)
    assert b.allow()
    b.release_probe()
    assert b.allow()


def test_disabled_breaker_never_opens():
    c, b = breaker(enabled=False)
    for _ in range(50):
        b.record_failure()
    assert b.state == CLOSED and b.allow()


# ---------------------------------------------------------------- backoff
def test_backoff_grows_exponentially_and_is_capped():
    top = lambda: 0.999999  # noqa: E731  jitter at its maximum
    d = [backoff_delay(n, base=0.25, cap=4.0, rand=top) for n in range(1, 8)]
    assert d[0] < d[1] < d[2] < d[3]
    assert max(d) <= 4.0


def test_backoff_has_full_jitter():
    assert backoff_delay(3, rand=lambda: 0.0) == 0.0
    assert backoff_delay(3, rand=lambda: 0.5) == 0.5 * 1.0  # base 0.25 * 2**2 = 1.0


def test_short_retry_after_is_honoured_and_long_one_means_fail_over():
    assert backoff_delay(1, retry_after=1.5, max_retry_after=2.0) == 1.5
    assert backoff_delay(1, retry_after=30, max_retry_after=2.0) is None
