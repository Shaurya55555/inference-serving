# inference-serving: llama.cpp behind a FastAPI gateway

Serves Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) with llama.cpp's `llama-server` (CUDA build b12255, continuous
batching) behind a FastAPI gateway that adds:
- exact-match LRU response cache for deterministic (temperature 0) requests
- merging of identical in-flight requests (one upstream call, many waiters)
- a concurrency cap and a `/metrics` endpoint (hit rate, upstream calls, tokens)

`bench/bench.py` is an async load generator: p50/p95 latency, requests/s and tokens/s per concurrency level.

Hardware: RTX 4060 Laptop (8 GB), all layers offloaded, context 4096, 48 requests per row, 96 max new
tokens, temperature 0, one warm-up burst before timing, fresh process (and fresh gateway cache) per row.

## Results

**1. Batching: llama-server parallel slots (direct, no gateway)**

| slots | clients | tok/s | req/s | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 1 | 1 | 68.6 | 0.91 | 1106 | 1486 |
| 1 | 4 | 69.8 | 0.93 | 4324 | 4856 |
| 1 | 8 | 68.8 | 0.91 | 8668 | 9849 |
| 4 | 1 | 99.3 | 1.32 | 783 | 1022 |
| 4 | 4 | 141.5 | 1.90 | 2024 | 2712 |
| 4 | 8 | **175.6** | 2.31 | **3315** | 4152 |

One slot serializes requests: throughput is flat (~69 tok/s) and latency grows linearly with clients.
Four slots reach **2.55x the throughput and 2.6x lower p50 latency at 8 clients** (175.6 vs 68.8 tok/s;
3.3 s vs 8.7 s).

**2. Gateway cache (4 slots)**. "Repeat" is the share of requests that reuse an earlier prompt.

| repeat | clients | tok/s (effective) | req/s | p50 ms | cache hits / merged |
|---|---|---|---|---|---|
| 0% | 8 | 189.9 | 2.54 | 3031 | 0 / 0 |
| 50% | 1 | 176.5 | 2.14 | 564 | 18 / 0 |
| 50% | 4 | 273.9 | 3.49 | 1253 | 16 / 2 |
| 50% | 8 | **400.9** | **4.96** | **1589** | 10 / 8 |

With 0% repeats the gateway adds no measurable cost (189.9 tok/s vs 175.6 direct, within run-to-run noise).
With 50% repeated prompts at 8 clients it more than doubles effective throughput and halves p50 latency.
"Effective" tok/s counts tokens returned to clients, including those served from cache or merged.
Counters in `/metrics` also include the warm-up requests.

## Run
    llama-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -np 4 -c 4096 --port 8080
    UPSTREAM=http://127.0.0.1:8080 uvicorn gateway.app:app --port 8000
    python bench/bench.py --url http://127.0.0.1:8000 --label gateway --conc 8 --repeat 0.5

Raw per-run results are in `results/`. 

## Multi-provider failover
`gateway/router.py` routes each request through providers in priority order (`PROVIDERS_FILE`, see `gateway/providers.py`;
without it the gateway uses the single local server as before). Per provider: a token-bucket rate limit (`rpm`),
retries with jittered exponential backoff, and a circuit breaker (opens after 5 consecutive failures, one probe after a cooldown).

| outcome of one attempt | behavior |
|---|---|
| 2xx with a usable body | return it, `served_by` names the provider |
| 429, 500, 502, 503, 504, timeout, connection error, unusable 200 body | retry on the same provider with backoff, then fail over |
| 429 with `Retry-After` of 2 s or less | wait that long and retry; a longer one fails over immediately instead of blocking the request |
| 401, 403, 404 (misconfigured provider) | no retry, fail over |
| 400, 413, 422 (bad request) | returned to the client, never sent to another provider |
| circuit open, or rate-limit tokens exhausted | provider skipped without a network call |
| every provider failed | HTTP 503 with a per-provider list of what happened (no keys in it) |
| empty `messages`, `max_tokens` outside 1 to 4096, `temperature` outside 0 to 2 | HTTP 422 before any upstream call |

`min_max_tokens` per provider raises small token budgets for reasoning models (they spend the budget on hidden reasoning and return an empty answer).
`/metrics` includes per-provider attempts, failures, retries, skips and circuit state. API keys are read from env vars named in the file.
Cache and request merging are unchanged; a cache hit reports the provider that originally answered.

Tests: `pytest` runs 61 tests with scripted providers and a fake clock (retry, failover, Retry-After, every status class, circuit
open / half-open / recovery, rate limiting, 40 concurrent requests during an outage, HTTP layer through `httpx.MockTransport`).

### Failover load test (simulated outage, real providers)
`bench/failover_bench.py`: 42 requests at 0.7 req/s over 60 s, cache off, 48 max tokens. Primary = real local llama-server
(Qwen2.5-1.5B Q4_K_M, RTX 4060) behind `bench/fault_proxy.py`, which returns HTTP 503 for every request from t=20 s to t=40 s.
Fallback = Groq `openai/gpt-oss-20b` (free tier, configured at 30 requests/min). Three gateway configurations, one run each:
baseline (first provider only, one attempt, as the gateway behaved before this work), retry (first provider only, 3 attempts), failover (all features).

| arm | success before / during / after outage | overall success | p50 ms | p95 ms (successful requests) |
|---|---|---|---|---|
| baseline | 14/14, 0/14, 14/14 | 28/42 = 66.7% | 425 | 651 |
| retry only | 14/14, 0/14, 14/14 | 28/42 = 66.7% | 435 | 643 |
| **failover** | 14/14, **14/14**, 13/14 | **41/42 = 97.6%** | 587 | 991 |

During the outage the failover arm answered all 14 requests from Groq (p50 819 ms, p95 995 ms vs 478 ms local before). Retries alone did
not help because the outage outlasted them; the circuit breaker opened and later requests skipped the dead provider without waiting.
The single failure in the failover arm came just after recovery: the primary's circuit was still open and Groq's configured 30 requests/min
budget was used up (offered load was 42/min), so the rate limiter refused the request. That is the limiter working, and it shows the fallback must be sized for the load.
Raw files: `results/failover_*.json`. Caveats: the outage is simulated (the provider keeps running, only the proxy returns 503), one run per arm, 42 requests per arm,
and only the fast-failure outage kind was run (a hanging provider was not measured). A first baseline run was discarded because it overlapped with a
large package install on the same machine and stalled all processes; all three arms were re-run on an idle machine.

## Failure handling (single local provider, no `PROVIDERS_FILE`)
Same router with one provider: a dead upstream is retried, then reported as HTTP 503 listing the outcome; bad upstream statuses
follow the table above; several identical requests merged onto one failing call all receive the same error and nothing is cached;
sampled requests (temperature above 0) are never cached or merged. Not handled: streaming.

## Limits and honesty notes
- llama.cpp, not vLLM: vLLM does not run natively on Windows. The serving concepts measured here
  (continuous batching across slots, KV cache, quantization, request-level caching) transfer; the engine does not.
- Single machine, single model, one prompt family, 48 requests per row, one run each: treat differences
  under ~10% as noise (the first benchmark pass showed run-to-run swings of that size).
- The cache is exact-match only; no semantic caching, no eviction by token cost, no streaming support in the gateway.
- An earlier pass reused one gateway process across concurrency levels, which made later rows near-100% cache hits;
  rows above use a fresh gateway each, and that run is discarded.

Write-up: [The benchmark that lied](docs/the-benchmark-that-lied.md)
