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

## Limits and honesty notes
- llama.cpp, not vLLM: vLLM does not run natively on Windows. The serving concepts measured here
  (continuous batching across slots, KV cache, quantization, request-level caching) transfer; the engine does not.
- Single machine, single model, one prompt family, 48 requests per row, one run each: treat differences
  under ~10% as noise (the first benchmark pass showed run-to-run swings of that size).
- The cache is exact-match only; no semantic caching, no eviction by token cost, no streaming support in the gateway.
- An earlier pass reused one gateway process across concurrency levels, which made later rows near-100% cache hits;
  rows above use a fresh gateway each, and that run is discarded.
