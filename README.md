# inference-serving: llama.cpp behind a FastAPI gateway

Status: **code written, benchmarks pending. Numbers below are filled in once runs finish.**

Serves Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) with llama.cpp's `llama-server` (CUDA, continuous batching)
behind a FastAPI gateway that adds:
- exact-match LRU response cache for deterministic (temperature 0) requests
- merging of identical in-flight requests (one upstream call, many waiters)
- a concurrency cap and a `/metrics` endpoint (hit rate, upstream calls, tokens)

`bench/bench.py` is an async load generator reporting p50/p95 latency, requests/s and tokens/s at several
concurrency levels.

## Experiments
1. `llama-server -np 1` (one slot) vs `-np 4` (four parallel slots, continuous batching), concurrency 1/4/8
2. Gateway with 0% vs 50% repeated prompts (cache effect)

## Results
| setup | concurrency | req/s | tok/s | p50 ms | p95 ms |
|---|---|---|---|---|---|
| TBD | | | | | |

## Run
    llama-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -np 4 -c 4096 --port 8080
    UPSTREAM=http://127.0.0.1:8080 uvicorn gateway.app:app --port 8000
    python bench/bench.py --url http://127.0.0.1:8000 --label gateway --repeat 0.5

Why llama.cpp and not vLLM: vLLM does not run natively on Windows; llama.cpp gives the same serving
concepts (batching, KV cache, quantization) on the available hardware (RTX 4060, 8 GB).
