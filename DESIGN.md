# Design decisions and likely questions: inference-serving

**Why llama.cpp and not vLLM?**
vLLM does not run natively on Windows, and this machine has an 8 GB laptop GPU. llama.cpp demonstrates the same
serving ideas: continuous batching across parallel slots, a KV cache per slot, and 4-bit quantization (Q4_K_M).
I do not claim vLLM experience from this project.

**What does `-np 4` change?**
With one slot requests are served one at a time: throughput stayed at about 69 tok/s while p50 latency grew from
1.1 s to 8.7 s as clients went from 1 to 8. With four slots the server decodes several sequences together, reaching
175.6 tok/s (2.55x) and 3.3 s p50 at 8 clients. The context is split across slots (4096 / 4 = 1024 each), which is the cost.

**Why a gateway in front?**
It is where production concerns live without touching the engine: caching, request merging, a concurrency cap and
metrics. The cache is exact-match and only for temperature 0, since sampled outputs are not repeatable.

**What is request merging?**
If an identical deterministic request is already running, later copies wait on that call's result instead of
starting another. Under 8 clients with 50% repeats this removed 8 upstream calls in one run.

**Why report "effective" tokens per second?**
Cached or merged responses return tokens without GPU work. Effective tok/s is what clients see; real GPU
throughput is the direct-server table. The two are labelled separately in the README.

**A methodology bug I caught:** my first run kept one gateway process across concurrency levels, so later rows
were almost all cache hits and showed 5,000+ tok/s. I discarded that run, added a warm-up and used a fresh
gateway per row.

**What would you do next?**
Streaming responses through the gateway, time-to-first-token measurement, speculative decoding or a larger
quantization comparison, and prefix caching for shared system prompts.
