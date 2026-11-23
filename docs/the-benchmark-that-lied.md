# The benchmark that lied: a cache bug in my own load test

I put a small FastAPI gateway in front of llama.cpp's `llama-server` (Qwen2.5-1.5B, Q4_K_M, RTX 4060 Laptop with 8 GB) to add an
exact-match response cache, merging of identical in-flight requests and a concurrency cap. Then I benchmarked it. The first
numbers were wonderful and wrong, and the way I found out is the reason I trust the final table.

## The problem

I wanted two answers. First, how much does serving several sequences at once help (llama.cpp's `-np` parallel slots)? Second, what
does the gateway cost and what does it save? A load generator sent 48 requests at 1, 4 and 8 concurrent clients and reported
p50 and p95 latency, requests per second and tokens per second.

## The wrong result

For the gateway runs the later concurrency levels showed more than 5,000 tokens per second. The same GPU, measured directly, tops out
at about 175 tokens per second. A cache cannot make a GPU decode 30 times faster, so either the cache was hit far more than I intended,
or the number was meaningless.

The cause was my own setup. I started one gateway process and ran concurrency 1, then 4, then 8 against it. The benchmark prompts
repeat (16 topics, 64 prompt variants), and the gateway caches deterministic requests, so by the time the 4-client and 8-client rows ran,
almost every prompt was already cached. Those rows measured dictionary lookups. The first row was honest and the later ones were not,
which also meant the rows were not comparable to each other.

## What I tried

1. **A fresh gateway per row.** Every table row now starts a new gateway process, so each row starts with an empty cache.
2. **A warm-up outside the benchmark set.** A short burst of prompts that are not in the measured set, so the first timed request is not paying for a cold start.
3. **An explicit repeat fraction.** The load generator takes `--repeat R`, the share of requests that reuse an earlier prompt, so cache benefit is something I set and report, not an accident of the prompt list.
4. **Separate tables.** Real GPU throughput (the server driven directly, no gateway) and gateway-with-cache results are different tables. "Effective" tokens per second counts tokens returned to clients including cached ones, and is labelled that way.

I discarded the original run entirely rather than correcting it.

## What I measured

Batching, server driven directly (48 requests, 96 max new tokens, temperature 0):

| parallel slots | clients | tokens/s | p50 ms |
|---|---|---|---|
| 1 | 8 | 68.8 | 8668 |
| 4 | 8 | 175.6 | 3315 |

With one slot, requests are served one at a time: throughput stays near 69 tokens/s and latency grows with clients. Four slots reach
2.55x the throughput and about 2.6x lower p50 at 8 clients. The cost is that the context is split across slots (4096 / 4 = 1024 each).

Gateway with a controlled repeat share, 8 clients: at 0% repeats the gateway is within noise of direct (189.9 vs 175.6 tokens/s);
at 50% repeats effective throughput is 400.9 tokens/s and p50 falls to 1589 ms, with 10 cache hits and 8 merged requests in that run.
That is a real effect, roughly 2x, not 30x.

## What I did not do

This is llama.cpp on Windows, not vLLM, which does not run natively here. Each row is one run of 48 requests, so differences below
about 10 percent are noise, which I saw directly between early passes. The cache is exact-match and temperature 0 only. There is no
streaming through the gateway. The counters in `/metrics` include the warm-up requests.

## What I learned

1. When a result is too good, check what is being measured before you celebrate. An implausible number is a free bug report.
2. State carries between benchmark rows. If a component has memory (a cache, a connection pool, a JIT, a warmed GPU), isolate each row.
3. Decide the cache hit rate you are testing instead of discovering it.
4. Keep real capacity and effective capacity in different tables.
5. Later I added multi-provider failover to the same gateway and used the same discipline: one clean run per configuration, and a
   discarded run when a large package install on the same machine stalled every process during a test. Discarding a run you have a
   reason to distrust, and saying so, is what makes the remaining ones credible.
