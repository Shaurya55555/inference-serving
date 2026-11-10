"""Load generator: latency percentiles and throughput at several concurrency levels.

python bench/bench.py --url http://127.0.0.1:8080 --label slots1 --repeat 0
--repeat R: fraction of requests that repeat an earlier prompt (to exercise the gateway cache).
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

TOPICS = ["compound interest", "how a hash map works", "the CAP theorem", "TCP slow start", "database indexes",
          "gradient descent", "the GIL in Python", "REST vs GraphQL", "a bloom filter", "consistent hashing",
          "OAuth 2.0", "the actor model", "B-trees", "rate limiting", "idempotency", "backpressure"]


def prompts(n, repeat, seed=0):
    rnd = random.Random(seed)
    base = [f"Explain {t} in three sentences, variant {i}." for i, t in enumerate(TOPICS * 4)]
    out = []
    for i in range(n):
        out.append(rnd.choice(out) if out and rnd.random() < repeat else base[i % len(base)])
    return out


async def one(client, url, prompt, max_tokens, path):
    t0 = time.time()
    r = await client.post(url + path, json={"messages": [{"role": "user", "content": prompt}],
                                            "max_tokens": max_tokens, "temperature": 0})
    r.raise_for_status()
    d = r.json()
    return time.time() - t0, d.get("usage", {}).get("completion_tokens", 0)


async def run(url, n, conc, repeat, max_tokens, path):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=300) as c:
        async def guarded(p):
            async with sem:
                return await one(c, url, p, max_tokens, path)
        t0 = time.time()
        res = await asyncio.gather(*[guarded(p) for p in prompts(n, repeat)])
        wall = time.time() - t0
    lat = sorted(r[0] for r in res)
    toks = sum(r[1] for r in res)
    return {"concurrency": conc, "requests": n, "wall_s": round(wall, 2),
            "req_per_s": round(n / wall, 2), "tok_per_s": round(toks / wall, 1),
            "p50_ms": round(1000 * statistics.median(lat)), "p95_ms": round(1000 * lat[int(0.95 * (len(lat) - 1))])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--path", default="/v1/chat/completions")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--repeat", type=float, default=0.0)
    ap.add_argument("--conc", default="1,4,8")
    a = ap.parse_args()

    rows = []
    for conc in [int(x) for x in a.conc.split(",")]:
        rows.append(asyncio.run(run(a.url, a.n, conc, a.repeat, a.max_tokens, a.path)))
        print(a.label, rows[-1], flush=True)
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    (out / f"{a.label}.json").write_text(json.dumps({"label": a.label, "repeat": a.repeat, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
