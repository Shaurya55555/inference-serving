"""Load test through the gateway while the primary provider is taken down and brought back.

Timeline (default): 60 s of open-loop traffic at 0.7 requests/s. The primary is healthy for 20 s, down for 20 s,
healthy again for 20 s. Latency, success rate and the provider that served each request are recorded per phase.

Needs: llama-server on :8080, bench/fault_proxy.py on :18081 (in front of it), and for the failover arm a
providers file whose second entry is a real free API provider. See README "Failover load test" for the exact commands.

python bench/failover_bench.py --arm baseline  --providers bench/providers.failover.json
python bench/failover_bench.py --arm retry     --providers bench/providers.failover.json
python bench/failover_bench.py --arm failover  --providers bench/providers.failover.json

arms:   baseline  first provider only, one attempt, no breaker (the gateway as it was before failover existed)
        retry     first provider only, 3 attempts with backoff, no breaker
        failover  all providers, retries, rate limits and circuit breaker (the defaults)
"""
import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"baseline": {"FAILOVER": "0", "BREAKER": "0", "MAX_ATTEMPTS": "1"},
        "retry": {"FAILOVER": "0", "BREAKER": "0", "MAX_ATTEMPTS": "3"},
        "failover": {"FAILOVER": "1", "BREAKER": "1"}}
TOPICS = ["compound interest", "how a hash map works", "the CAP theorem", "TCP slow start", "database indexes",
          "gradient descent", "the GIL in Python", "REST vs GraphQL", "a bloom filter", "consistent hashing"]


def pct(xs, q):
    xs = sorted(xs)
    return round(1000 * xs[int(q * (len(xs) - 1))]) if xs else None


async def fire(client, url, i, t_start, max_tokens, rec):
    t0 = time.time()
    body = {"messages": [{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} in one sentence (case {i})."}],
            "max_tokens": max_tokens, "temperature": 0}
    out = {"i": i, "offset": round(t0 - t_start, 2), "status": None, "served_by": None, "error": None}
    try:
        r = await client.post(url + "/v1/chat/completions", json=body)
        out["status"] = r.status_code
        if r.status_code == 200:
            out["served_by"] = r.json().get("served_by", "local")
        else:
            out["error"] = r.text[:120]
    except httpx.HTTPError as e:
        out["error"] = type(e).__name__
    out["latency"] = round(time.time() - t0, 3)
    rec.append(out)


async def run(a):
    env = {**os.environ, **ARMS[a.arm], "PROVIDERS_FILE": str(Path(a.providers).resolve()), "CACHE_SIZE": "0",
           "BREAKER_THRESHOLD": "3", "BREAKER_COOLDOWN": "5", "DEADLINE_S": "30"}
    gw = subprocess.Popen([sys.executable, "-m", "uvicorn", "gateway.app:app", "--port", str(a.gw_port),
                           "--log-level", "warning"], cwd=ROOT, env=env)
    url = f"http://127.0.0.1:{a.gw_port}"
    ctl = f"http://127.0.0.1:{a.fault_port}"
    rec: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=a.client_timeout) as c:
            for _ in range(60):  # wait for the gateway
                try:
                    if (await c.get(url + "/metrics")).status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.5)
            await c.post(ctl + "/__fault", params={"mode": "ok"})
            await c.post(url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "warm up"}],
                                                             "max_tokens": 4, "temperature": 0.1})
            n = int(a.duration * a.rate)
            t_start = time.time()
            tasks, phase_state = [], "ok"
            for i in range(n):
                due = t_start + i / a.rate
                now = time.time()
                if due > now:
                    await asyncio.sleep(due - now)
                el = time.time() - t_start
                want = "down" if a.down_from <= el < a.down_to else "ok"
                if want != phase_state:
                    await c.post(ctl + "/__fault", params={"mode": a.fault_mode if want == "down" else "ok"})
                    phase_state = want
                tasks.append(asyncio.create_task(fire(c, url, i, t_start, a.max_tokens, rec)))
            await asyncio.gather(*tasks)
            await c.post(ctl + "/__fault", params={"mode": "ok"})
            metrics = (await c.get(url + "/metrics")).json()
    finally:
        gw.terminate()
        gw.wait(timeout=20)

    def phase(r):
        return "before" if r["offset"] < a.down_from else ("during" if r["offset"] < a.down_to else "after")

    summary = {"arm": a.arm, "requests": len(rec), "rate_rps": a.rate, "duration_s": a.duration,
               "outage_window_s": [a.down_from, a.down_to], "fault_mode": a.fault_mode, "phases": {}}
    for name in ("before", "during", "after", "all"):
        rs = [r for r in rec if name == "all" or phase(r) == name]
        ok = [r for r in rs if r["status"] == 200]
        by = {}
        for r in ok:
            by[r["served_by"]] = by.get(r["served_by"], 0) + 1
        summary["phases"][name] = {"n": len(rs), "success": len(ok),
                                   "success_rate": round(len(ok) / len(rs), 3) if rs else None,
                                   "p50_ms": pct([r["latency"] for r in ok], 0.5),
                                   "p95_ms": pct([r["latency"] for r in ok], 0.95),
                                   "p50_ms_all_requests": pct([r["latency"] for r in rs], 0.5),
                                   "p95_ms_all_requests": pct([r["latency"] for r in rs], 0.95),
                                   "served_by": by}
    summary["gateway_providers"] = metrics.get("providers")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"failover_{a.arm}.json").write_text(json.dumps({"summary": summary, "requests": rec}, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--providers", required=True)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--rate", type=float, default=0.7, help="requests per second (open loop)")
    ap.add_argument("--down-from", type=float, default=20)
    ap.add_argument("--down-to", type=float, default=40)
    ap.add_argument("--fault-mode", choices=["down", "hang"], default="down")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--gw-port", type=int, default=8000)
    ap.add_argument("--fault-port", type=int, default=18081)
    ap.add_argument("--client-timeout", type=float, default=60)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
