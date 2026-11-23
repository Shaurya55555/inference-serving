"""Fault-injecting reverse proxy placed in front of the real local llama-server, so an outage can be simulated on demand.

python bench/fault_proxy.py --target http://127.0.0.1:8080 --port 18081
curl -X POST "http://127.0.0.1:18081/__fault?mode=down"     # every request now gets HTTP 503 immediately
curl -X POST "http://127.0.0.1:18081/__fault?mode=hang"     # requests are accepted and never answered (until the client gives up)
curl -X POST "http://127.0.0.1:18081/__fault?mode=ok"       # healthy again

This is a simulation of failure, not a measurement of a real outage: the provider itself keeps running, only the proxy lies.
"""
import argparse
import asyncio

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

app = FastAPI()
state = {"mode": "ok", "target": "http://127.0.0.1:8080", "served": 0, "faulted": 0}
client = httpx.AsyncClient(timeout=300)


@app.post("/__fault")
async def set_fault(mode: str):
    if mode not in ("ok", "down", "hang"):
        return Response("mode must be ok, down or hang", status_code=422)
    state["mode"] = mode
    return dict(state)


@app.get("/__fault")
async def get_fault():
    return dict(state)


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    if state["mode"] == "down":
        state["faulted"] += 1
        return Response('{"error": "injected outage"}', status_code=503, media_type="application/json")
    if state["mode"] == "hang":
        state["faulted"] += 1
        await asyncio.sleep(3600)
    r = await client.request(request.method, f"{state['target']}/{path}", content=await request.body(),
                             headers={"content-type": "application/json"})
    state["served"] += 1
    return Response(r.content, status_code=r.status_code, media_type="application/json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://127.0.0.1:8080")
    ap.add_argument("--port", type=int, default=18081)
    a = ap.parse_args()
    state["target"] = a.target
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
