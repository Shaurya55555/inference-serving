"""Provider configuration for the gateway. Every provider speaks the OpenAI chat-completions protocol:
llama.cpp's llama-server, Groq, Gemini's OpenAI-compatible endpoint, OpenRouter and so on.

PROVIDERS_FILE points at a JSON list; without it the gateway uses the single local server in UPSTREAM, which is the
behaviour it had before failover existed. API keys are never stored in the file: `api_key_env` names the environment
variable to read, and the key is only ever sent in the Authorization header.

[
  {"name": "local",  "base_url": "http://127.0.0.1:8080",                    "model": "qwen2.5-1.5b", "timeout_s": 60},
  {"name": "groq",   "base_url": "https://api.groq.com/openai/v1",           "model": "llama-3.1-8b-instant",
   "api_key_env": "GROQ_API_KEY", "rpm": 25, "timeout_s": 30},
  {"name": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
   "model": "gemini-2.5-flash-lite", "api_key_env": "GEMINI_API_KEY", "rpm": 10}
]
Order is priority: the first healthy provider with rate-limit headroom gets the request.
"""
import json
import os
from dataclasses import dataclass, field

import httpx


@dataclass
class Provider:
    name: str
    base_url: str
    model: str = ""
    api_key_env: str = ""
    rpm: float | None = None          # requests per minute this gateway may send; None means unlimited
    timeout_s: float = 60.0
    max_attempts: int = 2             # tries on this provider (first try plus retries) before failing over
    max_concurrency: int = 8
    min_max_tokens: int = 0           # raise a request's max_tokens to at least this; reasoning models spend part of
                                      # the budget on hidden reasoning and return an empty answer when it is too small
    enabled: bool = True
    extra_headers: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "") if self.api_key_env else ""

    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base + ("/chat/completions" if base.endswith(("/v1", "/openai", "/v1beta/openai")) else "/v1/chat/completions")


def load_providers(path: str | None = None, upstream: str | None = None) -> list[Provider]:
    path = path or os.getenv("PROVIDERS_FILE", "")
    if path:
        raw = json.load(open(path, encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("PROVIDERS_FILE must contain a non-empty JSON list")
        provs = [Provider(**item) for item in raw]
        names = [p.name for p in provs]
        if len(set(names)) != len(names):
            raise ValueError("provider names must be unique")
        return [p for p in provs if p.enabled]
    return [Provider(name="local", base_url=upstream or os.getenv("UPSTREAM", "http://127.0.0.1:8080"), timeout_s=300)]


@dataclass
class Result:
    status: int
    data: dict | None
    headers: dict


async def http_call(client: httpx.AsyncClient, p: Provider, body: dict) -> Result:
    """One HTTP attempt. Raises httpx.TimeoutException / httpx.TransportError; never returns for those cases."""
    headers = {"Content-Type": "application/json", **p.extra_headers}
    if p.api_key:
        headers["Authorization"] = f"Bearer {p.api_key}"
    payload = {**body, **({"model": p.model} if p.model else {})}
    if p.min_max_tokens and payload.get("max_tokens", 0) < p.min_max_tokens:
        payload["max_tokens"] = p.min_max_tokens
    r = await client.post(p.chat_url(), json=payload, headers=headers, timeout=p.timeout_s)
    try:
        data = r.json()
    except ValueError:
        data = None
    return Result(r.status_code, data, dict(r.headers))
