"""Gemini chat + embeddings over raw REST."""
from __future__ import annotations

import httpx

from ..types import Usage
from ._http import post_with_retry

BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_BATCH = 64


class GeminiChat:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def complete(self, messages: list[dict], model: str) -> tuple[str, Usage]:
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        resp = post_with_retry(self._client, f"{BASE}/models/{model}:generateContent",
                               headers={"x-goog-api-key": self.api_key}, json=body)
        resp.raise_for_status()
        data = resp.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        meta = data.get("usageMetadata", {})
        return text, Usage(meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0))


class GeminiEmbedder:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            body = {"requests": [{"model": f"models/{model}",
                                   "content": {"parts": [{"text": t}]}} for t in batch]}
            resp = post_with_retry(self._client, f"{BASE}/models/{model}:batchEmbedContents",
                                   headers={"x-goog-api-key": self.api_key}, json=body)
            resp.raise_for_status()
            out.extend(e["values"] for e in resp.json()["embeddings"])
        return out
