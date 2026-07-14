import json

import httpx
import pytest

from doclens.providers._http import post_with_retry
from doclens.providers.gemini import GeminiChat, GeminiEmbedder
from doclens.providers.registry import MissingKeyError, get_chat, get_embedder


def test_retry_then_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resp = post_with_retry(client, "https://x/y", headers={}, json={})
    assert resp.status_code == 200 and calls["n"] == 3 and sleeps == [2, 4]


def test_4xx_fails_fast():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400)))
    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(client, "https://x/y", headers={}, json={}).raise_for_status()


def test_registry_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        get_chat("gemini-3.1-flash-lite")
    provider, model = get_chat("gemini-3.1-flash-lite", api_key="k")
    assert model == "gemini-3.1-flash-lite" and provider.api_key == "k"


def test_chat_translation():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hi [p.2]"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}})

    chat = GeminiChat(api_key="K", client=httpx.Client(transport=httpx.MockTransport(handler)))
    text, usage = chat.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        "gemini-3.1-flash-lite")
    assert text == "hi [p.2]" and usage.input_tokens == 5
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "q"}]}]


def test_embedder_batches_and_parses():
    batches = []

    def handler(request):
        body = json.loads(request.content)
        batches.append(len(body["requests"]))
        return httpx.Response(200, json={"embeddings": [
            {"values": [0.1, 0.2]} for _ in body["requests"]]})

    emb = GeminiEmbedder(api_key="K", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = emb.embed([f"t{i}" for i in range(130)], "gemini-embedding-001")
    assert len(out) == 130 and out[0] == [0.1, 0.2]
    assert batches == [64, 64, 2]


def test_get_embedder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "e")
    provider, model = get_embedder()
    assert model == "gemini-embedding-001"


def test_chat_multi_turn_roles():
    """Multi-turn conversation with alternating user/assistant roles."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "response"}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}})

    chat = GeminiChat(api_key="K", client=httpx.Client(transport=httpx.MockTransport(handler)))
    text, usage = chat.complete(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}
        ],
        "gemini-3.1-flash-lite")

    # Verify systemInstruction is set
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "s"

    # Verify contents has alternating user/model roles
    contents = seen["body"]["contents"]
    assert len(contents) == 3
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert [c["parts"][0]["text"] for c in contents] == ["q1", "a1", "q2"]
