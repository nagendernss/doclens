from __future__ import annotations

import os

CHAT_MODELS = {
    "gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite", 0.0, 0.0),
    "gemini-3.5-flash": ("gemini", "gemini-3.5-flash", 0.0, 0.0),
}
EMBED_MODELS = {"gemini-embedding-001": ("gemini", "gemini-embedding-001")}
ENV_KEY = "GEMINI_API_KEY"


class UnknownModelError(Exception):
    pass


class MissingKeyError(Exception):
    pass


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get(ENV_KEY)
    if not key:
        raise MissingKeyError(f"needs env {ENV_KEY}")
    return key


def available_chat_models() -> list[str]:
    return list(CHAT_MODELS) if os.environ.get(ENV_KEY) else []


def get_chat(model: str, api_key: str | None = None):
    if model not in CHAT_MODELS:
        raise UnknownModelError(f"unknown chat model {model!r}")
    from .gemini import GeminiChat
    return GeminiChat(api_key=_key(api_key)), CHAT_MODELS[model][1]


def get_embedder(model: str = "gemini-embedding-001", api_key: str | None = None):
    if model not in EMBED_MODELS:
        raise UnknownModelError(f"unknown embedding model {model!r}")
    from .gemini import GeminiEmbedder
    return GeminiEmbedder(api_key=_key(api_key)), EMBED_MODELS[model][1]
