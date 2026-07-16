"""LLM listwise reranker: one chat call reorders a candidate pool to top-k."""
from __future__ import annotations

import json
import re

from .types import Retrieved, Usage

RERANK_INPUT_CHARS = 400

_ARRAY_RE = re.compile(r"\[[^\]]*\]")


def llm_rerank(chat, model: str, question: str,
                candidates: list[Retrieved], top_k: int) -> tuple[list[Retrieved], Usage]:
    """Reorder `candidates` to the final top-k using one listwise LLM call.

    Builds a single prompt numbering every candidate, asks the model to return
    a JSON array of passage numbers most-relevant-first, and maps that order
    back onto `candidates`. Robust by construction: any failure to get a
    usable order (the call raising, no JSON array in the reply, malformed
    JSON, or a non-list value) falls back to the original candidate order
    truncated to `top_k` — this function never raises.

    Args:
        chat: object exposing `.complete(messages, model) -> (text, Usage)`.
        model: model name passed through to `chat.complete`.
        question: the user question, embedded in the ranking prompt.
        candidates: candidate pool in incoming order (e.g. hybrid retrieval).
        top_k: number of candidates to return.

    Returns:
        `(reordered[:top_k], usage)`. On success each returned `Retrieved` is
        a fresh object (distinct from `candidates`, with a shallow-copied
        `components` dict) carrying `components["rerank_rank"]` (1-based). On
        fallback the original `candidates[:top_k]` objects are returned
        unmodified, so `rerank_rank` is absent. `usage` is `Usage()` when no
        call was made or the call itself raised before returning; otherwise
        it is the `Usage` the call reported, even if parsing failed after.

    """
    if len(candidates) <= 1:
        return candidates[:top_k], Usage()

    usage = Usage()
    try:
        prompt = _build_prompt(question, candidates)
        text, usage = chat.complete([{"role": "user", "content": prompt}], model)
        order = _parse_order(text, len(candidates))
    except Exception:
        return candidates[:top_k], usage

    out = []
    for rank, n in enumerate(order[:top_k], start=1):
        src = candidates[n - 1]
        item = Retrieved(chunk=src.chunk, score=src.score, components=dict(src.components))
        item.components["rerank_rank"] = rank
        out.append(item)
    return out, usage


def _build_prompt(question: str, candidates: list[Retrieved]) -> str:
    """Build the listwise ranking prompt (spec §3.4 shape).

    Args:
        question: the user question.
        candidates: candidate pool in incoming order; each is numbered
            `[1]..[N]` and shown as `(p.{page}) {text[:RERANK_INPUT_CHARS]}`.

    Returns:
        The full prompt string for a single `chat.complete` call.

    """
    lines = [
        "Rank the passages by how well they answer the question.",
        f"Question: {question}",
        "",
    ]
    for i, c in enumerate(candidates, start=1):
        snippet = c.chunk.text[:RERANK_INPUT_CHARS]
        lines.append(f"[{i}] (p.{c.chunk.page}) {snippet}")
    lines.append("")
    lines.append(
        "Return ONLY a JSON array of the passage numbers, most relevant first, "
        "each number exactly once. Example: [3, 1, 2]"
    )
    return "\n".join(lines)


def _parse_order(text: str, n: int) -> list[int]:
    """Parse a model reply into a full 1..n coverage order.

    Extracts the first `[...]` substring and `json.loads`s it, keeps ints in
    `1..n`, dedups preserving order, then appends any missing candidate
    numbers in original (ascending) order — this guarantees the result always
    covers every candidate exactly once (length always `n`), even if the
    model's list was empty, partial, or entirely out of range.

    Args:
        text: raw model reply text.
        n: number of candidates (valid passage numbers are `1..n`).

    Returns:
        A permutation of `1..n` as a list, most-relevant-first per the model
        where determinable, natural order for anything it missed.

    Raises:
        ValueError: no `[...]` substring found, or the parsed JSON value is
            not a list. (Malformed JSON inside the brackets propagates the
            `json.JSONDecodeError` raised by `json.loads`.) Callers should
            treat any exception from this function as "could not rerank".

    """
    match = _ARRAY_RE.search(text)
    if match is None:
        raise ValueError("no JSON array found in model reply")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("parsed JSON value is not a list")

    seen: set[int] = set()
    order: list[int] = []
    for item in data:
        if isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= n:
            if item not in seen:
                seen.add(item)
                order.append(item)
    for i in range(1, n + 1):
        if i not in seen:
            order.append(i)

    return order
