"""Grounded answering over retrieved chunks with [p.N] citations."""
from __future__ import annotations

import re

from .hybrid import HybridIndex
from .rerank import llm_rerank
from .trace import Tracer
from .types import AnswerResult, Usage

SYSTEM_PROMPT = """You are doclens, answering questions about ONE uploaded document.

Rules:
- Use ONLY the context chunks below. Never use outside knowledge.
- Cite the page for every claim using the exact format [p.N].
- If the context does not contain the answer, reply starting with exactly:
  Not in the document.
"""

REFUSAL_THRESHOLD = 0.30
_CITE_RE = re.compile(r"\[p\.(\d+)\]")
REFUSAL_TEXT = ("Not in the document. The uploaded content doesn't appear to cover this — "
                "try rephrasing or uploading a more relevant document.")

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")


def answer_question(chat, chat_model: str, embedder, embed_model: str,
                    index: HybridIndex, question: str, k: int = 5,
                    history: list[dict] | None = None, *,
                    retrieval_mode: str = "hybrid_rerank", pool: int = 20,
                    tracer: Tracer | None = None) -> AnswerResult:
    """Embed, hybrid-retrieve, optionally LLM-rerank, then generate a grounded answer.

    Each stage runs inside its own `tracer` span (`embed`, `retrieve`, an
    optional `rerank`, `generate`). Refusal is decided on the max dense cosine
    of the retrieved pool *before* any rerank or generate call, so a
    low-confidence retrieval never spends an LLM call.

    Args:
        chat: object exposing `.complete(messages, model) -> (text, Usage)`.
        chat_model: model name passed to `chat.complete`.
        embedder: object exposing `.embed(texts, model) -> list[list[float]]`.
        embed_model: model name passed to `embedder.embed`.
        index: a `HybridIndex` populated with the document's chunks.
        question: the user's question.
        k: number of chunks the final answer is generated from.
        history: prior `{"question", "answer"}` turns threaded before the
            final context/question user turn.
        retrieval_mode: one of `RETRIEVAL_MODES`. `"dense"` retrieves densely
            only; `"hybrid"` and `"hybrid_rerank"` fuse dense+BM25 via RRF,
            with `"hybrid_rerank"` (default) additionally reordering the pool
            with one LLM call before truncating to `k`.
        pool: candidate pool size handed to `index.retrieve`.
        tracer: optional `Tracer` to record stage spans into. When omitted a
            fresh `Tracer` is created and discarded, so spans are always
            recorded even if the caller doesn't want them.

    Returns:
        `AnswerResult`. On refusal, `retrieved` is `candidates[:k]` (so the
        caller can still show what was searched) and `usage` is empty; on a
        completed answer, `retrieved` is the chunks actually used and `usage`
        sums the rerank call (if any) and the generate call.

    """
    tracer = tracer or Tracer()

    with tracer.span("embed") as sp:
        qvec = embedder.embed([question], embed_model)[0]
        sp.meta["dims"] = len(qvec)

    base_mode = "dense" if retrieval_mode == "dense" else "hybrid"
    with tracer.span("retrieve") as sp:
        candidates = index.retrieve(qvec, question, mode=base_mode, pool=pool)
        sp.meta.update({"mode": base_mode, "pool": pool, "candidates": len(candidates)})

    top_dense = max((c.components.get("dense_score", c.score) for c in candidates),
                    default=0.0)
    if not candidates or top_dense < REFUSAL_THRESHOLD:
        return AnswerResult(answer=REFUSAL_TEXT, citations=[], retrieved=candidates[:k],
                            refused=True, model=chat_model, usage=Usage())

    usage = Usage()
    if retrieval_mode == "hybrid_rerank":
        with tracer.span("rerank") as sp:
            final, ru = llm_rerank(chat, chat_model, question, candidates, top_k=k)
            usage = usage + ru
            sp.meta.update({"in": len(candidates), "out": len(final),
                            "input_tokens": ru.input_tokens, "output_tokens": ru.output_tokens})
    else:
        final = candidates[:k]

    with tracer.span("generate") as sp:
        context = "\n\n".join(f"[p.{r.chunk.page}] {r.chunk.text}" for r in final)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user",
                         "content": f"Context chunks:\n\n{context}\n\nQuestion: {question}"})
        text, gu = chat.complete(messages, chat_model)
        usage = usage + gu
        sp.meta.update({"input_tokens": gu.input_tokens, "output_tokens": gu.output_tokens})

    citations = sorted({int(m) for m in _CITE_RE.findall(text)})
    refused = text.strip().startswith("Not in the document")
    return AnswerResult(answer=text, citations=citations if not refused else [],
                        retrieved=final, refused=refused, model=chat_model, usage=usage)
