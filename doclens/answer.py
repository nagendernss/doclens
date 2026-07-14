"""Grounded answering over retrieved chunks with [p.N] citations."""
from __future__ import annotations

import re

from .index import VectorIndex
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


def answer_question(chat, chat_model: str, embedder, embed_model: str,
                    index: VectorIndex, question: str, k: int = 5,
                    history: list[dict] | None = None) -> AnswerResult:
    qvec = embedder.embed([question], embed_model)[0]
    retrieved = index.search(qvec, k=k)
    if not retrieved or retrieved[0].score < REFUSAL_THRESHOLD:
        return AnswerResult(answer=REFUSAL_TEXT, citations=[], retrieved=retrieved,
                            refused=True, model=chat_model, usage=Usage())
    context = "\n\n".join(f"[p.{r.chunk.page}] {r.chunk.text}" for r in retrieved)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user",
                     "content": f"Context chunks:\n\n{context}\n\nQuestion: {question}"})
    text, usage = chat.complete(messages, chat_model)
    citations = sorted({int(m) for m in _CITE_RE.findall(text)})
    refused = text.strip().startswith("Not in the document")
    return AnswerResult(answer=text, citations=citations if not refused else [],
                        retrieved=retrieved, refused=refused, model=chat_model, usage=usage)
