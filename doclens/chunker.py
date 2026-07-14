"""Sliding-window chunking with sentence snapping and page metadata."""
from __future__ import annotations

from .types import Chunk, Document

_BOUNDARIES = (". ", "? ", "! ", "\n")


def _snap_end(text: str, end: int, window: int = 80) -> int:
    if end >= len(text):
        return len(text)
    best = -1
    lo = max(0, end - window)
    for mark in _BOUNDARIES:
        idx = text.rfind(mark, lo, end)
        if idx > best:
            best = idx + len(mark.rstrip()) or idx
    if best > lo:
        return best + 1 if text[best] in ".?!" else best
    return end


def chunk_document(doc: Document, target_chars: int = 2000, overlap: float = 0.15) -> list[Chunk]:
    chunks: list[Chunk] = []
    seq = 0
    step = max(1, int(target_chars * (1 - overlap)))
    for page in doc.pages:
        text = page.text
        first_line = text.splitlines()[0].strip() if text.splitlines() else ""
        heading = first_line if 0 < len(first_line) <= 80 else ""
        start = 0
        while start < len(text):
            raw_end = min(len(text), start + target_chars)
            end = _snap_end(text, raw_end)
            if end <= start:
                end = raw_end
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(chunk_id=f"{doc.doc_id}-{seq:04d}", doc_id=doc.doc_id,
                                    page=page.page, seq=seq, text=piece, heading=heading))
                seq += 1
            if end >= len(text):
                break
            start += step
    return chunks
