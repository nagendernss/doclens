"""Shared dataclasses for the doclens pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    pages: list[PageText]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    seq: int
    text: str
    heading: str = ""


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    components: dict = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens)


@dataclass
class AnswerResult:
    answer: str
    citations: list[int]
    retrieved: list[Retrieved]
    refused: bool
    model: str
    usage: Usage = field(default_factory=Usage)


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def fingerprint(doc_id: str, page: int, text: str) -> str:
    norm = _NORM_RE.sub("", text.lower())
    words = norm.split()[:8]
    return f"{doc_id}|p{page}|{' '.join(words)}"
