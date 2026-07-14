from __future__ import annotations

import re

import yaml

REQUIRED = ("id", "doc", "question", "relevant_fps", "expected_facts", "answerable")


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        cases = yaml.safe_load(fh)["cases"]
    for i, case in enumerate(cases):
        missing = [k for k in REQUIRED if k not in case]
        if missing:
            raise ValueError(f"case {case.get('id', i)}: missing {', '.join(missing)}")
    return cases


def recall_at_k(retrieved_fps: list[str], relevant_fps: list[str]) -> float:
    return 1.0 if any(fp in retrieved_fps for fp in relevant_fps) else 0.0


def mrr(retrieved_fps: list[str], relevant_fps: list[str]) -> float:
    for rank, fp in enumerate(retrieved_fps, start=1):
        if fp in relevant_fps:
            return 1.0 / rank
    return 0.0


def faithful(answer: str, citations: list[int], retrieved_pages: list[int],
             facts: list[str]) -> bool:
    if any(not re.search(f, answer, re.IGNORECASE) for f in facts):
        return False
    return all(page in retrieved_pages for page in citations)
