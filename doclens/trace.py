"""Hand-built tracing layer for observability (Span, Trace, Tracer)."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    """A timed span with metadata."""

    name: str
    start_ms: float
    end_ms: float
    meta: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        """Convert span to dict with rounded timestamps."""
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "meta": self.meta,
        }


class Trace:
    """A collection of spans with a unique trace ID."""

    def __init__(self, trace_id: str | None = None) -> None:
        """Initialize a trace with an optional trace ID."""
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.spans: list[Span] = []

    def to_dicts(self) -> list[dict]:
        """Convert all spans to dicts."""
        return [s.to_dict() for s in self.spans]

    def to_jsonl(self) -> str:
        """Convert all spans to JSONL (one JSON per line)."""
        return "\n".join(json.dumps(s.to_dict()) for s in self.spans)


class Tracer:
    """Context manager for recording spans into a trace."""

    def __init__(self, trace: Trace | None = None) -> None:
        """Initialize a tracer with an optional trace."""
        self.trace = trace or Trace()

    @contextmanager
    def span(self, name: str, **meta):
        """Context manager for recording a span.

        Yields the Span object so the caller can mutate meta after the
        wrapped call returns. The span is appended on exit even if an
        exception is raised.
        """
        start = time.perf_counter() * 1000
        sp = Span(name, start, start, dict(meta))
        try:
            yield sp
        finally:
            sp.end_ms = time.perf_counter() * 1000
            self.trace.spans.append(sp)
