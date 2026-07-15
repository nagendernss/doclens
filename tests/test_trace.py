import json

import pytest

from doclens.trace import Trace, Tracer


def test_span_records_duration_and_meta_mutation():
    t = Tracer()
    with t.span("retrieve", mode="hybrid") as sp:
        sp.meta["candidates"] = 20
    s = t.trace.spans[0]
    assert s.name == "retrieve" and s.duration_ms >= 0
    assert s.meta == {"mode": "hybrid", "candidates": 20}


def test_spans_appended_in_exit_order():
    t = Tracer()
    with t.span("a"):
        pass
    with t.span("b"):
        pass
    assert [s.name for s in t.trace.spans] == ["a", "b"]


def test_trace_id_is_12_hex():
    tid = Trace().trace_id
    assert len(tid) == 12 and all(c in "0123456789abcdef" for c in tid)


def test_span_appended_even_if_block_raises():
    # Observability guarantee: the span must be recorded even when the wrapped
    # stage throws — you most want the trace when generate()/rerank() fails.
    t = Tracer()
    with pytest.raises(ValueError):
        with t.span("boom", stage="generate"):
            raise ValueError("upstream failed")
    assert [s.name for s in t.trace.spans] == ["boom"]
    assert t.trace.spans[0].duration_ms >= 0
    assert t.trace.spans[0].meta == {"stage": "generate"}


def test_to_dicts_and_jsonl_shape():
    t = Tracer()
    with t.span("embed", dims=768):
        pass
    d = t.trace.to_dicts()[0]
    assert {"name", "start_ms", "end_ms", "duration_ms", "meta"} <= set(d)
    assert json.loads(t.trace.to_jsonl().splitlines()[0])["name"] == "embed"
