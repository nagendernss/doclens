"""Tests for the FastAPI /api/ask SSE endpoint."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from doclens.index import VectorIndex
from doclens.ratelimit import RateLimiter
from doclens.server import DEFAULT_MODEL, DEFAULT_RETRIEVAL_MODE, create_app, sanitize_history
from doclens.sessions import SessionDoc, SessionStore
from doclens.types import AnswerResult, Chunk, Retrieved, Usage

SID = "a" * 32
DOC_ID = "doc1"
NOT_FOUND_MESSAGE = "document not found — upload it again (sessions reset on restart)"


def sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body (event:/data: blocks separated by a blank line) into (event, data) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if ev:
            events.append((ev, data))
    return events


def make_doc(doc_id: str = DOC_ID, num_chunks: int = 1) -> SessionDoc:
    chunks = [Chunk(f"c{i}", doc_id, 1, i, "hello world") for i in range(num_chunks)]
    return SessionDoc(doc_id=doc_id, title="Test Doc", pages=2, chunks=chunks,
                      index=VectorIndex(), created=100.0)


def fake_answer_question(chat, chat_model, embedder, embed_model, index, question, k=5, history=None,
                          *, retrieval_mode="hybrid_rerank", pool=20, tracer=None):
    # Mirrors the real answer_question's span shape closely enough for the SSE/trace
    # tests: embed+retrieve+generate always, rerank only in hybrid_rerank mode.
    if tracer is not None:
        with tracer.span("embed"):
            pass
        with tracer.span("retrieve"):
            pass
        if retrieval_mode == "hybrid_rerank":
            with tracer.span("rerank"):
                pass
        with tracer.span("generate"):
            pass
    return AnswerResult(
        answer="The answer is 42 [p.1].",
        citations=[1],
        retrieved=[Retrieved(chunk=Chunk("c0", DOC_ID, 1, 0, "hello world " * 20), score=0.9,
                             components={"dense_rank": 1, "bm25_rank": 2,
                                        "rerank_rank": 1 if retrieval_mode == "hybrid_rerank" else None})],
        refused=False,
        model=chat_model,
        usage=Usage(input_tokens=100, output_tokens=20),
    )


def make_app(monkeypatch, *, per_ip_question=10, store=None, patch_answer=True):
    """Build a TestClient app with a seeded session doc and answer_question mocked (network-free)."""
    import doclens.server as srv

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    if patch_answer:
        monkeypatch.setattr(srv, "answer_question", fake_answer_question)

    store = store if store is not None else SessionStore()
    if DOC_ID not in store.sessions.get(SID, {}).get("docs", {}):
        store.add(SID, make_doc())

    app = create_app(store=store,
                     limiter=RateLimiter(per_ip_ingest=10, per_ip_question=per_ip_question,
                                        global_cap=100))
    c = TestClient(app, base_url="http://test")
    c.cookies.set("dl_sid", SID)
    return c


@pytest.fixture
def client(monkeypatch):
    return make_app(monkeypatch)


def test_ask_emits_retrieval_then_answer_events(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "what is it?"})
    assert r.status_code == 200
    events = sse_events(r.text)
    assert [ev for ev, _ in events] == ["retrieval", "trace", "answer"]


def test_retrieval_payload_shape_and_preview_truncated(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "what is it?"})
    events = sse_events(r.text)
    retrieval = dict(events)["retrieval"]
    assert set(retrieval) == {"chunks"}
    chunk = retrieval["chunks"][0]
    assert set(chunk) == {"page", "score", "preview", "dense_rank", "bm25_rank", "rerank_rank"}
    assert chunk["page"] == 1
    assert chunk["score"] == pytest.approx(0.9)
    assert chunk["dense_rank"] == 1
    assert chunk["bm25_rank"] == 2
    assert chunk["rerank_rank"] == 1
    long_text = "hello world " * 20
    assert chunk["preview"] == long_text[:160]
    assert len(chunk["preview"]) <= 160


def test_trace_payload_has_id_and_nonempty_spans(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "what is it?"})
    events = sse_events(r.text)
    trace = dict(events)["trace"]
    assert set(trace) == {"trace_id", "spans"}
    assert isinstance(trace["trace_id"], str) and len(trace["trace_id"]) > 0
    assert isinstance(trace["spans"], list) and len(trace["spans"]) > 0


def test_hybrid_rerank_mode_trace_has_rerank_span(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    events = sse_events(r.text)
    trace = dict(events)["trace"]
    names = [s["name"] for s in trace["spans"]]
    assert "rerank" in names


def test_dense_mode_trace_has_no_rerank_span(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q", "mode": "dense"})
    events = sse_events(r.text)
    assert events[-1][0] == "answer"
    trace = dict(events)["trace"]
    names = [s["name"] for s in trace["spans"]]
    assert "rerank" not in names


@pytest.mark.parametrize("sent_mode, expected_mode", [
    ("bogus", DEFAULT_RETRIEVAL_MODE),
    (None, DEFAULT_RETRIEVAL_MODE),
    ("dense", "dense"),
    ("hybrid", "hybrid"),
    ("hybrid_rerank", "hybrid_rerank"),
])
def test_mode_validation_and_fallback(client, monkeypatch, sent_mode, expected_mode):
    """An invalid/absent mode silently falls back to the default; valid modes pass through unchanged."""
    import doclens.server as srv
    seen = {}

    def capture_mode(chat, chat_model, embedder, embed_model, index, question, k=5, history=None,
                     *, retrieval_mode="hybrid_rerank", pool=20, tracer=None):
        seen["mode"] = retrieval_mode
        return fake_answer_question(chat, chat_model, embedder, embed_model, index, question, k,
                                    retrieval_mode=retrieval_mode, pool=pool, tracer=tracer)

    monkeypatch.setattr(srv, "answer_question", capture_mode)
    body = {"doc_id": DOC_ID, "question": "q"}
    if sent_mode is not None:
        body["mode"] = sent_mode
    r = client.post("/api/ask", json=body)
    assert r.status_code == 200
    events = sse_events(r.text)
    assert events[-1][0] == "answer"
    assert seen["mode"] == expected_mode


def test_get_trace_returns_ndjson_for_known_id(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "what is it?"})
    trace_id = dict(sse_events(r.text))["trace"]["trace_id"]
    r2 = client.get(f"/api/trace/{trace_id}")
    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("application/x-ndjson")
    lines = [ln for ln in r2.text.strip().split("\n") if ln]
    assert len(lines) >= 1
    for ln in lines:
        obj = json.loads(ln)
        assert "name" in obj and "duration_ms" in obj


def test_get_trace_unknown_id_yields_404(client):
    r = client.get("/api/trace/unknownid")
    assert r.status_code == 404
    assert r.json() == {"error": "trace not found"}


def test_answer_payload_exact_shape(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "what is it?"})
    events = sse_events(r.text)
    answer = dict(events)["answer"]
    assert set(answer) == {"answer", "citations", "refused", "model", "input_tokens", "output_tokens"}
    assert answer["answer"] == "The answer is 42 [p.1]."
    assert answer["citations"] == [1]
    assert answer["refused"] is False
    assert answer["input_tokens"] == 100
    assert answer["output_tokens"] == 20


def test_refusal_result_emits_answer_with_refused_true(client, monkeypatch):
    import doclens.server as srv

    def fake_refusal(chat, chat_model, embedder, embed_model, index, question, k=5, history=None,
                      *, retrieval_mode="hybrid_rerank", pool=20, tracer=None):
        if tracer is not None:
            with tracer.span("embed"):
                pass
            with tracer.span("retrieve"):
                pass
        return AnswerResult(
            answer="Not in the document. Try rephrasing.",
            citations=[],
            retrieved=[Retrieved(chunk=Chunk("c0", DOC_ID, 1, 0, "x"), score=0.1)],
            refused=True, model=chat_model, usage=Usage(),
        )

    monkeypatch.setattr(srv, "answer_question", fake_refusal)
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "unrelated?"})
    events = sse_events(r.text)
    # A refusal must not short-circuit the retrieval or trace events — all three still fire, in order.
    assert [ev for ev, _ in events] == ["retrieval", "trace", "answer"]
    assert events[-1][1]["refused"] is True
    assert events[-1][1]["citations"] == []


def test_unknown_doc_id_yields_not_found_error(client):
    r = client.post("/api/ask", json={"doc_id": "nope-not-real", "question": "what is it?"})
    assert r.status_code == 200
    events = sse_events(r.text)
    assert events == [("error", {"message": NOT_FOUND_MESSAGE})]


def test_missing_doc_id_field_yields_not_found_error(client):
    r = client.post("/api/ask", json={"question": "what is it?"})
    events = sse_events(r.text)
    assert events == [("error", {"message": NOT_FOUND_MESSAGE})]


def test_no_session_cookie_yields_not_found_error(monkeypatch):
    c = make_app(monkeypatch)
    c.cookies.clear()
    r = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "what is it?"})
    events = sse_events(r.text)
    assert events == [("error", {"message": NOT_FOUND_MESSAGE})]


def test_question_too_long_yields_error(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "x" * 501})
    events = sse_events(r.text)
    assert len(events) == 1 and events[0][0] == "error"


def test_empty_question_yields_error(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": ""})
    events = sse_events(r.text)
    assert len(events) == 1 and events[0][0] == "error"


def test_missing_question_field_yields_error(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID})
    events = sse_events(r.text)
    assert len(events) == 1 and events[0][0] == "error"


def test_question_at_max_length_500_is_accepted(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "x" * 500})
    events = sse_events(r.text)
    assert events[-1][0] == "answer"


def test_rate_limited_question_yields_error(monkeypatch):
    c = make_app(monkeypatch, per_ip_question=2)
    for _ in range(2):
        r = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
        assert sse_events(r.text)[-1][0] == "answer"
    r3 = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    events = sse_events(r3.text)
    assert events == [("error", {"message": "question daily limit"})]


def test_byo_key_bypasses_rate_limit(monkeypatch):
    c = make_app(monkeypatch, per_ip_question=2)
    for _ in range(2):
        c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    r = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q", "byo_key": "sk-test-key"})
    assert sse_events(r.text)[-1][0] == "answer"


def test_whitespace_byo_key_treated_as_absent(monkeypatch):
    """Regression: a whitespace-only byo_key must not bypass the rate cap (mirrors ingest fix)."""
    c = make_app(monkeypatch, per_ip_question=2)
    for _ in range(2):
        c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    r = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q", "byo_key": "   "})
    events = sse_events(r.text)
    assert events == [("error", {"message": "question daily limit"})]


def test_byo_key_not_in_response_or_logs(client, capsys):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q", "byo_key": "sk-hidden-secret"})
    assert "sk-hidden-secret" not in r.text
    assert "sk-hidden-secret" not in capsys.readouterr().out


def test_byo_key_never_leaked_on_provider_error(client, monkeypatch, capsys):
    """A provider exception that happens to embed the key must still surface a generic message."""
    import doclens.server as srv
    secret = "sk-super-secret-xyz"

    def leaky_get_chat(model, api_key=None):
        raise RuntimeError(f"upstream rejected key {api_key}")

    monkeypatch.setattr(srv, "get_chat", leaky_get_chat)
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q", "byo_key": secret})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert secret not in events[-1][1]["message"]
    assert secret not in r.text
    assert secret not in capsys.readouterr().out


def test_sse_response_headers(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"


def test_ask_sets_dl_sid_cookie(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    assert r.cookies.get("dl_sid") == SID


def test_ask_without_cookie_still_sets_a_new_dl_sid_cookie(monkeypatch):
    c = make_app(monkeypatch)
    c.cookies.clear()
    r = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    sid = r.cookies.get("dl_sid")
    assert sid is not None and len(sid) == 32


def test_default_model_used_when_model_field_omitted(client):
    r = client.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    events = sse_events(r.text)
    assert dict(events)["answer"]["model"] == DEFAULT_MODEL


def test_explicit_valid_model_flows_through_to_answer_event(client):
    r = client.post("/api/ask",
                    json={"doc_id": DOC_ID, "question": "q", "model": "gemini-3.5-flash"})
    events = sse_events(r.text)
    assert dict(events)["answer"]["model"] == "gemini-3.5-flash"


def test_unknown_model_yields_error_event(client):
    r = client.post("/api/ask",
                    json={"doc_id": DOC_ID, "question": "q", "model": "not-a-real-model"})
    events = sse_events(r.text)
    assert events[-1][0] == "error"


def test_missing_api_key_yields_error_event(monkeypatch):
    """No GEMINI_API_KEY and no byo_key: real get_chat raises MissingKeyError -> error event."""
    import doclens.server as srv
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(srv, "answer_question", fake_answer_question)
    store = SessionStore()
    store.add(SID, make_doc())
    app = create_app(store=store,
                     limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    c.cookies.set("dl_sid", SID)
    r = c.post("/api/ask", json={"doc_id": DOC_ID, "question": "q"})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert "GEMINI_API_KEY" in events[-1][1]["message"]


def test_malformed_json_body_yields_not_found_error(client):
    r = client.post(
        "/api/ask",
        content=b"not json at all {{{",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200
    events = sse_events(r.text)
    assert events == [("error", {"message": NOT_FOUND_MESSAGE})]


def test_sanitize_history_rules():
    raw = ([{"question": f"q{i}", "answer": "a" * 2000} for i in range(8)]
           + ["junk", {"question": "", "answer": "x"}, {"question": "ok"},
              {"question": "x", "answer": "   "}, {"question": "last", "answer": "short"}])
    out = sanitize_history(raw)
    assert len(out) == 6
    assert out[-1] == {"question": "last", "answer": "short"}
    assert all(len(t["answer"]) <= 1500 for t in out)
    assert out[0]["question"] == "q3"


def test_sanitize_history_non_list():
    assert sanitize_history(None) == []
    assert sanitize_history("x") == []
    assert sanitize_history({"question": "q", "answer": "a"}) == []


def test_ask_passes_sanitized_history(client, monkeypatch):
    # client: fixture that seeds a session doc + monkeypatches answer_question.
    # Capture history parameter to verify sanitization.
    import doclens.server as srv
    seen = {}

    def capture(chat, chat_model, embedder, embed_model, index, question, k=5, history=None,
                *, retrieval_mode="hybrid_rerank", pool=20, tracer=None):
        seen["history"] = history
        return fake_answer_question(chat, chat_model, embedder, embed_model, index, question, k,
                                    retrieval_mode=retrieval_mode, pool=pool, tracer=tracer)

    monkeypatch.setattr(srv, "answer_question", capture)
    r = client.post("/api/ask", json={
        "doc_id": DOC_ID, "question": "follow-up?",
        "history": [{"question": "q1", "answer": "a1"}, "junk"],
    })
    events = sse_events(r.text)
    assert events[-1][0] == "answer"
    assert seen["history"] == [{"question": "q1", "answer": "a1"}]
