"""Tests for the FastAPI /api/ingest SSE endpoint."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from doclens.ingest import MAX_PDF_BYTES
from doclens.ratelimit import RateLimiter
from doclens.server import create_app
from doclens.sessions import SessionStore
from doclens.types import Document, PageText


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


class FakeEmbedder:
    """Deterministic embedder: one fixed-length vector per input text, no network."""

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def fake_ingest_pdf_bytes(data: bytes, source: str) -> Document:
    return Document(doc_id="d1", title="Fake Doc", source=source,
                    pages=[PageText(1, "hello world"), PageText(2, "more text")])


def fake_ingest_url(url: str, **kwargs) -> Document:
    return Document(doc_id="u1", title="URL Doc", source=url,
                    pages=[PageText(1, "web content here")])


def fake_get_embedder(model: str = "gemini-embedding-001", api_key: str | None = None):
    return FakeEmbedder(), model


@pytest.fixture
def client(monkeypatch):
    """TestClient with core pipeline functions monkeypatched: no network, no real PDFs."""
    import doclens.server as srv

    monkeypatch.setattr(srv, "ingest_pdf_bytes", fake_ingest_pdf_bytes)
    monkeypatch.setattr(srv, "ingest_url", fake_ingest_url)
    monkeypatch.setattr(srv, "get_embedder", fake_get_embedder)

    app = create_app(store=SessionStore(),
                     limiter=RateLimiter(per_ip_ingest=3, per_ip_question=10, global_cap=100))
    return TestClient(app, base_url="http://test")


def test_healthz():
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_models_endpoint_default_prefers_configured_default(monkeypatch):
    import doclens.server as srv
    monkeypatch.setattr(srv, "available_chat_models",
                        lambda: ["gemini-3.1-flash-lite", "gemini-3.5-flash"])
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/api/models")
    body = r.json()
    assert body["models"] == ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
    assert body["default"] == "gemini-3.1-flash-lite"


def test_models_endpoint_empty_when_no_key(monkeypatch):
    """No GEMINI_API_KEY -> available_chat_models() is empty -> default is None, not an error."""
    import doclens.server as srv
    monkeypatch.setattr(srv, "available_chat_models", lambda: [])
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/api/models")
    body = r.json()
    assert body["models"] == []
    assert body["default"] is None


def test_ingest_upload_progress_order_then_ready(client):
    """Fake PDF upload -> progress stages in order, then a ready event with doc metadata."""
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"%PDF-fake", "application/pdf")})
    assert r.status_code == 200
    events = sse_events(r.text)
    stages = [data["stage"] for ev, data in events if ev == "progress"]
    assert stages == ["fetch", "parse", "chunk", "embed"]
    assert events[-1][0] == "ready"
    ready = events[-1][1]
    assert ready == {"doc_id": "d1", "title": "Fake Doc", "pages": 2, "chunks": 2}


def test_ingest_progress_payload_shape(client):
    """Every progress event carries exactly stage/done/total, per the SSE contract."""
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"%PDF-fake", "application/pdf")})
    events = sse_events(r.text)
    for ev, data in events:
        if ev == "progress":
            assert set(data) == {"stage", "done", "total"}
    embed_events = [data for ev, data in events if ev == "progress" and data["stage"] == "embed"]
    assert embed_events[-1]["done"] == embed_events[-1]["total"] == 2


def test_ingest_url_json_path(client):
    """JSON body {"url": ...} ingests via ingest_url, not ingest_pdf_bytes."""
    r = client.post("/api/ingest", json={"url": "https://example.test/doc.pdf"})
    events = sse_events(r.text)
    assert events[-1][0] == "ready"
    assert events[-1][1] == {"doc_id": "u1", "title": "URL Doc", "pages": 1, "chunks": 1}


def test_ingest_url_form_encoded_path(client):
    """Form-encoded (non-multipart) {"url": ...} also works, per 'JSON/form url'."""
    r = client.post("/api/ingest", data={"url": "https://example.test/doc.pdf"})
    events = sse_events(r.text)
    assert events[-1][0] == "ready"
    assert events[-1][1]["doc_id"] == "u1"


def test_no_file_or_url_yields_error(client):
    r = client.post("/api/ingest")
    events = sse_events(r.text)
    assert events == [("error", {"message": "upload a PDF or paste a URL"})]


def test_ingest_sets_dl_sid_cookie(client):
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    sid = r.cookies.get("dl_sid")
    assert sid is not None
    assert len(sid) == 32
    assert all(c in "0123456789abcdef" for c in sid)
    raw = r.headers.get("set-cookie", "")
    assert "HttpOnly" in raw
    assert "samesite=lax" in raw.lower()


def test_ingest_reuses_existing_dl_sid_cookie(client):
    """Second request on the same client (same cookie jar) keeps the same sid."""
    r1 = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    r2 = client.post("/api/ingest", files={"file": ("b.pdf", b"y", "application/pdf")})
    assert r1.cookies.get("dl_sid") == r2.cookies.get("dl_sid")


def test_sse_response_headers(client):
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"


def test_rate_limited_fourth_ingest_yields_error(client):
    """per_ip_ingest=3 in the fixture limiter: 4th call in a day is denied, HTTP 200 + error event."""
    for _ in range(3):
        r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
        assert sse_events(r.text)[-1][0] == "ready"
    r4 = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    assert r4.status_code == 200
    events = sse_events(r4.text)
    assert events == [("error", {"message": "ingest daily limit"})]


def test_byo_key_bypasses_rate_limit(client):
    for _ in range(3):
        client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")},
                    data={"byo_key": "sk-test-key"})
    assert sse_events(r.text)[-1][0] == "ready"


def test_byo_key_not_in_response_or_logs(client, capsys):
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")},
                    data={"byo_key": "sk-hidden-secret"})
    assert "sk-hidden-secret" not in r.text
    assert "sk-hidden-secret" not in capsys.readouterr().out


def test_byo_key_never_leaked_on_provider_error(client, monkeypatch, capsys):
    """A provider exception that happens to embed the key must still surface a generic message."""
    import doclens.server as srv
    secret = "sk-super-secret-xyz"

    def leaky_get_embedder(model="gemini-embedding-001", api_key=None):
        raise RuntimeError(f"upstream rejected key {api_key}")

    monkeypatch.setattr(srv, "get_embedder", leaky_get_embedder)
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")},
                    data={"byo_key": secret})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert secret not in events[-1][1]["message"]
    assert secret not in r.text
    assert secret not in capsys.readouterr().out


def test_whitespace_byo_key_treated_as_absent(client):
    """Regression: a whitespace-only byo_key must not bypass the rate cap.

    Before the fix, `if not byo_key` only checked truthiness, so " " (truthy)
    counted as "present" and skipped the rate-limit check entirely.
    """
    for _ in range(3):
        client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    r = client.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")},
                    data={"byo_key": "   "})
    events = sse_events(r.text)
    assert events == [("error", {"message": "ingest daily limit"})]


def test_session_doc_created_uses_store_clock(monkeypatch):
    """Regression: SessionDoc.created must come from the injected store clock, not wall time.

    SessionStore.add() picks eviction victims by `created`, and SessionStore
    exposes an injectable `now()` specifically so callers can control time in
    tests/ops tooling. server.py must stamp new docs with that same clock.
    """
    import doclens.server as srv
    monkeypatch.setattr(srv, "ingest_pdf_bytes", fake_ingest_pdf_bytes)
    monkeypatch.setattr(srv, "get_embedder", fake_get_embedder)
    clock = {"t": 12345.0}
    store = SessionStore(now=lambda: clock["t"])
    app = create_app(store=store,
                     limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    r = c.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    events = sse_events(r.text)
    doc_id = events[-1][1]["doc_id"]
    sid = r.cookies.get("dl_sid")
    sdoc = store.get(sid, doc_id)
    assert sdoc.created == 12345.0


def test_ingest_builds_hybrid_index_not_bare_vector_index(monkeypatch):
    """Session doc must be indexed by HybridIndex (dense+BM25), not the old bare VectorIndex."""
    import doclens.server as srv
    from doclens.hybrid import HybridIndex

    monkeypatch.setattr(srv, "ingest_pdf_bytes", fake_ingest_pdf_bytes)
    monkeypatch.setattr(srv, "get_embedder", fake_get_embedder)
    store = SessionStore()
    app = create_app(store=store,
                     limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    r = c.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    events = sse_events(r.text)
    doc_id = events[-1][1]["doc_id"]
    sid = r.cookies.get("dl_sid")
    sdoc = store.get(sid, doc_id)
    assert isinstance(sdoc.index, HybridIndex)
    assert len(sdoc.index) == events[-1][1]["chunks"]


def test_garbage_pdf_yields_error_event():
    """Real ingest_pdf_bytes (not monkeypatched): unparseable bytes -> IngestError -> error event."""
    app = create_app(limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    r = c.post("/api/ingest", files={"file": ("junk.pdf", b"not a pdf at all", "application/pdf")})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert "PDF" in events[-1][1]["message"]


def test_oversized_pdf_yields_error_event():
    """Real ingest_pdf_bytes: a payload over the 10 MB cap -> IngestError -> error event."""
    app = create_app(limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    big = b"x" * (MAX_PDF_BYTES + 1)
    r = c.post("/api/ingest", files={"file": ("big.pdf", big, "application/pdf")})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert "10" in events[-1][1]["message"]


def test_session_error_yields_error_event(monkeypatch):
    """A session over its chunk budget surfaces SessionError as an error event, not a 500."""
    import doclens.server as srv
    monkeypatch.setattr(srv, "ingest_pdf_bytes", fake_ingest_pdf_bytes)
    monkeypatch.setattr(srv, "get_embedder", fake_get_embedder)
    store = SessionStore(max_chunks=0)
    app = create_app(store=store,
                     limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    r = c.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert "chunk budget" in events[-1][1]["message"]


def test_missing_api_key_yields_error_event(monkeypatch):
    """No GEMINI_API_KEY and no byo_key: real get_embedder raises MissingKeyError -> error event."""
    import doclens.server as srv
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(srv, "ingest_pdf_bytes", fake_ingest_pdf_bytes)
    app = create_app(limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    r = c.post("/api/ingest", files={"file": ("a.pdf", b"x", "application/pdf")})
    events = sse_events(r.text)
    assert events[-1][0] == "error"
    assert "GEMINI_API_KEY" in events[-1][1]["message"]


def test_malformed_multipart_yields_error_event():
    """Malformed multipart body -> returns HTTP 200 with SSE error event, not HTTP 500."""
    app = create_app(limiter=RateLimiter(per_ip_ingest=5, per_ip_question=5, global_cap=100))
    c = TestClient(app, base_url="http://test")
    # Send a request with multipart content-type header but invalid body (garbage, not multipart)
    r = c.post(
        "/api/ingest",
        content=b"garbage not multipart",
        headers={"content-type": "multipart/form-data; boundary=X"}
    )
    # Should return 200, not 500
    assert r.status_code == 200
    # Should have SSE content-type
    assert r.headers["content-type"].startswith("text/event-stream")
    # Should have exactly one error event with friendly message
    events = sse_events(r.text)
    assert len(events) == 1
    assert events[0][0] == "error"
    assert isinstance(events[0][1]["message"], str)
    # Should not leak exception details
    assert "MultipartError" not in events[0][1]["message"]
    assert "garbage" not in events[0][1]["message"]
