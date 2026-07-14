"""FastAPI server: /api/ingest + /api/ask SSE, session cookie, rate caps."""
from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .answer import answer_question
from .chunker import chunk_document
from .index import VectorIndex
from .ingest import IngestError, ingest_pdf_bytes
from .ingest_url import ingest_url
from .providers.registry import (MissingKeyError, UnknownModelError,
                                 available_chat_models, get_chat, get_embedder)
from .ratelimit import RateLimiter
from .sessions import SessionDoc, SessionError, SessionStore

DEFAULT_MODEL = "gemini-3.1-flash-lite"
EMBED_PROGRESS_BATCH = 16
MAX_QUESTION_CHARS = 500
RETRIEVAL_PREVIEW_CHARS = 160
ASK_K = 5
DOC_NOT_FOUND_MESSAGE = "document not found — upload it again (sessions reset on restart)"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_DONE = object()


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _read_ingest_request(request: Request) -> tuple[bytes | None, str, str | None, str | None]:
    """Extract (file_bytes, filename, url, byo_key) from a multipart, form, or JSON body.

    Accepts multipart/form-data (file upload, optionally alongside url/byo_key fields),
    application/x-www-form-urlencoded (url/byo_key as form fields), or application/json
    ({"url", "byo_key"}). Unknown/empty bodies yield all-None/defaults, which the caller
    turns into a friendly SSE error rather than a 4xx.
    """
    content_type = request.headers.get("content-type", "")
    data: bytes | None = None
    filename = "upload.pdf"
    url: str | None = None
    byo_key: str | None = None

    if content_type.startswith("multipart/form-data") or \
            content_type.startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        file = form.get("file")
        if file is not None and hasattr(file, "read"):
            data = await file.read()
            filename = getattr(file, "filename", None) or filename
        raw_url = form.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            url = raw_url.strip()
        raw_key = form.get("byo_key")
        if isinstance(raw_key, str) and raw_key.strip():
            byo_key = raw_key.strip()
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            raw_url = body.get("url")
            if isinstance(raw_url, str) and raw_url.strip():
                url = raw_url.strip()
            raw_key = body.get("byo_key")
            if isinstance(raw_key, str) and raw_key.strip():
                byo_key = raw_key.strip()

    return data, filename, url, byo_key


async def _read_ask_request(request: Request) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract (doc_id, question, model, byo_key) from a JSON body.

    A malformed or non-dict body yields all-None, which the caller turns into a
    friendly "document not found" SSE error rather than a 4xx.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    raw_doc_id = body.get("doc_id")
    doc_id = raw_doc_id if isinstance(raw_doc_id, str) else None

    raw_question = body.get("question")
    question = raw_question if isinstance(raw_question, str) else None

    raw_model = body.get("model")
    model = raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else None

    raw_key = body.get("byo_key")
    byo_key = raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None

    return doc_id, question, model, byo_key


def create_app(store: SessionStore | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="doclens")
    app.state.store = store or SessionStore()
    app.state.limiter = limiter or RateLimiter(
        per_ip_ingest=int(os.environ.get("PER_IP_INGEST_CAP", "3")),
        per_ip_question=int(os.environ.get("PER_IP_QUESTION_CAP", "15")),
        global_cap=int(os.environ.get("DAILY_GLOBAL_CAP", "300")),
    )

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/models")
    def models():
        avail = available_chat_models()
        default = DEFAULT_MODEL if DEFAULT_MODEL in avail else (avail[0] if avail else None)
        return {"models": avail, "default": default}

    @app.post("/api/ingest")
    async def ingest(request: Request):
        store: SessionStore = app.state.store
        limiter: RateLimiter = app.state.limiter
        ip = request.client.host if request.client else "unknown"
        sid = request.cookies.get("dl_sid") or store.new_sid()

        # Try to parse the request body. If it fails (e.g., malformed multipart),
        # return an SSE stream with a single error event (HTTP 200).
        try:
            data, filename, url, byo_key = await _read_ingest_request(request)
        except Exception:
            # Never leak exception details (e.g., exception type).
            def error_stream():
                yield _sse("error", {"message": "could not read the upload — try again"})

            resp = StreamingResponse(error_stream(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            })
            resp.set_cookie("dl_sid", sid, httponly=True, samesite="lax")
            return resp

        def stream():
            if data is None and not url:
                yield _sse("error", {"message": "upload a PDF or paste a URL"})
                return
            if not byo_key:
                ok, reason = limiter.allow(ip, "ingest")
                if not ok:
                    yield _sse("error", {"message": reason})
                    return

            q: queue.Queue = queue.Queue()

            def emit(event: str, payload: dict) -> None:
                q.put((event, payload))

            def work():
                try:
                    emit("progress", {"stage": "fetch", "done": 0, "total": 0})
                    doc = ingest_pdf_bytes(data, filename) if data is not None else ingest_url(url)
                    n_pages = len(doc.pages)
                    emit("progress", {"stage": "parse", "done": n_pages, "total": n_pages})

                    chunks = chunk_document(doc)
                    n_chunks = len(chunks)
                    emit("progress", {"stage": "chunk", "done": n_chunks, "total": n_chunks})

                    embedder, embed_model = get_embedder(api_key=byo_key)
                    vectors: list[list[float]] = []
                    for i in range(0, n_chunks, EMBED_PROGRESS_BATCH):
                        batch = chunks[i:i + EMBED_PROGRESS_BATCH]
                        vectors.extend(embedder.embed([c.text for c in batch], embed_model))
                        emit("progress", {"stage": "embed", "done": len(vectors), "total": n_chunks})

                    index = VectorIndex()
                    index.add(chunks, vectors)
                    sdoc = SessionDoc(doc_id=doc.doc_id, title=doc.title, pages=n_pages,
                                      chunks=chunks, index=index, created=store.now())
                    store.add(sid, sdoc)
                    emit("ready", {"doc_id": doc.doc_id, "title": doc.title,
                                   "pages": n_pages, "chunks": n_chunks})
                except (IngestError, SessionError, MissingKeyError, UnknownModelError) as exc:
                    emit("error", {"message": str(exc)})
                except Exception:
                    # Never leak exception details: byo_key could appear in a provider's
                    # error message (e.g. an upstream HTTP error echoing the request).
                    emit("error", {"message": "ingest failed - try again"})
                finally:
                    q.put(_DONE)

            threading.Thread(target=work, daemon=True).start()
            while True:
                item = q.get()
                if item is _DONE:
                    break
                event, payload = item
                yield _sse(event, payload)

        resp = StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        resp.set_cookie("dl_sid", sid, httponly=True, samesite="lax")
        return resp

    @app.post("/api/ask")
    async def ask(request: Request):
        store: SessionStore = app.state.store
        limiter: RateLimiter = app.state.limiter
        ip = request.client.host if request.client else "unknown"
        sid = request.cookies.get("dl_sid") or store.new_sid()

        doc_id, question, model, byo_key = await _read_ask_request(request)
        chat_model_name = model or DEFAULT_MODEL

        def stream():
            sdoc = store.get(sid, doc_id)
            if sdoc is None:
                yield _sse("error", {"message": DOC_NOT_FOUND_MESSAGE})
                return

            if not byo_key:
                ok, reason = limiter.allow(ip, "question")
                if not ok:
                    yield _sse("error", {"message": reason})
                    return

            if not question or len(question) > MAX_QUESTION_CHARS:
                yield _sse("error", {"message": "question must be between 1 and 500 characters"})
                return

            q: queue.Queue = queue.Queue()

            def emit(event: str, payload: dict) -> None:
                q.put((event, payload))

            def work():
                try:
                    chat, chat_model = get_chat(chat_model_name, api_key=byo_key)
                    embedder, embed_model = get_embedder(api_key=byo_key)
                    result = answer_question(chat, chat_model, embedder, embed_model,
                                             sdoc.index, question, k=ASK_K)
                    chunks = [
                        {"page": r.chunk.page, "score": r.score,
                         "preview": r.chunk.text[:RETRIEVAL_PREVIEW_CHARS]}
                        for r in result.retrieved
                    ]
                    emit("retrieval", {"chunks": chunks})
                    emit("answer", {
                        "answer": result.answer,
                        "citations": result.citations,
                        "refused": result.refused,
                        "model": result.model,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                    })
                except (MissingKeyError, UnknownModelError) as exc:
                    emit("error", {"message": str(exc)})
                except Exception:
                    # Never leak exception details: byo_key could appear in a provider's
                    # error message (e.g. an upstream HTTP error echoing the request).
                    emit("error", {"message": "question failed - try again"})
                finally:
                    q.put(_DONE)

            threading.Thread(target=work, daemon=True).start()
            while True:
                item = q.get()
                if item is _DONE:
                    break
                event, payload = item
                yield _sse(event, payload)

        resp = StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        resp.set_cookie("dl_sid", sid, httponly=True, samesite="lax")
        return resp

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
