# doclens Web (FastAPI + SSE + Sessions + Frontend + Deploy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Public web app on Render where anyone uploads a PDF or pastes a URL, watches ingestion stream (parse → chunk → embed), then asks questions and gets page-cited answers streamed live — with per-visitor session corpora, rate caps, and optional BYO key.

**Architecture:** One FastAPI service serving a static vanilla-JS frontend. Two SSE endpoints: `/api/ingest` (multipart file OR `{url}` → progress events → a doc_id) and `/api/ask` (`{doc_id, question}` → retrieval + answer events). Per-visitor corpus (cookie session id → {doc_id: (chunks, VectorIndex)}) held in memory with a 30-min idle TTL sweeper. The synchronous pipeline runs in a worker thread; a `queue.Queue` drains to SSE. In-memory rate limiter (per-IP + global daily). BYO Gemini key bypasses caps, never logged.

**Tech Stack:** FastAPI + uvicorn (new `[web]` extra) + python-multipart (uploads), existing doclens core, vanilla JS frontend, Docker (python:3.12-slim), Render blueprint.

## Global Constraints

- Core runtime deps unchanged (httpx, pyyaml, pypdf, selectolax, numpy). Web deps ONLY under `[project.optional-dependencies] web = ["fastapi>=0.115", "uvicorn>=0.30", "python-multipart>=0.0.9"]`.
- No live network in tests: ASGI via `fastapi.testclient.TestClient` (NOT httpx.ASGITransport — httpx 0.28 ASGITransport is async-only, learned in repolens); monkeypatch `doclens.server.get_embedder`/`get_chat`/`ingest_pdf_bytes`/`ingest_url` per test; fixture PDFs generated in-repo.
- Session: cookie `dl_sid` (random 32-hex, httponly, samesite=lax); corpus `{sid: {doc_id: SessionDoc(title, pages, chunks, index, created)}}`; idle TTL 30 min (per-sid last-access); caps `MAX_DOCS_PER_SESSION=3`, `MAX_CHUNKS_PER_SESSION=1500`; background sweeper thread (daemon, 60s interval) OR lazy sweep on each request (pick lazy — simpler, testable via injectable clock). Render restart wipes all — documented.
- Rate caps (env-overridable): `PER_IP_INGEST_CAP=3`, `PER_IP_QUESTION_CAP=15`, `DAILY_GLOBAL_CAP=300` (ingests+questions combined); BYO key bypasses; capped → SSE `error` event, HTTP 200.
- Upload limits: PDF ≤ 10 MB (reuse ingest cap), question ≤ 500 chars, URL via existing SSRF-guarded ingest_url. BYO key body/form field `byo_key`, in-memory only, NEVER logged/echoed/in-exception-text.
- SSE contracts (exact — frontend + tests depend):
  - `/api/ingest`: `event: progress` `data: {"stage":"fetch|parse|chunk|embed","done":int,"total":int}` → `event: ready` `data: {"doc_id","title","pages","chunks"}` | `event: error` `data: {"message"}`
  - `/api/ask`: `event: retrieval` `data: {"chunks":[{"page","score","preview"}]}` → `event: answer` `data: {"answer","citations":[int],"refused":bool,"model","input_tokens","output_tokens"}` | `event: error` `data: {"message"}`
- Endpoints: `POST /api/ingest`, `POST /api/ask`, `GET /api/models` → `{"models":[...],"default":...}`, `GET /healthz` → `{"ok":true}`, `GET /` serves web/index.html, `/static` mount.
- `registry.get_chat(model, api_key=None)` and `get_embedder(model=..., api_key=None)` — the api_key override already exists from core? NO — core's get_chat/get_embedder take api_key already (registry `_key(api_key)`). Verify and reuse.
- Frontend: vanilla JS/CSS, no build, no CDN; dark theme consistent with repolens/portfolio (ink #08080b, emerald #00d179, mono labels). og:image + meta.
- Commits conventional; every task committed; branch `feat/web`.

---

### Task 1: Rate limiter

**Files:** Create `doclens/ratelimit.py`, `tests/test_ratelimit.py`.

**Interfaces:** `class RateLimiter(per_ip_ingest, per_ip_question, global_cap, today=None)`:
- `allow(ip: str, kind: str) -> tuple[bool, str]` where kind ∈ {"ingest","question"}; counts only on allow; per-IP counter is per-kind, global counter is combined; UTC-date reset via injectable `today()`; thread-safe (`threading.Lock`); friendly deny messages (per-IP names the kind + "daily limit"; global says "global").
- `remaining(ip, kind) -> int`.

- [ ] **Step 1: failing tests** — `tests/test_ratelimit.py`:
```python
from doclens.ratelimit import RateLimiter


def test_per_ip_ingest_cap():
    rl = RateLimiter(per_ip_ingest=2, per_ip_question=10, global_cap=100)
    assert rl.allow("1.1.1.1", "ingest")[0] is True
    assert rl.allow("1.1.1.1", "ingest")[0] is True
    ok, reason = rl.allow("1.1.1.1", "ingest")
    assert ok is False and "daily limit" in reason
    assert rl.allow("1.1.1.1", "question")[0] is True  # separate kind counter


def test_global_cap_combined():
    rl = RateLimiter(per_ip_ingest=10, per_ip_question=10, global_cap=2)
    rl.allow("1.1.1.1", "ingest")
    rl.allow("2.2.2.2", "question")
    ok, reason = rl.allow("3.3.3.3", "ingest")
    assert ok is False and "global" in reason


def test_utc_reset():
    day = {"d": "2026-07-14"}
    rl = RateLimiter(per_ip_ingest=1, per_ip_question=1, global_cap=99, today=lambda: day["d"])
    assert rl.allow("1.1.1.1", "ingest")[0] is True
    assert rl.allow("1.1.1.1", "ingest")[0] is False
    day["d"] = "2026-07-15"
    assert rl.allow("1.1.1.1", "ingest")[0] is True


def test_denied_not_counted():
    rl = RateLimiter(per_ip_ingest=1, per_ip_question=1, global_cap=99)
    rl.allow("1.1.1.1", "ingest")
    rl.allow("1.1.1.1", "ingest")  # denied
    assert rl.remaining("1.1.1.1", "ingest") == 0
    assert rl.remaining("2.2.2.2", "ingest") == 1
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — mirror repolens' RateLimiter but with per-kind per-IP counters and a combined global counter (UTC date key, lock). Deny reasons contain the required substrings.
- [ ] **Step 4:** full suite (expect 63) + ruff clean.
- [ ] **Step 5:** commit `feat: per-IP (per-kind) + global daily rate limiter`.

---

### Task 2: Session store

**Files:** Create `doclens/sessions.py`, `tests/test_sessions.py`.

**Interfaces:**
- `@dataclass SessionDoc: doc_id, title, pages: int, chunks: list[Chunk], index: VectorIndex, created: float`
- `class SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=None)`:
  - `add(sid, doc: SessionDoc) -> None` — enforces caps: if adding exceeds max_docs, evict oldest doc in that sid; if total chunks would exceed max_chunks, raise `SessionError("session chunk budget exceeded")`.
  - `get(sid, doc_id) -> SessionDoc | None` (touches last-access; returns None if sid/doc absent or swept).
  - `new_sid() -> str` (32-hex).
  - `sweep()` — drop sids idle > ttl; also called lazily at the start of `add`/`get` (injectable `now()` clock for tests).
  - thread-safe (Lock).
- `class SessionError(Exception)`.

- [ ] **Step 1: failing tests** — cover: add+get roundtrip; max_docs evicts oldest; chunk-budget raises; TTL sweep drops idle sid (injected clock); get touches access (prevents sweep); unknown sid/doc → None; new_sid uniqueness+length.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** implement per interface.
- [ ] **Step 4:** suite (expect ~70) + ruff.
- [ ] **Step 5:** commit `feat: in-memory per-visitor session store with TTL and caps`.

---

### Task 3: FastAPI server — ingest SSE

**Files:** Modify `pyproject.toml` (web extra); Create `doclens/server.py`, `tests/test_server_ingest.py`.

**Interfaces:**
- `create_app(store=None, limiter=None) -> FastAPI`; `app = create_app()` (uvicorn target `doclens.server:app`).
- `POST /api/ingest`: multipart form `file` OR JSON/form `url`; optional `byo_key`. Flow (in worker thread, streamed): rate-limit (kind="ingest", skip if byo_key) → get/set `dl_sid` cookie → emit progress(fetch) → ingest (ingest_pdf_bytes for upload / ingest_url for url) → progress(parse) → chunk_document → progress(chunk, total=len) → embed in batches emitting progress(embed, done/total) → build VectorIndex → store.add(SessionDoc) → emit ready{doc_id,title,pages,chunks}. Errors (IngestError, SessionError, provider) → error event. byo_key never logged; generic message on unexpected exceptions.
- `GET /healthz`, `GET /api/models` (from available_chat_models + default).
- Set-Cookie on the response (SSE StreamingResponse with cookie header).

- [ ] **Step 1:** add web extra, `pip install -e .[dev,web]`.
- [ ] **Step 2: failing tests** — `tests/test_server_ingest.py` (TestClient; monkeypatch server.ingest_pdf_bytes → fake Document, server.get_embedder → fake vectors): healthz; models; ingest a fake PDF upload → progress events in order (fetch,parse,chunk,embed…) then ready with doc_id; ingest sets dl_sid cookie; rate-limited 4th ingest → error event; byo_key not in output/logs; oversized/garbage → error event. SSE parse helper (split \n\n).
- [ ] **Step 3:** implement.
- [ ] **Step 4:** suite + ruff.
- [ ] **Step 5:** commit `feat: FastAPI ingest SSE (upload + URL), session cookie, caps`.

---

### Task 4: Server — ask SSE

**Files:** Modify `doclens/server.py`; Create `tests/test_server_ask.py`.

**Interfaces:**
- `POST /api/ask`: JSON `{doc_id, question, model?, byo_key?}`; read `dl_sid` cookie → `store.get(sid, doc_id)`; if missing → error "document not found — upload it again (sessions reset on restart)"; rate-limit kind="question"; validate question 1..500; run answer_question in worker thread; emit `retrieval` {chunks: top-k previews (page, score, first ~160 chars)} then `answer` {answer, citations, refused, model, tokens}. Refusal still emits answer event (refused=true, citations=[]). Provider/GitHub-less errors → error event; byo_key leak-safe.

- [ ] **Step 1: failing tests** — monkeypatch server.answer_question → fake AnswerResult; seed a session doc via the store fixture: ask returns retrieval then answer events; unknown doc_id → error; question>500 → error; rate-limited → error; refusal AnswerResult → answer event refused=true; byo_key absent from output.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** suite + ruff.
- [ ] **Step 5:** commit `feat: FastAPI ask SSE — retrieval preview + cited answer`.

---

### Task 5: Frontend

**Files:** Create `web/index.html`, `web/app.js`, `web/style.css`.

**Interfaces:** consumes both SSE endpoints + `/api/models`. No frameworks/CDN.

Layout: header (doclens brand, one-line pitch, "sessions reset on restart / history stays in this browser tab" note). Ingest panel: drag-drop / file picker for PDF **and** a URL text input (either-or), model select, optional BYO key field, "Ingest" button. On ingest: show a live progress line per stage (fetch/parse/chunk/embed n/m) into a bar; on ready, reveal the ask panel titled with the doc title + "N pages · M chunks", and add the doc to an in-tab document switcher (localStorage `doclens.docs.v1` = list of {doc_id, title, pages, chunks} for THIS tab; note doc_ids die on server restart — show a friendly re-upload prompt if ask returns "not found"). Ask panel: question input + Ask; streams a "retrieving…" state showing the retrieval chunk previews (page + score bar), then the answer with `[p.N]` citations rendered (page pills, non-clickable — no source URL to link, just show page). Refusal renders distinctly ("Not in the document"). Reuse repolens-style esc()/escAttr() — escape ALL model output before innerHTML; no new sinks (security gate).

- [ ] **Step 1:** write the three files (design language: ink bg, emerald accent, mono; responsive ≤760px).
- [ ] **Step 2:** gates — `node -c web/app.js`; full suite unchanged; ruff; curl (no key): `/` 200 contains "doclens", `/static/app.js` 200, `/static/style.css` 200, `/api/models` JSON.
- [ ] **Step 3:** commit `feat: web frontend — upload/URL ingest with live progress, cited Q&A`.

---

### Task 6: Docker + render.yaml + deploy docs

**Files:** Create `Dockerfile`, `.dockerignore`, `render.yaml`; Modify `README.md` (Live demo placeholder + Deploy section; flip Scope note — web app now exists).

- [ ] **Step 1:** Dockerfile (python:3.12-slim, COPY pyproject/README/LICENSE + doclens + web + evals, `pip install -e .[web]`, CMD uvicorn `doclens.server:app` on `${PORT:-10000}`). `.dockerignore` (.git, .superpowers, tests, docs, __pycache__, *.egg-info, results*.json, .github). `render.yaml` (docker, free, healthCheckPath /healthz, GEMINI_API_KEY sync:false, cap envVars). README Live-demo placeholder + Deploy-your-own + docker run one-liner.
- [ ] **Step 2:** static COPY-path sanity (Docker likely absent — mark skipped, verify paths exist).
- [ ] **Step 3:** full gate + commit `feat: Dockerfile + Render blueprint + deploy docs`.

---

### Task 7: Deploy + live verify (HUMAN-IN-LOOP)

- [ ] **Step 1:** merge feat/web → main, push. User: Render → New → Blueprint → nagendernss/doclens → set GEMINI_API_KEY → Apply.
- [ ] **Step 2:** live verify: `/healthz`; upload a small PDF via UI → progress → ask → cited answer; paste an HTTPS URL → answer; rate-limit + refusal spot-check; SSE streams (confirm no proxy buffering — add `X-Accel-Buffering: no` header to SSE responses in Task 3/4 if Render buffers).
- [ ] **Step 3:** README live URL; commit + push. Update memory + ledger.
- [ ] **Step 4 (follow-up):** doclens card on portfolio + LinkedIn; pin repo.

## Verification (whole plan)

1. Full suite green (~80 tests), ruff clean.
2. Local uvicorn: upload PDF → progress streams → cited answer; URL ingest works; rate cap + refusal; BYO key bypass.
3. Docker healthz (or Render build green).
4. Live onrender.com: end-to-end upload+ask and URL+ask.

## Out of scope

Multi-doc cross-corpus questions, reranking, OCR, accounts/persistence, answer token-streaming, thread history across tabs.
