# doclens Core (Ingest → Chunk → Embed → Index → Answer → CLI → Evals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Working RAG core: ingest a PDF/URL/text file, chunk with metadata, embed via Gemini (adapter), retrieve from a hand-built cosine index, answer with `[p.N]` citations — driveable from a CLI — plus an eval harness (recall@5, MRR, faithfulness, refusal accuracy) that writes the README table.

**Architecture:** Pure-Python pipeline modules with one job each (`ingest → chunker → embeddings → index → answer`), provider adapters on raw httpx (no SDKs), in-memory numpy vector index behind a tiny Store protocol. Evals label gold chunks by stable fingerprints (doc + page + normalized 8-word prefix) so re-chunking doesn't invalidate labels.

**Tech Stack:** Python ≥3.11; deps `httpx`, `pyyaml`, `pypdf`, `selectolax`, `numpy`. Dev: `pytest`, `ruff`. (FastAPI/SSE/frontend/deploy = Plan B.)

## Global Constraints

- Package `doclens`; runtime deps EXACTLY: httpx, pyyaml, pypdf, selectolax, numpy. No LangChain, no vendor SDKs.
- No live network in tests (`httpx.MockTransport`, monkeypatched DNS, fixture bytes).
- Caps (spec §4): PDF ≤ 10 MB and ≤ 300 pages; URL fetch ≤ 5 MB, 15 s timeout; chunk target 2,000 chars (~500 tokens), 15% overlap, sentence-snap window ±80 chars; retrieval k=5; refusal threshold: max cosine < 0.30 ⇒ "not in the document" without an LLM call.
- SSRF guard on URL ingestion: scheme must be http/https; resolved IPs must all be public (reject loopback, private RFC1918, link-local 169.254.0.0/16, ULA fc00::/7, ::1).
- Embedding batches ≤ 64 texts/request; retry 429/5xx with 2s/4s/8s backoff, fail-fast 4xx (shared `_http.py`, same as repolens).
- Models (env `GEMINI_API_KEY`): embeddings `gemini-embedding-001`; chat `gemini-3.1-flash-lite` (free) + `gemini-3.5-flash` row.
- Citation format in answers: `[p.N]` (page-based). Answer must refuse when context insufficient.
- Chunk fingerprint = `f"{doc_id}|p{page}|{' '.join(normalized_text.split()[:8])}"` where normalized = lowercase, alnum+space only.
- Conventional commits; every task ends committed; branch `feat/core` off main.

---

### Task 1: Skeleton

**Files:** Create `pyproject.toml`, `doclens/__init__.py`, `tests/__init__.py`, `tests/test_package.py`, `.gitignore`, `LICENSE`.

**Interfaces:** installable `doclens` with `__version__ = "0.1.0"`; console script `doclens = "doclens.cli:main"`; ruff line-length 100 target py311; pytest testpaths ["tests"].

- [ ] **Step 1: failing test** — `tests/test_package.py`:
```python
import doclens


def test_version():
    assert doclens.__version__ == "0.1.0"
```
- [ ] **Step 2:** `python -m pytest -q` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: implement** — `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "doclens"
version = "0.1.0"
description = "RAG Q&A over your PDFs and links — with retrieval evals"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Nagender Swaroop Srivastava", email = "nagendernss.work@gmail.com" }]
dependencies = ["httpx>=0.27", "pyyaml>=6.0", "pypdf>=4.0", "selectolax>=0.3", "numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]

[project.scripts]
doclens = "doclens.cli:main"

[tool.setuptools.packages.find]
include = ["doclens*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
```
`doclens/__init__.py`: `__version__ = "0.1.0"`. `.gitignore`: `__pycache__/`, `*.egg-info/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `results*.json`, `.env`. `LICENSE`: MIT, `Copyright (c) 2026 Nagender Swaroop Srivastava`.
- [ ] **Step 4:** `pip install -e .[dev] && python -m pytest -q && ruff check .` → 1 passed, clean.
- [ ] **Step 5:** `git add -A && git commit -m "chore: project skeleton"`

---

### Task 2: Types

**Files:** Create `doclens/types.py`, `tests/test_types.py`.

**Interfaces (everything downstream imports these exactly):**
```python
@dataclass PageText: page: int; text: str
@dataclass Document: doc_id: str; title: str; source: str; pages: list[PageText]
@dataclass Chunk: chunk_id: str; doc_id: str; page: int; seq: int; text: str; heading: str = ""
@dataclass Retrieved: chunk: Chunk; score: float
@dataclass Usage: input_tokens: int = 0; output_tokens: int = 0   # with __add__
@dataclass AnswerResult: answer: str; citations: list[int]; retrieved: list[Retrieved]; refused: bool; model: str; usage: Usage
def fingerprint(doc_id: str, page: int, text: str) -> str   # per Global Constraints
```
`citations` = sorted unique page numbers actually cited.

- [ ] **Step 1: failing test** — `tests/test_types.py`:
```python
from doclens.types import Chunk, Document, PageText, Retrieved, Usage, fingerprint


def test_usage_add():
    assert (Usage(1, 2) + Usage(3, 4)) == Usage(4, 6)


def test_fingerprint_normalizes():
    fp = fingerprint("d1", 3, "  The QUICK, brown fox—jumps over the lazy dog today!  ")
    assert fp == "d1|p3|the quick brown foxjumps over the lazy dog"


def test_shapes():
    d = Document(doc_id="d1", title="T", source="file.pdf", pages=[PageText(1, "hi")])
    c = Chunk(chunk_id="c1", doc_id="d1", page=1, seq=0, text="hi")
    r = Retrieved(chunk=c, score=0.5)
    assert d.pages[0].page == 1 and r.chunk.chunk_id == "c1" and c.heading == ""
```
Note the fingerprint expectation: em-dash is stripped as non-alnum WITHOUT inserting a space (`fox—jumps` → `foxjumps`), then first 8 whitespace-split words are joined.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `doclens/types.py`:
```python
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
```
- [ ] **Step 4:** tests pass; full suite green; ruff clean.
- [ ] **Step 5:** commit `feat: pipeline types + chunk fingerprint`.

---

### Task 3: PDF + text-file ingestion

**Files:** Create `doclens/ingest.py`, `tests/test_ingest_pdf.py`.

**Interfaces:**
- `class IngestError(Exception)`
- `ingest_pdf_bytes(data: bytes, source: str) -> Document` — pypdf extract per page; skip empty pages; raise IngestError beyond 10 MB / 300 pages / unparseable; `doc_id` = first 12 hex of sha256(data); title = PDF metadata title or source basename.
- `ingest_text(text: str, source: str, title: str | None = None) -> Document` — single "page" per ~3,000-char block (page numbers 1..n); used for .txt/.md files and by evals.
- `ingest_file(path: str) -> Document` — dispatch by suffix: .pdf → pdf bytes; .txt/.md → text; .html/.htm → `ingest_html` (Task 4; import lazily so Task 3 tests pass before Task 4 exists).
- `MAX_PDF_BYTES = 10 * 1024 * 1024`, `MAX_PDF_PAGES = 300`.

- [ ] **Step 1: failing tests** — `tests/test_ingest_pdf.py`:
```python
import pytest
from pypdf import PdfWriter

from doclens.ingest import MAX_PDF_BYTES, IngestError, ingest_file, ingest_pdf_bytes, ingest_text


def make_pdf(pages_text):
    """Build a real PDF in memory with one text page per entry (pypdf only)."""
    import io

    from pypdf.generic import (ArrayObject, DictionaryObject, NameObject,
                               NumberObject, StreamObject)
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        stream = StreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        stream[NameObject("/Length")] = NumberObject(len(stream.get_data()))
        page[NameObject("/Contents")] = ArrayObject([writer._add_object(stream)])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_pages_extracted():
    data = make_pdf(["hello page one", "second page here"])
    doc = ingest_pdf_bytes(data, "sample.pdf")
    assert len(doc.pages) == 2
    assert "hello page one" in doc.pages[0].text
    assert doc.pages[1].page == 2
    assert len(doc.doc_id) == 12


def test_pdf_size_cap():
    with pytest.raises(IngestError, match="10"):
        ingest_pdf_bytes(b"x" * (MAX_PDF_BYTES + 1), "big.pdf")


def test_pdf_garbage_raises():
    with pytest.raises(IngestError):
        ingest_pdf_bytes(b"not a pdf at all", "junk.pdf")


def test_ingest_text_paginates():
    doc = ingest_text("A" * 7000, "notes.txt")
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert doc.title == "notes.txt"


def test_ingest_file_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    doc = ingest_file(str(f))
    assert doc.pages[0].text == "hello world"
```
If pypdf internals differ for the manual content stream, adjust `make_pdf` (the test helper, not the implementation) until `PdfReader.extract_text()` returns the words — verify with a quick REPL probe first.
- [ ] **Step 2:** run → FAIL (module missing).
- [ ] **Step 3: implement** — `doclens/ingest.py`:
```python
"""Document ingestion: PDF bytes, plain text, files (URL/HTML lands in Task 4)."""
from __future__ import annotations

import hashlib
import io
import os

from pypdf import PdfReader

from .types import Document, PageText

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 300
TEXT_PAGE_CHARS = 3000


class IngestError(Exception):
    pass


def _doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def ingest_pdf_bytes(data: bytes, source: str) -> Document:
    if len(data) > MAX_PDF_BYTES:
        raise IngestError(f"PDF is {len(data)} bytes; cap is 10 MB")
    try:
        reader = PdfReader(io.BytesIO(data))
        n = len(reader.pages)
    except Exception as exc:
        raise IngestError(f"could not parse PDF: {exc}") from exc
    if n > MAX_PDF_PAGES:
        raise IngestError(f"PDF has {n} pages; cap is {MAX_PDF_PAGES}")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(PageText(page=i, text=text))
    if not pages:
        raise IngestError("no extractable text (scanned/OCR PDFs are not supported)")
    meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    return Document(doc_id=_doc_id(data), title=meta_title or os.path.basename(source),
                    source=source, pages=pages)


def ingest_text(text: str, source: str, title: str | None = None) -> Document:
    text = text.strip()
    if not text:
        raise IngestError("empty document")
    pages = [PageText(page=i + 1, text=text[start:start + TEXT_PAGE_CHARS])
             for i, start in enumerate(range(0, len(text), TEXT_PAGE_CHARS))]
    return Document(doc_id=_doc_id(text.encode()), title=title or os.path.basename(source),
                    source=source, pages=pages)


def ingest_file(path: str) -> Document:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pdf":
        with open(path, "rb") as fh:
            return ingest_pdf_bytes(fh.read(), path)
    if suffix in (".txt", ".md"):
        with open(path, encoding="utf-8") as fh:
            return ingest_text(fh.read(), path)
    if suffix in (".html", ".htm"):
        from .ingest_url import ingest_html  # lazy: Task 4
        with open(path, encoding="utf-8") as fh:
            return ingest_html(fh.read(), path)
    raise IngestError(f"unsupported file type {suffix!r} (pdf/txt/md/html)")
```
- [ ] **Step 4:** tests pass; ruff clean.
- [ ] **Step 5:** commit `feat: PDF and text ingestion with caps`.

---

### Task 4: URL/HTML ingestion + SSRF guard

**Files:** Create `doclens/ingest_url.py`, `tests/test_ingest_url.py`.

**Interfaces:**
- `ingest_html(html: str, source: str) -> Document` — selectolax: drop script/style/nav/footer; title = `<title>` or source; text with h1-h3 kept as their own lines; paginate via `ingest_text` blocks (reuse), preserving title.
- `ingest_url(url: str, client: httpx.Client | None = None, resolver=None) -> Document` — scheme http/https only; `resolver(host) -> list[str]` (default: `socket.getaddrinfo` wrapper) must yield ONLY public IPs else `IngestError("refusing private/internal address")`; GET with 15 s timeout, `follow_redirects=True` (each hop re-checked? v1: after response, re-validate `resp.url.host` via resolver too); content-type application/pdf or URL endswith .pdf → `ingest_pdf_bytes`; text/html → `ingest_html`; else IngestError; body cap 5 MB (`MAX_URL_BYTES`).
- `_is_public_ip(ip: str) -> bool` exported for tests (uses `ipaddress`: reject private, loopback, link_local, reserved, multicast, unspecified, ULA).

- [ ] **Step 1: failing tests** — `tests/test_ingest_url.py`:
```python
import httpx
import pytest

from doclens.ingest_url import _is_public_ip, ingest_html, ingest_url
from doclens.ingest import IngestError


def test_is_public_ip():
    for bad in ("127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.5.5",
                "169.254.169.254", "::1", "fc00::1"):
        assert _is_public_ip(bad) is False, bad
    assert _is_public_ip("93.184.216.34") is True


def test_ingest_html_strips_chrome():
    html = ("<html><head><title>My Doc</title><style>x{}</style></head><body>"
            "<nav>menu</nav><h2>Intro</h2><p>Real content here.</p>"
            "<script>evil()</script><footer>foot</footer></body></html>")
    doc = ingest_html(html, "http://example.com/a")
    assert doc.title == "My Doc"
    text = doc.pages[0].text
    assert "Real content here." in text and "Intro" in text
    assert "menu" not in text and "evil" not in text and "foot" not in text


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ingest_url_html():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>hello url content</p>")

    doc = ingest_url("https://example.com/page", client=make_client(handler),
                     resolver=lambda host: ["93.184.216.34"])
    assert "hello url content" in doc.pages[0].text


def test_ingest_url_blocks_private():
    with pytest.raises(IngestError, match="private"):
        ingest_url("https://internal.corp/x", client=make_client(lambda r: httpx.Response(200)),
                   resolver=lambda host: ["10.0.0.5"])


def test_ingest_url_scheme():
    with pytest.raises(IngestError, match="scheme"):
        ingest_url("ftp://example.com/x", resolver=lambda host: ["93.184.216.34"])


def test_ingest_url_size_cap():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"},
                              content=b"a" * (5 * 1024 * 1024 + 1))

    with pytest.raises(IngestError, match="5 MB"):
        ingest_url("https://example.com/big", client=make_client(handler),
                   resolver=lambda host: ["93.184.216.34"])
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `doclens/ingest_url.py`:
```python
"""URL + HTML ingestion with an SSRF guard."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from selectolax.parser import HTMLParser

from .ingest import IngestError, ingest_pdf_bytes, ingest_text

MAX_URL_BYTES = 5 * 1024 * 1024
TIMEOUT_S = 15.0
_DROP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "iframe")


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


def _default_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise IngestError(f"could not resolve {host}: {exc}") from exc
    return [info[4][0] for info in infos]


def _assert_public(host: str, resolver) -> None:
    ips = resolver(host)
    if not ips or not all(_is_public_ip(ip) for ip in ips):
        raise IngestError("refusing private/internal address")


def ingest_html(html: str, source: str) -> Document:  # noqa: F821 (Document via ingest_text)
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else source
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()
    lines = []
    body = tree.body or tree.root
    for node in body.css("h1, h2, h3, p, li, td, pre, blockquote"):
        text = node.text(separator=" ", strip=True)
        if text:
            lines.append(text)
    content = "\n".join(lines) or (body.text(separator="\n", strip=True) if body else "")
    if not content.strip():
        raise IngestError("no readable text at that URL")
    return ingest_text(content, source, title=title)


def ingest_url(url: str, client: httpx.Client | None = None, resolver=None):
    resolver = resolver or _default_resolver
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestError(f"unsupported scheme {parsed.scheme!r}")
    if not parsed.hostname:
        raise IngestError("no host in URL")
    _assert_public(parsed.hostname, resolver)
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=True)
    try:
        resp = client.get(url)
        if resp.status_code >= 400:
            raise IngestError(f"fetch failed with HTTP {resp.status_code}")
        if resp.url.host and resp.url.host != parsed.hostname:
            _assert_public(resp.url.host, resolver)
        body = resp.content
        if len(body) > MAX_URL_BYTES:
            raise IngestError("document over the 5 MB URL cap")
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype == "application/pdf" or str(resp.url).lower().endswith(".pdf"):
            return ingest_pdf_bytes(body, url)
        if ctype in ("text/html", "application/xhtml+xml", "text/plain", ""):
            if ctype == "text/plain":
                return ingest_text(body.decode(resp.encoding or "utf-8", "replace"), url)
            return ingest_html(body.decode(resp.encoding or "utf-8", "replace"), url)
        raise IngestError(f"unsupported content type {ctype!r}")
    except httpx.HTTPError as exc:
        raise IngestError(f"fetch failed: {exc}") from exc
    finally:
        if own_client:
            client.close()
```
- [ ] **Step 4:** tests pass; ruff clean.
- [ ] **Step 5:** commit `feat: URL/HTML ingestion with SSRF guard and caps`.

---

### Task 5: Chunker

**Files:** Create `doclens/chunker.py`, `tests/test_chunker.py`.

**Interfaces:** `chunk_document(doc: Document, target_chars: int = 2000, overlap: float = 0.15) -> list[Chunk]` — per page: windows of ≤ target_chars stepping by `target*(1-overlap)`; window end snaps back to the nearest sentence boundary (`. `, `? `, `! `, `\n`) within 80 chars if one exists; `chunk_id = f"{doc.doc_id}-{seq:04d}"` with seq global across the doc; short pages (< target) become one chunk; heading = first line of the page if it is ≤ 80 chars else "".

- [ ] **Step 1: failing tests** — `tests/test_chunker.py`:
```python
from doclens.chunker import chunk_document
from doclens.types import Document, PageText


def make_doc(texts):
    return Document(doc_id="d1", title="t", source="s",
                    pages=[PageText(i + 1, t) for i, t in enumerate(texts)])


def test_short_page_single_chunk():
    doc = make_doc(["Heading line\nBody text here."])
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1-0000"
    assert chunks[0].page == 1 and chunks[0].heading == "Heading line"


def test_long_page_overlapping_windows():
    sentence = "Alpha bravo charlie delta echo foxtrot golf hotel. "
    doc = make_doc([sentence * 100])  # ~5200 chars
    chunks = chunk_document(doc, target_chars=2000, overlap=0.15)
    assert len(chunks) >= 3
    # windows overlap: next chunk starts before previous ends
    joined = doc.pages[0].text
    first_end = joined.find(chunks[1].text[:40])
    assert 0 < first_end < 2000
    # sentence snap: every non-final chunk ends at a sentence boundary
    for c in chunks[:-1]:
        assert c.text.rstrip().endswith((".", "?", "!"))


def test_seq_spans_pages():
    doc = make_doc(["one. " * 500, "two. " * 500])
    chunks = chunk_document(doc)
    assert [c.page for c in chunks[:1]][0] == 1
    assert chunks[-1].page == 2
    seqs = [c.seq for c in chunks]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `doclens/chunker.py`:
```python
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
```
If `test_long_page_overlapping_windows`'s snap assertion is brittle against the exact `_snap_end` arithmetic, fix the IMPLEMENTATION until non-final chunks end on sentence punctuation — that is the contract; do not weaken the test.
- [ ] **Step 4:** tests pass; ruff clean.
- [ ] **Step 5:** commit `feat: sliding-window chunker with sentence snap`.

---

### Task 6: Embedding + chat providers (raw httpx)

**Files:** Create `doclens/providers/__init__.py`, `doclens/providers/_http.py`, `doclens/providers/registry.py`, `doclens/providers/gemini.py`, `tests/test_providers.py`.

**Interfaces:**
- `_http.post_with_retry(client, url, headers, json, retries=3)` — 429/5xx backoff 2/4/8 (monkeypatch `time.sleep` in tests), 4xx fail-fast (verbatim pattern from repolens).
- `registry.CHAT_MODELS = {"gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite", 0.0, 0.0), "gemini-3.5-flash": ("gemini", "gemini-3.5-flash", 0.0, 0.0)}`; `EMBED_MODELS = {"gemini-embedding-001": ("gemini", "gemini-embedding-001")}`; `get_chat(model, api_key=None) -> (provider, model_id)`; `get_embedder(model="gemini-embedding-001", api_key=None) -> (provider, model_id)`; `MissingKeyError`, `UnknownModelError`; env `GEMINI_API_KEY`; `available_chat_models()`.
- `gemini.GeminiChat(api_key, client=None).complete(messages: list[dict], model: str) -> tuple[str, Usage]` — plain text chat, NO tools: messages `{"role": "system"|"user", "content": str}` → systemInstruction + contents; returns `(text, Usage)`.
- `gemini.GeminiEmbedder(api_key, client=None).embed(texts: list[str], model: str) -> list[list[float]]` — POST `{BASE}/models/{model}:batchEmbedContents` with `{"requests": [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} ...]}`, batching ≤ 64 per call internally; parse `{"embeddings": [{"values": [...]}, ...]}`.

- [ ] **Step 1: failing tests** — `tests/test_providers.py`:
```python
import json

import httpx
import pytest

from doclens.providers._http import post_with_retry
from doclens.providers.gemini import GeminiChat, GeminiEmbedder
from doclens.providers.registry import MissingKeyError, get_chat, get_embedder


def test_retry_then_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resp = post_with_retry(client, "https://x/y", headers={}, json={})
    assert resp.status_code == 200 and calls["n"] == 3 and sleeps == [2, 4]


def test_4xx_fails_fast():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400)))
    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(client, "https://x/y", headers={}, json={}).raise_for_status()


def test_registry_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        get_chat("gemini-3.1-flash-lite")
    provider, model = get_chat("gemini-3.1-flash-lite", api_key="k")
    assert model == "gemini-3.1-flash-lite" and provider.api_key == "k"


def test_chat_translation():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hi [p.2]"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}})

    chat = GeminiChat(api_key="K", client=httpx.Client(transport=httpx.MockTransport(handler)))
    text, usage = chat.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        "gemini-3.1-flash-lite")
    assert text == "hi [p.2]" and usage.input_tokens == 5
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "q"}]}]


def test_embedder_batches_and_parses():
    batches = []

    def handler(request):
        body = json.loads(request.content)
        batches.append(len(body["requests"]))
        return httpx.Response(200, json={"embeddings": [
            {"values": [0.1, 0.2]} for _ in body["requests"]]})

    emb = GeminiEmbedder(api_key="K", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = emb.embed([f"t{i}" for i in range(130)], "gemini-embedding-001")
    assert len(out) == 130 and out[0] == [0.1, 0.2]
    assert batches == [64, 64, 2]


def test_get_embedder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "e")
    provider, model = get_embedder()
    assert model == "gemini-embedding-001"
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `_http.py` (same as repolens pattern):
```python
from __future__ import annotations

import time

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


def post_with_retry(client: httpx.Client, url: str, headers: dict, json: dict,
                    retries: int = 3) -> httpx.Response:
    delay = 2
    for attempt in range(retries + 1):
        resp = client.post(url, headers=headers, json=json)
        if resp.status_code not in RETRY_STATUSES or attempt == retries:
            return resp
        time.sleep(delay)
        delay *= 2
    return resp
```
`registry.py`:
```python
from __future__ import annotations

import os

CHAT_MODELS = {
    "gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite", 0.0, 0.0),
    "gemini-3.5-flash": ("gemini", "gemini-3.5-flash", 0.0, 0.0),
}
EMBED_MODELS = {"gemini-embedding-001": ("gemini", "gemini-embedding-001")}
ENV_KEY = "GEMINI_API_KEY"


class UnknownModelError(Exception):
    pass


class MissingKeyError(Exception):
    pass


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get(ENV_KEY)
    if not key:
        raise MissingKeyError(f"needs env {ENV_KEY}")
    return key


def available_chat_models() -> list[str]:
    return list(CHAT_MODELS) if os.environ.get(ENV_KEY) else []


def get_chat(model: str, api_key: str | None = None):
    if model not in CHAT_MODELS:
        raise UnknownModelError(f"unknown chat model {model!r}")
    from .gemini import GeminiChat
    return GeminiChat(api_key=_key(api_key)), CHAT_MODELS[model][1]


def get_embedder(model: str = "gemini-embedding-001", api_key: str | None = None):
    if model not in EMBED_MODELS:
        raise UnknownModelError(f"unknown embedding model {model!r}")
    from .gemini import GeminiEmbedder
    return GeminiEmbedder(api_key=_key(api_key)), EMBED_MODELS[model][1]
```
`gemini.py`:
```python
"""Gemini chat + embeddings over raw REST."""
from __future__ import annotations

import httpx

from ..types import Usage
from ._http import post_with_retry

BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_BATCH = 64


class GeminiChat:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def complete(self, messages: list[dict], model: str) -> tuple[str, Usage]:
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        resp = post_with_retry(self._client, f"{BASE}/models/{model}:generateContent",
                               headers={"x-goog-api-key": self.api_key}, json=body)
        resp.raise_for_status()
        data = resp.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        meta = data.get("usageMetadata", {})
        return text, Usage(meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0))


class GeminiEmbedder:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            body = {"requests": [{"model": f"models/{model}",
                                   "content": {"parts": [{"text": t}]}} for t in batch]}
            resp = post_with_retry(self._client, f"{BASE}/models/{model}:batchEmbedContents",
                                   headers={"x-goog-api-key": self.api_key}, json=body)
            resp.raise_for_status()
            out.extend(e["values"] for e in resp.json()["embeddings"])
        return out
```
`__init__.py` empty.
- [ ] **Step 4:** tests pass; ruff clean.
- [ ] **Step 5:** commit `feat: gemini chat + batched embeddings on raw httpx with retry`.

---

### Task 7: Vector index

**Files:** Create `doclens/index.py`, `tests/test_index.py`.

**Interfaces:** `class VectorIndex: __init__(); add(chunks: list[Chunk], vectors: list[list[float]]); search(vector: list[float], k: int = 5) -> list[Retrieved]; __len__`. Cosine via normalized matrix dot product; `add` L2-normalizes rows (zero vectors → left as zeros, never NaN); `search` returns top-k by score desc, stable. Raises `ValueError` on chunk/vector length mismatch or dim mismatch with existing matrix.

- [ ] **Step 1: failing tests** — `tests/test_index.py`:
```python
import math

import pytest

from doclens.index import VectorIndex
from doclens.types import Chunk


def C(i):
    return Chunk(chunk_id=f"c{i}", doc_id="d", page=1, seq=i, text=f"t{i}")


def test_cosine_ranking_hand_computed():
    idx = VectorIndex()
    idx.add([C(0), C(1), C(2)], [[1, 0], [1, 1], [0, 1]])
    out = idx.search([1, 0], k=2)
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
    assert math.isclose(out[0].score, 1.0, abs_tol=1e-9)
    assert math.isclose(out[1].score, 1 / math.sqrt(2), abs_tol=1e-9)


def test_mismatch_raises():
    idx = VectorIndex()
    with pytest.raises(ValueError):
        idx.add([C(0)], [[1, 0], [0, 1]])
    idx.add([C(0)], [[1, 0]])
    with pytest.raises(ValueError):
        idx.add([C(1)], [[1, 0, 0]])


def test_zero_vector_safe():
    idx = VectorIndex()
    idx.add([C(0), C(1)], [[0, 0], [1, 0]])
    out = idx.search([1, 0], k=5)
    assert out[0].chunk.chunk_id == "c1"
    assert all(not math.isnan(r.score) for r in out)


def test_len():
    idx = VectorIndex()
    assert len(idx) == 0
    idx.add([C(0)], [[1.0]])
    assert len(idx) == 1
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `doclens/index.py`:
```python
"""Hand-built in-memory cosine-similarity index (numpy)."""
from __future__ import annotations

import numpy as np

from .types import Chunk, Retrieved


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._mat: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        mat = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._mat is not None and mat.shape[1] != self._mat.shape[1]:
            raise ValueError("embedding dimension mismatch")
        self._mat = mat if self._mat is None else np.vstack([self._mat, mat])
        self._chunks.extend(chunks)

    def search(self, vector: list[float], k: int = 5) -> list[Retrieved]:
        if self._mat is None or not len(self._chunks):
            return []
        q = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self._mat @ q
        order = np.argsort(-scores)[:k]
        return [Retrieved(chunk=self._chunks[i], score=float(scores[i])) for i in order]
```
- [ ] **Step 4:** tests pass; ruff clean.
- [ ] **Step 5:** commit `feat: hand-built cosine vector index`.

---

### Task 8: Grounded answering

**Files:** Create `doclens/answer.py`, `tests/test_answer.py`.

**Interfaces:**
- `SYSTEM_PROMPT` — instructs: answer ONLY from provided context; cite pages `[p.N]` per claim; if the context does not contain the answer, reply starting with exactly `Not in the document.`
- `REFUSAL_THRESHOLD = 0.30`
- `answer_question(chat, chat_model: str, embedder, embed_model: str, index: VectorIndex, question: str, k: int = 5) -> AnswerResult` — embed question → search; if no results or top score < threshold → `AnswerResult(refused=True, answer="Not in the document…", citations=[], retrieved=…, usage=Usage())` WITHOUT calling chat; else build context block `[p.N] <chunk text>` per chunk, call chat, parse `[p.N]` citations (regex `\[p\.(\d+)\]`, sorted unique ints), `refused = answer.startswith("Not in the document")`.

- [ ] **Step 1: failing tests** — `tests/test_answer.py`:
```python
from doclens.answer import REFUSAL_THRESHOLD, answer_question
from doclens.index import VectorIndex
from doclens.types import Chunk, Usage


class FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages, model):
        self.calls.append(messages)
        return self.reply, Usage(10, 5)


class FakeEmbedder:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, texts, model):
        return [self.vec for _ in texts]


def make_index():
    idx = VectorIndex()
    idx.add(
        [Chunk("c0", "d", 2, 0, "The refund window is 30 days."),
         Chunk("c1", "d", 5, 1, "Contact support by email.")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    return idx


def test_grounded_answer_with_citations():
    chat = FakeChat("Refunds are allowed within 30 days [p.2].")
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(),
                          "what is the refund window?")
    assert res.refused is False
    assert res.citations == [2]
    assert res.retrieved[0].chunk.chunk_id == "c0"
    context = chat.calls[0][-1]["content"]
    assert "[p.2]" in context
    assert "The refund window is 30 days." in context


def test_low_score_refuses_without_llm():
    chat = FakeChat("should never be called")
    weak = [0.001, 0.0009]  # cosine vs both chunks ≈ .74? use orthogonal-ish tiny…
    res = answer_question(chat, "m", FakeEmbedder([0.0, 0.0]), "e", make_index(),
                          "unrelated question")
    assert res.refused is True and chat.calls == []
    assert res.answer.startswith("Not in the document")


def test_model_refusal_detected():
    chat = FakeChat("Not in the document. The context never mentions pricing.")
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(), "pricing?")
    assert res.refused is True and res.citations == []
```
Note: `FakeEmbedder([0.0, 0.0])` makes every cosine 0 < threshold — deterministic refusal path.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `doclens/answer.py`:
```python
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
                    index: VectorIndex, question: str, k: int = 5) -> AnswerResult:
    qvec = embedder.embed([question], embed_model)[0]
    retrieved = index.search(qvec, k=k)
    if not retrieved or retrieved[0].score < REFUSAL_THRESHOLD:
        return AnswerResult(answer=REFUSAL_TEXT, citations=[], retrieved=retrieved,
                            refused=True, model=chat_model, usage=Usage())
    context = "\n\n".join(f"[p.{r.chunk.page}] {r.chunk.text}" for r in retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context chunks:\n\n{context}\n\nQuestion: {question}"},
    ]
    text, usage = chat.complete(messages, chat_model)
    citations = sorted({int(m) for m in _CITE_RE.findall(text)})
    refused = text.strip().startswith("Not in the document")
    return AnswerResult(answer=text, citations=citations if not refused else [],
                        retrieved=retrieved, refused=refused, model=chat_model, usage=usage)
```
- [ ] **Step 4:** tests pass; ruff clean.
- [ ] **Step 5:** commit `feat: grounded answering with refusal threshold and page citations`.

---

### Task 9: CLI

**Files:** Create `doclens/cli.py`, `tests/test_cli.py`.

**Interfaces:** `doclens ask <path-or-url> "<question>" [--model NAME] [-k N]` — ingest (path via `ingest_file`, http(s):// via `ingest_url`) → chunk → embed → index → answer; prints ingest stats line (`title · pages · chunks`), retrieved pages line, answer, usage line. `doclens models`. Exit codes: 0 ok, 2 config (missing key/unknown model), 1 runtime (IngestError/httpx). `main(argv=None) -> int`; streams reconfigure `errors="replace"` (Windows).

- [ ] **Step 1: failing tests** — `tests/test_cli.py`:
```python
from unittest.mock import MagicMock, patch

from doclens.cli import main
from doclens.types import AnswerResult, Chunk, Document, PageText, Retrieved, Usage


def fake_doc():
    return Document("d1", "Title", "s.pdf", [PageText(1, "text")])


def fake_answer():
    ch = Chunk("c0", "d1", 2, 0, "chunk text")
    return AnswerResult("Answer [p.2].", [2], [Retrieved(ch, 0.9)], False, "m", Usage(9, 3))


@patch("doclens.cli.answer_question", return_value=fake_answer())
@patch("doclens.cli.VectorIndex")
@patch("doclens.cli.get_embedder", return_value=(MagicMock(embed=lambda t, m: [[1.0]] * len(t)), "e"))
@patch("doclens.cli.get_chat", return_value=(MagicMock(), "m"))
@patch("doclens.cli.chunk_document", return_value=[Chunk("c0", "d1", 1, 0, "x")])
@patch("doclens.cli.ingest_file", return_value=fake_doc())
def test_ask_happy(mi, mc, mg, me, mv, ma, capsys):
    code = main(["ask", "doc.pdf", "what?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Title" in out and "Answer [p.2]." in out and "tokens=9+3" in out


@patch("doclens.cli.ingest_file", side_effect=__import__(
    "doclens.ingest", fromlist=["IngestError"]).IngestError("no extractable text"))
def test_ask_ingest_error_exit_1(mi, capsys):
    code = main(["ask", "bad.pdf", "q"])
    assert code == 1 and "no extractable text" in capsys.readouterr().err


def test_models_lists(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "gemini-3.1-flash-lite *" in out
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `doclens/cli.py`:
```python
"""doclens CLI: one-shot ingest + ask from the terminal."""
from __future__ import annotations

import argparse
import sys

from .answer import answer_question
from .chunker import chunk_document
from .index import VectorIndex
from .ingest import IngestError, ingest_file
from .ingest_url import ingest_url
from .providers.registry import (CHAT_MODELS, MissingKeyError, UnknownModelError,
                                 available_chat_models, get_chat, get_embedder)

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(prog="doclens")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("source")
    ask.add_argument("question")
    ask.add_argument("--model", default=DEFAULT_MODEL)
    ask.add_argument("-k", type=int, default=5)
    sub.add_parser("models")
    args = parser.parse_args(argv)

    if args.cmd == "models":
        avail = set(available_chat_models())
        for name in CHAT_MODELS:
            print(f"{name} *" if name in avail else name)
        return 0

    try:
        chat, chat_model = get_chat(args.model)
        embedder, embed_model = get_embedder()
    except (MissingKeyError, UnknownModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        source = args.source
        doc = ingest_url(source) if source.startswith(("http://", "https://")) \
            else ingest_file(source)
        chunks = chunk_document(doc)
        vectors = embedder.embed([c.text for c in chunks], embed_model)
        index = VectorIndex()
        index.add(chunks, vectors)
        print(f"doclens · {doc.title} · {len(doc.pages)} pages · {len(chunks)} chunks")
        res = answer_question(chat, chat_model, embedder, embed_model, index,
                              args.question, k=args.k)
        pages = ", ".join(f"p.{r.chunk.page}" for r in res.retrieved)
        print(f"retrieved: {pages}")
        print()
        print(res.answer)
        print(f"\nmodel={args.model} tokens={res.usage.input_tokens}+{res.usage.output_tokens}")
        return 0
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # provider/network
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```
- [ ] **Step 4:** tests pass; full suite; ruff clean.
- [ ] **Step 5:** commit `feat: CLI — one-shot ingest+ask, models list`.

---

### Task 10: Seed corpus + gold set + metrics

**Files:** Create `evals/__init__.py`, `evals/corpus/` (3 docs), `evals/gold.yaml`, `evals/metrics.py`, `tests/test_metrics.py`.

**Interfaces:**
- Corpus (AUTHORING step — judgment): 3 small documents committed to the repo, each 2–6 pages of real prose the implementer WRITES (original text, no copyright risk — style them as: `rfc-style-spec.md` a fictional-but-technical protocol spec; `device-manual.md` an appliance manual; `policy.md` a leave/refund policy). Each ≥ 4,000 chars with named sections, concrete numbers, and at least 3 facts NOT present in the other docs. (Original authored text beats downloading: zero licensing risk, stable forever, facts controllable for gold labels.)
- `evals/gold.yaml`: 30 cases — fields `id`, `doc` (corpus filename), `question`, `relevant_fps` (list of chunk fingerprints per `types.fingerprint`), `expected_facts` (regex all-of vs answer), `answerable` (bool; ≥ 5 false cases with `relevant_fps: []`, `expected_facts: []`).
- `evals/metrics.py`:
  - `recall_at_k(retrieved_fps: list[str], relevant_fps: list[str]) -> float` (1.0 if any relevant in retrieved else 0.0; unanswerable → skip)
  - `mrr(retrieved_fps, relevant_fps) -> float` (1/rank of first relevant, else 0)
  - `faithful(answer: str, citations: list[int], retrieved_pages: list[int], facts: list[str]) -> bool` — all fact regexes match AND every citation page ∈ retrieved pages
  - `load_gold(path) -> list[dict]` validating required keys (ValueError names the case id).
- Gold authoring loop (in-task, no live network needed): chunk each corpus doc via the real chunker, print fingerprints, pick relevant ones for each question.

- [ ] **Step 1: failing tests** — `tests/test_metrics.py`:
```python
import pytest

from evals.metrics import faithful, load_gold, mrr, recall_at_k


def test_recall():
    assert recall_at_k(["a", "b"], ["b"]) == 1.0
    assert recall_at_k(["a"], ["z"]) == 0.0


def test_mrr():
    assert mrr(["x", "rel", "y"], ["rel"]) == 0.5
    assert mrr(["x"], ["rel"]) == 0.0


def test_faithful():
    assert faithful("Refund in 30 days [p.2].", [2], [2, 5], ["30 days"]) is True
    assert faithful("Refund [p.9].", [9], [2], ["refund"]) is False   # cites unretrieved page
    assert faithful("No facts here [p.2].", [2], [2], ["missing fact"]) is False


def test_load_gold_validates(tmp_path):
    bad = tmp_path / "g.yaml"
    bad.write_text("cases:\n  - id: x\n    doc: a.md\n")
    with pytest.raises(ValueError, match="x"):
        load_gold(str(bad))


def test_seed_gold_loads():
    cases = load_gold("evals/gold.yaml")
    assert len(cases) >= 30
    assert sum(1 for c in cases if not c["answerable"]) >= 5
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** `evals/metrics.py`:
```python
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
```
Then AUTHOR the 3 corpus docs and the 30 gold cases (run the chunker locally to grab fingerprints; verify every `relevant_fps` value actually appears in that doc's chunk fingerprints via a small throwaway script; delete the script after).
- [ ] **Step 4:** tests pass (incl. `test_seed_gold_loads`); ruff clean.
- [ ] **Step 5:** commit `feat: eval metrics + authored seed corpus and 30-case gold set`.

---

### Task 11: Eval runner + README report

**Files:** Create `evals/run.py`, `evals/report.py`, `tests/test_eval_run.py`.

**Interfaces:**
- `evals/run.py`: `run_eval(models: list[str], gold: list[dict], out_path: str, *, embedder_factory=get_embedder, chat_factory=get_chat, sleep_s=2.0) -> dict` — ONE ingest+embed pass per corpus doc (cache: doc → (chunks, index) built once, reused across models/cases); per (model, case): answer via `answer_question`, record `{model, case_id, recall5, mrr, faithful (bool), refused_correctly (bool|None), latency_s, input_tokens, output_tokens, error}`; resume by (model, case_id); atomic writes (`.tmp` + `os.replace`); corrupt file → warn + fresh. CLI `python -m evals.run --models a,b --out results.json [--gold path] [--sleep 2]`.
  - Per-case grading: retrieved_fps = fingerprints of retrieved chunks (`fingerprint(chunk.doc_id, chunk.page, chunk.text)`); answerable → recall/mrr/faithful; unanswerable → `refused_correctly = res.refused`, recall/mrr = None.
- `evals/report.py`: `summarize(records) -> list[dict]` per model: `recall_at_5` (mean over answerable), `mrr` (mean), `faithfulness` (fraction), `refusal_acc` (fraction of unanswerable refused), `p50_latency`; `to_markdown` header `| Model | Recall@5 | MRR | Faithful | Refusal acc | p50 s |`; `splice_readme` between `<!-- evals:start -->`/`<!-- evals:end -->` (ValueError if missing). CLI `python -m evals.report results.json [--readme README.md]`.

- [ ] **Step 1: failing tests** — `tests/test_eval_run.py` (fake factories; no network):
```python
import json

from evals.report import splice_readme, summarize, to_markdown
from evals.run import run_eval
from doclens.types import Usage


class FakeChat:
    def complete(self, messages, model):
        return "The spec version is 3 [p.1].", Usage(5, 2)


class FakeEmb:
    def embed(self, texts, model):
        return [[1.0, 0.0] for _ in texts]


def gold_cases():
    return [
        {"id": "g1", "doc": "rfc-style-spec.md", "question": "what version?",
         "relevant_fps": ["WILL_BE_REPLACED"], "expected_facts": ["version is 3"],
         "answerable": True},
        {"id": "g2", "doc": "rfc-style-spec.md", "question": "who is the CEO?",
         "relevant_fps": [], "expected_facts": [], "answerable": False},
    ]


def test_run_eval_and_summarize(tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    cases = gold_cases()
    results = run_eval(["gemini-3.1-flash-lite"], cases, str(out),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    records = results["records"]
    assert len(records) == 2
    g1 = next(r for r in records if r["case_id"] == "g1")
    assert g1["error"] is None and isinstance(g1["recall5"], float)
    g2 = next(r for r in records if r["case_id"] == "g2")
    assert g2["recall5"] is None and g2["refused_correctly"] in (True, False)
    # resume: second run does nothing new
    n_before = len(json.loads(out.read_text())["records"])
    run_eval(["gemini-3.1-flash-lite"], cases, str(out),
             embedder_factory=lambda **kw: (FakeEmb(), "e"),
             chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    assert len(json.loads(out.read_text())["records"]) == n_before


def test_report_table_and_splice():
    records = [
        {"model": "m", "case_id": "a", "recall5": 1.0, "mrr": 1.0, "faithful": True,
         "refused_correctly": None, "latency_s": 1.0, "input_tokens": 1,
         "output_tokens": 1, "error": None},
        {"model": "m", "case_id": "b", "recall5": None, "mrr": None, "faithful": None,
         "refused_correctly": True, "latency_s": 2.0, "input_tokens": 1,
         "output_tokens": 1, "error": None},
    ]
    (s,) = summarize(records)
    assert s["recall_at_5"] == 1.0 and s["refusal_acc"] == 1.0
    md = to_markdown([s])
    assert "| Model | Recall@5 |" in md and "| m |" in md
    spliced = splice_readme("a\n<!-- evals:start -->\nold\n<!-- evals:end -->\nb", "T")
    assert "T" in spliced and "old" not in spliced
```
Note: `run_eval` must ingest the real corpus docs from `evals/corpus/` — for `g1`, replace `"WILL_BE_REPLACED"` at test setup with the actual first-chunk fingerprint: compute it in the test via `ingest_file` + `chunk_document` + `fingerprint` so the test stays hermetic and correct against the committed corpus.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** both modules following the repolens `run.py`/`report.py` structure (sequential loop, resume set, atomic write, corrupt-file recovery, mean/median aggregation). Corpus cache: module-level per-invocation dict `{doc_filename: (chunks, index, retrieved_pages_lookup)}` built with the REAL pipeline (ingest_file → chunk_document → embedder.embed → VectorIndex).
- [ ] **Step 4:** tests pass; full suite; ruff clean.
- [ ] **Step 5:** commit `feat: resumable eval runner + README report (recall/MRR/faithfulness/refusals)`.

---

### Task 12: README + CI + live smoke

**Files:** Create `README.md`, `.github/workflows/ci.yml`.

- [ ] **Step 1:** README (real prose, fact-checked against code): pitch ("Upload a PDF or paste a link. Ask questions. Get answers cited to the page — with retrieval quality measured, not vibed."), quickstart (`pip install -e . && set GEMINI_API_KEY=… && doclens ask paper.pdf "what's the main result?"`), architecture mermaid (ingest → chunk → embed → index → answer), evals section with markers + methodology, design decisions (hand-built index rationale, fingerprint labels, refusal threshold, SSRF guard, original authored corpus), caps table, sibling link to repolens, MIT.
- [ ] **Step 2:** `.github/workflows/ci.yml` — same as repolens (checkout@v4, setup-python@v5 py3.12, `pip install -e .[dev]`, `ruff check .`, `python -m pytest -q`).
- [ ] **Step 3:** gates: full suite + ruff.
- [ ] **Step 4 (HUMAN/coordinator):** live smoke with real `GEMINI_API_KEY`: `doclens ask evals/corpus/device-manual.md "..."` then a real arXiv PDF URL; then `python -m evals.run --models gemini-3.1-flash-lite --out results.json` + `python -m evals.report results.json --readme README.md`.
- [ ] **Step 5:** commit `docs: README with eval markers + CI`.

## Verification (whole plan)

1. Full suite green (~45+ tests), ruff clean.
2. `doclens ask <local md> "<q>"` answers with `[p.N]` citations live.
3. URL ingest works on a real public page; private IP refused.
4. Eval run produces real README table.

## Out of scope (Plan B)

FastAPI server, SSE, sessions, rate caps, frontend, Docker/Render deploy, BYO-key UI.
