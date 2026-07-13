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
