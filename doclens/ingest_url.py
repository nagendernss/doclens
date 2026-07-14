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
