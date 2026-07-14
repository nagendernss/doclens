"""URL + HTML ingestion with an SSRF guard."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import httpx
from selectolax.parser import HTMLParser

from .ingest import IngestError, ingest_pdf_bytes, ingest_text
from .types import Document

MAX_URL_BYTES = 5 * 1024 * 1024
TIMEOUT_S = 15.0
MAX_REDIRECTS = 5
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_DROP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "iframe")


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    # is_global is the primary gate. is_private alone is not sufficient: it
    # misses ranges that are reserved-but-not-"private", most notably the
    # CGNAT / shared-address-space block 100.64.0.0/10 (RFC 6598), which
    # ISPs use to front many customers behind one public IP and which can
    # route to carrier-internal hosts. The explicit denies below are kept
    # as defense in depth alongside is_global.
    return addr.is_global and not (
        addr.is_loopback or addr.is_link_local or addr.is_reserved
        or addr.is_multicast or addr.is_unspecified
    )


def _default_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise IngestError(f"could not resolve {host}: {exc}") from exc
    return [info[4][0] for info in infos]


def _assert_public(host: str, resolver) -> list[str]:
    """Validate every IP `host` resolves to is public; return that IP list.

    The caller reuses the returned IPs to pin the connection (see
    `_pin_to_ip`) so validation and connection use the *same* lookup.
    """
    ips = resolver(host)
    if not ips or not all(_is_public_ip(ip) for ip in ips):
        raise IngestError("refusing private/internal address")
    return ips


def _pin_to_ip(url: str, host: str, ips: list[str]) -> tuple[str, dict[str, str]]:
    """Rewrite `url`'s host to a pre-validated IP, preserving the Host header.

    Why: `_assert_public` and the HTTP client's own connect-time DNS lookup
    are two *independent* resolutions of the same hostname. An attacker who
    controls DNS for `host` can answer the validation lookup with a public
    IP and the connect-time lookup (moments later) with a private one --
    classic DNS-rebinding TOCTOU. Pinning collapses this to one lookup: we
    resolve once, validate it, and force the actual connection to the exact
    IP we validated.

    Caveats (accepted, not solved here):
    - For https:// URLs, rewriting the URL's host to the IP literal would,
      on its own, also repoint TLS SNI/certificate-hostname verification at
      that IP literal -- failing cert verification against any real HTTPS
      server. This function does not solve that by itself: the caller
      (`ingest_url`) pairs this rewrite with an `sni_hostname` request
      extension set to the original hostname, so the socket still connects
      to the pinned IP (DNS-rebind protection intact) while TLS SNI and
      certificate verification use the real hostname.
    - This path only runs for the client we construct ourselves
      (`own_client`, see `ingest_url`). Callers that inject their own
      `httpx.Client` (every test in this suite uses `httpx.MockTransport`,
      which never touches DNS or real sockets) skip this rewrite entirely
      -- pinning a mock's URL would only break the test's own routing, and
      the resolver seam already gives those tests full control over what
      "DNS" resolves to.
    - Host matching is case-insensitive; URLs with mixed-case hostnames are
      correctly pinned to the IP.
    """
    parts = urlsplit(url)
    netloc = parts.netloc

    # Parse netloc to extract userinfo, host portion, and port.
    # Format: [userinfo@]host[:port], where host may be an IPv6 literal [...]
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        userinfo = userinfo + "@"
    else:
        userinfo = ""
        hostport = netloc

    # Split hostport into host and port, handling IPv6 literals.
    if hostport.startswith("["):
        # IPv6 literal in the current URL (e.g., "[::1]:8080" or "[::1]")
        if "]:" in hostport:
            host_part, port = hostport.rsplit(":", 1)
            port = ":" + port
        else:
            host_part = hostport
            port = ""
    else:
        # IPv4 or hostname
        if ":" in hostport:
            host_part, port = hostport.rsplit(":", 1)
            port = ":" + port
        else:
            host_part = hostport
            port = ""

    # Case-insensitive host matching; replace only if it matches.
    if host_part.lower() == host.lower():
        # Bracket IPv6 IP literals; leave IPv4 as-is.
        ip_literal = f"[{ips[0]}]" if ":" in ips[0] else ips[0]
        new_netloc = userinfo + ip_literal + port

        # Rebuild the URL with the pinned IP in the netloc.
        new_parts = (parts.scheme, new_netloc, parts.path, parts.query, parts.fragment)
        fetch_url = urlunsplit(new_parts)
    else:
        # Host doesn't match, return original URL.
        fetch_url = url

    # Host header: original host + port (if present in the original URL).
    host_header = host + port

    return fetch_url, {"Host": host_header}


def ingest_html(html: str, source: str) -> Document:
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


def ingest_url(url: str, client: httpx.Client | None = None, resolver=None) -> Document:
    resolver = resolver or _default_resolver
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=False)
    current_url = url
    try:
        for _ in range(MAX_REDIRECTS):
            parsed = urlparse(current_url)
            if parsed.scheme not in ("http", "https"):
                raise IngestError(f"unsupported scheme {parsed.scheme!r}")
            if not parsed.hostname:
                raise IngestError("no host in URL")
            # Every hop -- the initial URL and every redirect target -- is
            # validated here BEFORE it is contacted. This closes the
            # redirect-chain bypass where only the final URL was re-checked
            # and a hop through (or bouncing off) a private host in between
            # was invisible to the guard.
            ips = _assert_public(parsed.hostname, resolver)

            if own_client:
                fetch_url, extra_headers = _pin_to_ip(current_url, parsed.hostname, ips)
            else:
                fetch_url, extra_headers = current_url, {}

            # sni_hostname: on the own_client path, `fetch_url` points at
            # the validated IP literal (see `_pin_to_ip`), so the TLS layer
            # would otherwise default to using that IP for both SNI and
            # certificate-hostname verification -- which fails against any
            # real HTTPS server. Passing the original hostname as the
            # `sni_hostname` request extension makes httpcore open the TCP
            # connection to the pinned IP (DNS-rebind protection intact)
            # while presenting and verifying the real hostname over TLS.
            # This is a no-op for the injected-client path (every test in
            # this suite): {} adds no extension, leaving MockTransport's
            # own routing untouched.
            extensions = {"sni_hostname": parsed.hostname} if own_client else {}

            # follow_redirects=False per-request (belt-and-suspenders on top
            # of how `client` was constructed): we must see raw 3xx
            # responses ourselves so each hop gets validated above before
            # it's contacted -- if httpx followed redirects internally we'd
            # never get the chance.
            with client.stream("GET", fetch_url, headers=extra_headers,
                                follow_redirects=False, extensions=extensions) as resp:
                location = resp.headers.get("location")
                if resp.status_code in _REDIRECT_STATUSES and location:
                    current_url = urljoin(current_url, location)
                    continue

                if resp.status_code >= 400:
                    raise IngestError(f"fetch failed with HTTP {resp.status_code}")

                # Stream + cap instead of buffering the whole body first, so
                # an oversized response is aborted as soon as it crosses the
                # cap rather than after it's fully downloaded into memory.
                chunks = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > MAX_URL_BYTES:
                        raise IngestError("document over the 5 MB URL cap")
                    chunks.append(chunk)
                body = b"".join(chunks)

                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype == "application/pdf" or current_url.lower().endswith(".pdf"):
                    return ingest_pdf_bytes(body, url)
                if ctype in ("text/html", "application/xhtml+xml", "text/plain", ""):
                    if ctype == "text/plain":
                        return ingest_text(body.decode(resp.encoding or "utf-8", "replace"), url)
                    return ingest_html(body.decode(resp.encoding or "utf-8", "replace"), url)
                raise IngestError(f"unsupported content type {ctype!r}")
        raise IngestError(f"too many redirects (max {MAX_REDIRECTS})")
    except httpx.HTTPError as exc:
        raise IngestError(f"fetch failed: {exc}") from exc
    finally:
        if own_client:
            client.close()
