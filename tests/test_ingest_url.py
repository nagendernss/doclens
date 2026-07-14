import httpx
import pytest

from doclens.ingest_url import _is_public_ip, _pin_to_ip, ingest_html, ingest_url
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


def test_is_public_ip_cgnat():
    assert _is_public_ip("100.64.0.1") is False
    assert _is_public_ip("93.184.216.34") is True


def test_redirect_to_private_blocked():
    hit_hosts = []

    def handler(request):
        hit_hosts.append(request.url.host)
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://internal/secret"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>secret data</p>")

    def resolver(host):
        return {"example.com": ["93.184.216.34"], "internal": ["10.0.0.5"]}[host]

    with pytest.raises(IngestError, match="private|internal"):
        ingest_url("https://example.com/start", client=make_client(handler), resolver=resolver)

    assert "internal" not in hit_hosts


def test_redirect_bounceback_blocked():
    hit_hosts = []

    def handler(request):
        hit_hosts.append(request.url.host)
        if request.url.host == "example.com" and request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://internal/x"})
        if request.url.host == "internal":
            return httpx.Response(302, headers={"location": "http://example.com/final"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>final content</p>")

    def resolver(host):
        return {"example.com": ["93.184.216.34"], "internal": ["10.0.0.5"]}[host]

    with pytest.raises(IngestError, match="private|internal"):
        ingest_url("https://example.com/start", client=make_client(handler), resolver=resolver)

    assert "internal" not in hit_hosts


def test_redirect_to_public_followed():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.com/final"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>final page content</p>")

    doc = ingest_url("https://example.com/start", client=make_client(handler),
                     resolver=lambda host: ["93.184.216.34"])
    assert "final page content" in doc.pages[0].text


def test_too_many_redirects():
    def handler(request):
        n = int(request.url.path.strip("/").removeprefix("hop") or 0)
        if n < 6:
            return httpx.Response(302, headers={"location": f"https://example.com/hop{n + 1}"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>done</p>")

    with pytest.raises(IngestError, match="redirects"):
        ingest_url("https://example.com/hop0", client=make_client(handler),
                   resolver=lambda host: ["93.184.216.34"])


def test_stream_size_cap():
    pulled = []
    chunk = b"a" * (1024 * 1024)

    def gen():
        for i in range(20):
            pulled.append(i)
            yield chunk

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=gen())

    with pytest.raises(IngestError, match="5 MB"):
        ingest_url("https://example.com/big", client=make_client(handler),
                   resolver=lambda host: ["93.184.216.34"])

    assert len(pulled) < 20, "size cap must abort mid-stream, not after buffering the full body"


def test_pin_to_ip_plain():
    fetch_url, headers = _pin_to_ip("http://example.com/p", "example.com", ["93.184.216.34"])
    assert fetch_url == "http://93.184.216.34/p"
    assert headers["Host"] == "example.com"


def test_pin_to_ip_mixed_case():
    fetch_url, headers = _pin_to_ip("http://Example.COM/p", "example.com", ["93.184.216.34"])
    assert fetch_url == "http://93.184.216.34/p"
    assert headers["Host"] == "example.com"


def test_pin_to_ip_port_userinfo():
    fetch_url, headers = _pin_to_ip("http://u:p@example.com:8080/x", "example.com", ["93.184.216.34"])
    assert fetch_url == "http://u:p@93.184.216.34:8080/x"
    assert headers["Host"] == "example.com:8080"


def test_pin_to_ip_ipv6():
    fetch_url, headers = _pin_to_ip("http://example.com/x", "example.com", ["2606:2800:220:1::"])
    assert fetch_url == "http://[2606:2800:220:1::]/x"
    assert headers["Host"] == "example.com"


def test_pinned_fetch_sets_sni_hostname(monkeypatch):
    # own_client path: pinning the socket to the validated IP must not also
    # repoint TLS SNI/cert-hostname verification at the IP literal, or every
    # real HTTPS fetch fails cert verification (the production bug). The
    # fetch must reach the pinned IP while telling httpx to do the TLS
    # handshake against the original hostname.
    captured = {}

    def handler(request):
        captured["url_host"] = request.url.host
        captured["host_header"] = request.headers["host"]
        captured["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>pinned fetch content</p>")

    real_client_cls = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: real_client_cls(transport=httpx.MockTransport(handler)),
    )

    doc = ingest_url("https://example.com/secure", resolver=lambda host: ["93.184.216.34"])

    assert captured["url_host"] == "93.184.216.34"
    assert captured["host_header"] == "example.com"
    assert captured["sni_hostname"] == "example.com"
    assert "pinned fetch content" in doc.pages[0].text


def test_injected_client_stream_extensions_empty():
    # own_client=False (every real caller in this suite uses MockTransport):
    # pinning is skipped, so there must be no sni_hostname override either --
    # an injected client's own request routing must be left untouched.
    captured = {}

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>hello url content</p>")

    client = make_client(handler)
    original_stream = client.stream

    def spy_stream(*args, **kwargs):
        captured["extensions"] = kwargs.get("extensions")
        return original_stream(*args, **kwargs)

    client.stream = spy_stream

    doc = ingest_url("https://example.com/page", client=client,
                     resolver=lambda host: ["93.184.216.34"])

    assert captured["extensions"] == {}
    assert "hello url content" in doc.pages[0].text


def test_sends_user_agent():
    from doclens.ingest_url import USER_AGENT
    recorded_ua = None

    def handler(request):
        nonlocal recorded_ua
        recorded_ua = request.headers.get("user-agent")
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>T</title><p>ua test</p>")

    doc = ingest_url("https://example.com/page", client=make_client(handler),
                     resolver=lambda host: ["93.184.216.34"])
    assert recorded_ua == USER_AGENT
    assert "doclens" in recorded_ua
    assert "ua test" in doc.pages[0].text
