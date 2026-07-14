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
