"""Tests for the `/` index route and `/static` mount added in Task 5 (frontend).

Task 3 deliberately deferred these (see task-3-web-report.md Concerns): its own
interface list only covered create_app/api routes, so the mount is wired here,
alongside the frontend files that make it meaningful.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from doclens.server import create_app


def test_root_serves_index_html_containing_doclens():
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/")
    assert r.status_code == 200
    assert "doclens" in r.text


def test_static_app_js_served():
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/static/app.js")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_static_style_css_served():
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/static/style.css")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_no_static_mount_or_root_route_when_web_dir_missing(monkeypatch, tmp_path):
    """The mount is guarded by `if WEB_DIR.exists()` — prove the guard actually guards."""
    import doclens.server as srv

    monkeypatch.setattr(srv, "WEB_DIR", tmp_path / "no-such-web-dir")
    app = srv.create_app()
    c = TestClient(app, base_url="http://test")
    assert c.get("/").status_code == 404
    assert c.get("/static/app.js").status_code == 404


def test_api_routes_unaffected_by_static_mount():
    """Sanity check per the task's gate note: adding / + /static must not disturb /api/models."""
    app = create_app()
    c = TestClient(app, base_url="http://test")
    r = c.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "default" in body
