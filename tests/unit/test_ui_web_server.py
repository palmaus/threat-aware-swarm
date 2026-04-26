import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from fastapi.responses import HTMLResponse  # noqa: E402

from ui.web_server import WebSession, create_app, handle_control_message  # noqa: E402


def test_web_session_telemetry_payload():
    session = WebSession(screen_size=200, max_steps=5, seed=0, fps=1)
    telemetry = session.get_telemetry()
    assert "left" in telemetry
    left = telemetry["left"]
    assert "agents" in left
    assert "stats" in left
    assert "field_size" in left


def test_web_ui_index():
    session = WebSession(screen_size=200, max_steps=5, seed=0, fps=1)
    app = create_app(session)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/")
    resp = route.endpoint()
    assert isinstance(resp, HTMLResponse)
    assert "Threat-Aware Swarm" in resp.body.decode("utf-8")
    body = resp.body.decode("utf-8")
    assert 'id="root"' in body
    assert "assets/" in body


def test_web_session_creates_compare_controller_lazily():
    session = WebSession(screen_size=200, max_steps=5, seed=7, fps=1, compare=False)
    try:
        assert set(session.controllers) == {"left"}
        assert session.get_state()["right"] is None

        handle_control_message(session, {"type": "control", "action": "compare", "value": True})

        assert set(session.controllers) == {"left", "right"}
        assert session.get_state()["right"] is not None
        assert session.seed == 7
    finally:
        session.close()
