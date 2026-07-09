"""v1.12 Stage B4 — playground live LLM-4 visualizer toggle (/api/visualization)
+ the session-export viz note.

The user must be able to switch the inline visualizer on/off mid-session. The
chat marker-extraction seam and the deep-research report hook both read
``config.visualization.enabled`` fresh, so the endpoint just flips that flag (and
the session setting); the next turn obeys. Mirrors the /api/long_term test.

``build_export_markdown`` must also surface a rendered viz as a link where its
⟦viz:…⟧ placeholder sat, so a handed-off export records the visual.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock import Sherlock  # noqa: E402


def _make_fake_build():
    def fake_build_agent(session, system_prompt, settings):
        session.settings = settings or {}
        session.system_prompt = system_prompt or "…"
        agent = Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            inference_chat=lambda m: "{}",
            system_prompt=system_prompt or "…",
            storage_dir=session.storage_dir or None,
            embedding="fake",
            background=False,
            main_search_engine="disabled",
            inference_search_engine="disabled",
        )
        agent.set_event_sink(session.emit)
        session.agent = agent
        return agent

    return fake_build_agent


def _client(monkeypatch):
    import playground.server as server

    monkeypatch.setattr(server, "build_agent", _make_fake_build())
    return TestClient(server.app), server


def _start(client):
    r = client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "persona.", "settings": {}},
    )
    return r.json()["session_id"]


# ------------------------------------------------------------ live toggle


def test_visualization_toggle_flips_live(monkeypatch):
    client, server = _client(monkeypatch)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    # default construction (no visualization= passed) → dormant
    assert sess.agent.config.visualization.enabled is False

    r = client.post("/api/visualization", json={"session_id": sid, "on": True})
    assert r.json() == {"ok": True, "on": True}
    assert sess.agent.config.visualization.enabled is True
    assert sess.settings["visualization"] is True

    r = client.post("/api/visualization", json={"session_id": sid, "on": False})
    assert r.json() == {"ok": True, "on": False}
    assert sess.agent.config.visualization.enabled is False
    assert sess.settings["visualization"] is False


def test_visualization_toggle_unknown_session_errors(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/api/visualization", json={"session_id": "nope", "on": True})
    assert "error" in r.json()


# ------------------------------------------------------------ export note


def _export_session(server, events):
    """Build a bare Session with a scripted events_log and render the export md."""
    from playground.session import Session

    sess = Session(sid="exp123", models={"main": "gemini/x"}, loop=None, queue=None)
    sess.turn = 1
    sess.events_log = events
    return server.build_export_markdown(sess)


def test_export_includes_rendered_viz(monkeypatch):
    _, server = _client(monkeypatch)
    md = _export_session(
        server,
        [
            {"type": "turn.start", "turn": 1, "data": {"user_text": "sales?"}},
            {"type": "turn.completed", "turn": 1, "data": {"response_text": "Here: ⟦viz:t1-1⟧"}},
            {
                "type": "viz.pending",
                "turn": 1,
                "data": {"viz_id": "t1-1", "description": "bar chart of quarterly sales"},
            },
            {
                "type": "viz.rendered",
                "turn": 1,
                "data": {"viz_id": "t1-1", "path": "/tmp/x/viz/t1-1.html", "validated": "static"},
            },
        ],
    )
    assert "[📊 visualization: bar chart of quarterly sales](viz/t1-1.html)" in md


def test_export_no_viz_line_when_none_rendered(monkeypatch):
    _, server = _client(monkeypatch)
    md = _export_session(
        server,
        [
            {"type": "turn.start", "turn": 1, "data": {"user_text": "hi"}},
            {"type": "turn.completed", "turn": 1, "data": {"response_text": "hello"}},
        ],
    )
    assert "📊 visualization" not in md


def test_export_viz_deduped_per_turn(monkeypatch):
    """A viz_id that emits viz.rendered twice (e.g. static then a runtime re-render)
    surfaces as ONE export line, not a duplicate."""
    _, server = _client(monkeypatch)
    md = _export_session(
        server,
        [
            {"type": "turn.start", "turn": 1, "data": {"user_text": "sales?"}},
            {"type": "turn.completed", "turn": 1, "data": {"response_text": "⟦viz:t1-1⟧"}},
            {"type": "viz.pending", "turn": 1, "data": {"viz_id": "t1-1", "description": "chart"}},
            {"type": "viz.rendered", "turn": 1, "data": {"viz_id": "t1-1", "validated": "static"}},
            {"type": "viz.rendered", "turn": 1, "data": {"viz_id": "t1-1", "validated": "runtime"}},
        ],
    )
    assert md.count("(viz/t1-1.html)") == 1
