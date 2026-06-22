"""v1.8 — playground live async toggle (/api/background).

The user must be able to switch async (background companions) on/off mid-session,
at any time. chat() reads agent._background_enabled fresh each turn, so the
endpoint just flips that flag (and the session setting) and the next turn obeys.
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
            background=False,  # start inline; the endpoint flips it live
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


def test_background_toggle_flips_live(monkeypatch):
    client, server = _client(monkeypatch)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    assert sess.agent._background_enabled is False  # fake started inline

    # Turn async ON live.
    r = client.post("/api/background", json={"session_id": sid, "on": True})
    assert r.json() == {"ok": True, "on": True}
    assert sess.agent._background_enabled is True
    assert sess.settings["background"] is True

    # Turn async OFF again.
    r = client.post("/api/background", json={"session_id": sid, "on": False})
    assert r.json() == {"ok": True, "on": False}
    assert sess.agent._background_enabled is False
    assert sess.settings["background"] is False


def test_background_toggle_unknown_session_errors(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/api/background", json={"session_id": "nope", "on": True})
    assert "error" in r.json()
