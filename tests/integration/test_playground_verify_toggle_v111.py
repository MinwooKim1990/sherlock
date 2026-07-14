"""v1.11 — playground deep-research VERIFY tier: live toggle + build wiring.

The v1.10 accuracy layer (LLM-2 faithfulness + consistency, opt-in LLM-3 web
re-check) was invisible and un-A/B-able in the playground: build_agent never
touched config.search and there was no live endpoint. This pins both: build_agent
now reads settings["deep_research_verify"], and /api/verify flips it mid-session.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock import Sherlock  # noqa: E402


# ---------------------------------------------------------------- build_agent
def test_build_agent_applies_verify_tier(monkeypatch):
    import playground.providers as providers
    import playground.session as session_mod

    monkeypatch.setattr(providers, "make_role_callable", lambda role, sess, emit: (lambda m: "ok"))

    def _agent(settings):
        sess = session_mod.Session(sid="s", models={}, loop=None, queue=None)
        return session_mod.build_agent(sess, "sys", settings)

    base = {"embedding": "fake", "search_engine": "off", "background": False}
    assert (
        _agent(
            {**base, "deep_research_verify": "faithfulness+web"}
        ).config.search.deep_research_verify
        == "faithfulness+web"
    )
    assert (
        _agent({**base, "deep_research_verify": "off"}).config.search.deep_research_verify == "off"
    )
    # unset → library default; invalid → falls through to the same default
    assert _agent(base).config.search.deep_research_verify == "faithfulness"
    assert (
        _agent({**base, "deep_research_verify": "bogus"}).config.search.deep_research_verify
        == "faithfulness"
    )


def test_build_agent_applies_bounded_deep_research_rounds(monkeypatch):
    import playground.providers as providers
    import playground.session as session_mod

    monkeypatch.setattr(providers, "make_role_callable", lambda role, sess, emit: (lambda m: "ok"))

    def _rounds(value):
        sess = session_mod.Session(sid="s", models={}, loop=None, queue=None)
        settings = {
            "embedding": "fake",
            "search_engine": "off",
            "background": False,
            "deep_research_max_rounds": value,
        }
        return session_mod.build_agent(sess, "sys", settings).config.search.deep_research_max_rounds

    assert _rounds(7) == 7
    assert _rounds(0) == 1
    assert _rounds(99) == 20
    assert _rounds("bad") == 20


# ---------------------------------------------------------------- /api/verify
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
        # apply the tier just like the real build_agent so the toggle has a baseline
        _vt = (settings or {}).get("deep_research_verify", "faithfulness")
        if _vt in ("off", "faithfulness", "faithfulness+web"):
            agent.config.search.deep_research_verify = _vt
        agent.set_event_sink(session.emit)
        session.agent = agent
        return agent

    return fake_build_agent


def _client(monkeypatch):
    import playground.server as server

    monkeypatch.setattr(server, "build_agent", _make_fake_build())
    return TestClient(server.app), server


def _start(client):
    return client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "p.", "settings": {}},
    ).json()["session_id"]


def test_verify_toggle_flips_live(monkeypatch):
    client, server = _client(monkeypatch)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    assert sess.agent.config.search.deep_research_verify == "faithfulness"  # default

    r = client.post("/api/verify", json={"session_id": sid, "tier": "off"})
    assert r.json() == {"ok": True, "tier": "off"}
    assert sess.agent.config.search.deep_research_verify == "off"
    assert sess.settings["deep_research_verify"] == "off"

    r = client.post("/api/verify", json={"session_id": sid, "tier": "faithfulness+web"})
    assert r.json() == {"ok": True, "tier": "faithfulness+web"}
    assert sess.agent.config.search.deep_research_verify == "faithfulness+web"


def test_verify_invalid_tier_errors(monkeypatch):
    client, server = _client(monkeypatch)
    sid = _start(client)
    r = client.post("/api/verify", json={"session_id": sid, "tier": "bogus"})
    assert "error" in r.json()
    # unchanged after a rejected tier
    assert server.SESSIONS[sid].agent.config.search.deep_research_verify == "faithfulness"


def test_verify_unknown_session_errors(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/api/verify", json={"session_id": "nope", "tier": "off"})
    assert "error" in r.json()


def test_deep_research_max_rounds_toggle_flips_live(monkeypatch):
    client, server = _client(monkeypatch)
    sid = _start(client)
    sess = server.SESSIONS[sid]

    r = client.post("/api/deep_research/max_rounds", json={"session_id": sid, "max_rounds": 6})
    assert r.json() == {"ok": True, "max_rounds": 6}
    assert sess.agent.config.search.deep_research_max_rounds == 6
    assert sess.settings["deep_research_max_rounds"] == 6


@pytest.mark.parametrize("value", [0, 21])
def test_deep_research_max_rounds_rejects_out_of_range(monkeypatch, value):
    client, server = _client(monkeypatch)
    sid = _start(client)

    r = client.post("/api/deep_research/max_rounds", json={"session_id": sid, "max_rounds": value})
    assert "error" in r.json()
    assert server.SESSIONS[sid].agent.config.search.deep_research_max_rounds == 20


def test_deep_research_max_rounds_unknown_session_errors(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post(
        "/api/deep_research/max_rounds",
        json={"session_id": "nope", "max_rounds": 5},
    )
    assert "error" in r.json()
