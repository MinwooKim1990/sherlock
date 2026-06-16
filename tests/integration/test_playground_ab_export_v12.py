"""v1.2 — playground A/B comparison mode + session export.

Same pattern as test_playground_deep_research_v07: a real FastAPI TestClient
with `build_agent` swapped for a scripted Sherlock.with_callable agent, plus a
monkeypatched `playground.providers._call_litellm` so the bare-model baseline
needs no key / network. Verifies: mode="both" returns reply + baseline with
history continuity, mode="single" skips the agent entirely, the default mode
stays byte-compatible, /api/export renders the markdown doc, and the
events_log stays bounded.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock import Sherlock  # noqa: E402


class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


MAIN_CALLS: list = []  # every prompt the scripted LLM-1 receives


def _main(messages):
    MAIN_CALLS.append(messages)
    return "sherlock answer."


def _make_fake_build(captured):
    def fake_build_agent(session, system_prompt, settings):
        MAIN_CALLS.clear()
        session.settings = settings or {}
        session.system_prompt = system_prompt or "You are a helpful assistant."
        agent = Sherlock.with_callable(
            main_chat=_main,
            inference_chat=lambda m: json.dumps(["q1?", "q2?", "q3?"]),
            system_prompt=system_prompt or "…",
            storage_dir=session.storage_dir or None,
            embedding="fake",
            background=False,
            main_search_engine="disabled",
            inference_search_engine="disabled",
        )

        def sink(ev):
            captured.append(ev)
            session.emit(ev)

        agent.set_event_sink(sink)
        session.agent = agent
        return agent

    return fake_build_agent


def _no_search(session, message):
    return ""


def _client(monkeypatch, captured):
    import playground.server as server

    monkeypatch.setattr(server, "build_agent", _make_fake_build(captured))
    import playground.providers as prov_mod

    monkeypatch.setattr(prov_mod, "_baseline_search_block", _no_search)
    return TestClient(server.app), server


def _patch_litellm(monkeypatch):
    """Canned bare-model completions; records every (model, messages) call."""
    import playground.providers as prov

    calls: list[dict] = []

    def fake_call_litellm(model, messages, **extra):
        calls.append({"model": model, "messages": [dict(m) for m in messages], "extra": extra})
        return _FakeCompletion(f"bare answer #{len(calls)}")

    monkeypatch.setattr(prov, "_call_litellm", fake_call_litellm)
    return calls


def _start_session(client):
    r = client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "persona.", "settings": {}},
    )
    return r.json()["session_id"]


def test_mode_both_reply_baseline_and_history_continuity(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    calls = _patch_litellm(monkeypatch)
    sid = _start_session(client)

    r = client.post("/api/chat", json={"session_id": sid, "message": "hello there", "mode": "both"})
    j = r.json()
    assert j["reply"] == "sherlock answer."
    assert isinstance(j["latency_ms"], int) and j["latency_ms"] >= 0
    assert j["baseline"]["text"] == "bare answer #1"
    assert j["baseline"]["error"] is None
    assert j["baseline"]["prompt_tokens"] == 11
    assert j["baseline"]["completion_tokens"] == 7

    sess = server.SESSIONS[sid]
    assert sess.baseline_history == [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "bare answer #1"},
    ]

    # Second turn: the baseline prompt must carry the FIRST turn's exchange.
    client.post("/api/chat", json={"session_id": sid, "message": "second question", "mode": "both"})
    msgs = calls[1]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("persona.")
    assert "Today is" in msgs[0]["content"]  # fair baseline knows the date
    assert {"role": "user", "content": "hello there"} in msgs
    assert {"role": "assistant", "content": "bare answer #1"} in msgs
    assert msgs[-1] == {"role": "user", "content": "second question"}
    assert len(sess.baseline_history) == 4
    assert sess.baseline_tokens == {"in": 22, "out": 14}
    assert any(e["type"] == "baseline.reply" for e in sess.events_log)


def test_mode_single_skips_agent(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    calls = _patch_litellm(monkeypatch)
    sid = _start_session(client)

    before = len(MAIN_CALLS)
    r = client.post("/api/chat", json={"session_id": sid, "message": "solo run", "mode": "single"})
    j = r.json()
    assert j["reply"] is None
    assert j["baseline"]["text"] == "bare answer #1"
    assert len(MAIN_CALLS) == before, "agent.chat ran in single mode"
    assert len(calls) == 1
    assert server.SESSIONS[sid].baseline_history[-1]["content"] == "bare answer #1"


def test_default_mode_back_compat(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    calls = _patch_litellm(monkeypatch)
    sid = _start_session(client)

    r = client.post("/api/chat", json={"session_id": sid, "message": "plain hi"})
    j = r.json()
    assert j["reply"] == "sherlock answer."
    assert "baseline" not in j
    assert calls == [], "baseline ran without being asked"
    assert server.SESSIONS[sid].baseline_history == []


def test_export_markdown(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    _patch_litellm(monkeypatch)
    sid = _start_session(client)
    client.post(
        "/api/chat", json={"session_id": sid, "message": "export me please", "mode": "both"}
    )

    r = client.get("/api/export", params={"session_id": sid})
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert f"sherlock-session-{sid}.md" in r.headers.get("content-disposition", "")
    body = r.text
    assert f"# Sherlock session export — {sid}" in body
    assert "export me please" in body  # user message
    assert "sherlock answer." in body  # LLM-1 reply
    assert "LLM-2" in body
    assert "LLM-3" in body
    assert "Single LLM" in body  # baseline section (A/B was used)
    assert "bare answer #1" in body
    assert "## Session totals" in body

    # Unknown session → error JSON, not a 500.
    r2 = client.get("/api/export", params={"session_id": "nope"})
    assert r2.json()["error"]


def test_export_sherlock_latency_and_no_unknown_usage(monkeypatch):
    """v1.3: the Sherlock line carries '⏱ Xms' (via the synthetic
    sherlock.latency event) and unknown/zero usage renders '—', never '?/?'."""
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    _patch_litellm(monkeypatch)
    sid = _start_session(client)
    client.post("/api/chat", json={"session_id": sid, "message": "latency check", "mode": "both"})

    sess = server.SESSIONS[sid]
    lat_events = [e for e in sess.events_log if e.get("type") == "sherlock.latency"]
    assert lat_events and lat_events[-1]["data"]["latency_ms"] >= 0

    body = client.get("/api/export", params={"session_id": sid}).text
    sherlock_lines = [ln for ln in body.splitlines() if ln.startswith("**Sherlock (LLM-1)")]
    assert sherlock_lines, "no Sherlock line in export"
    assert all("⏱" in ln and "ms" in ln for ln in sherlock_lines)
    assert "?/?" not in body
    # The scripted agent reports zero usage → the per-turn tokens line shows '—'.
    assert "- tokens in/out: —" in body


def test_role_callable_accepts_cache_hints(monkeypatch):
    """v1.3: make_role_callable's _call takes the cache_hints kwarg (so
    sherlock's CallableProvider passes hints through); for anthropic the hinted
    message becomes cache_control content blocks, elsewhere hints are ignored."""
    import inspect

    import playground.providers as prov

    calls = _patch_litellm(monkeypatch)
    emitted: list[dict] = []

    class _Sess:
        models = {"main": {"provider": "anthropic", "model": "claude-test"}}
        providers = {"anthropic": {"api_key": "k"}}
        settings: dict = {}
        turn = 1

    call = prov.make_role_callable("main", _Sess(), emitted.append)
    assert "cache_hints" in inspect.signature(call).parameters

    msgs = [
        {"role": "system", "content": "STABLE-PREFIX volatile tail"},
        {"role": "user", "content": "hi"},
    ]
    # Plain call (no hints) keeps the byte-identical payload.
    resp = call(msgs)
    assert "wrapper-error" not in resp.text
    assert calls[0]["messages"] == msgs

    # Hinted call → anthropic: stable prefix split off with cache_control.
    resp = call(msgs, cache_hints={"stable_prefix_chars": {0: 13}})
    assert "wrapper-error" not in resp.text
    sent = calls[1]["messages"]
    blocks = sent[0]["content"]
    assert isinstance(blocks, list)
    assert blocks[0]["text"] == "STABLE-PREFIX"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1] == {"type": "text", "text": " volatile tail"}
    assert sent[1] == {"role": "user", "content": "hi"}
    assert emitted and all(e["type"] == "llm.call" for e in emitted)

    # Non-anthropic provider: hints ignored, payload untouched.
    class _GemSess(_Sess):
        models = {"main": {"provider": "gemini", "model": "g-test"}}
        providers = {"gemini": {"api_key": "k"}}

    gcall = prov.make_role_callable("main", _GemSess(), emitted.append)
    resp = gcall(msgs, cache_hints={"stable_prefix_chars": {0: 13}})
    assert "wrapper-error" not in resp.text
    assert calls[2]["messages"] == msgs


def test_events_log_capped(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    sid = _start_session(client)
    sess = server.SESSIONS[sid]

    for i in range(20_500):
        sess.emit({"type": "synthetic", "actor": "system", "turn": 0, "data": {"i": i}})
    assert len(sess.events_log) <= 20_000
    assert sess.events_log[-1]["data"]["i"] == 20_499  # newest kept, oldest dropped


def test_fair_baseline_gets_search_results_and_date(monkeypatch):
    """The single-LLM baseline is a FAIR control: same search engine (one naive
    pass with the raw user message) + today's date — so the A/B isolates
    Sherlock's curation, not tool access."""
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    import playground.providers as prov_mod

    # one canned search pass instead of the network engine
    monkeypatch.setattr(
        prov_mod,
        "_baseline_search_block",
        lambda session, message: "Web search results for the user's message:\n- T — https://x.com/1: SNIPPET_MARKER",
    )
    calls: list[dict] = []

    class _Resp:
        class _C:
            class _M:
                content = "baseline answer"

            message = _M()

        choices = [_C()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    def fake_litellm(model, messages, **extra):
        calls.append({"model": model, "messages": messages})
        return _Resp()

    monkeypatch.setattr(prov_mod, "_call_litellm", fake_litellm)

    r = client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "persona.", "settings": {}},
    )
    sid = r.json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "hello", "mode": "single"})
    assert r.json()["baseline"]["searched"] is True
    user_msg = calls[0]["messages"][-1]["content"]
    assert "SNIPPET_MARKER" in user_msg, "search results must reach the baseline prompt"
    assert user_msg.startswith("hello")
    # opt-out flag disables the search pass
    r = client.post(
        "/api/chat",
        json={"session_id": sid, "message": "again", "mode": "single", "baseline_search": False},
    )
    assert r.json()["baseline"]["searched"] is False
    assert "SNIPPET_MARKER" not in calls[-1]["messages"][-1]["content"]
    # history kept the PLAIN user message (no search residue accumulates)
    sess = server.SESSIONS[sid]
    assert {"role": "user", "content": "hello"} in sess.baseline_history
