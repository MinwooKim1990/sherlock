"""v1.12 Stage B1 — LLM-4 VISUALIZER end-to-end plumbing.

The enabled path exercised through the real agent (marker → placeholder swap,
``viz.pending`` events, stashed render jobs, TIER-1 guidance injection) on BOTH
the sync ``chat()`` and async ``achat()`` seams, plus the playground backend
wiring (v1.12 F1: build_agent ALWAYS builds a viz callable — with no viz model it
resolves the MAIN model under role="viz", staying non-streaming with actor
``llm4``; ``_ROLE_ACTOR`` maps the viz role to the ``llm4`` actor). The pure
parser + the off-state kill switch live in the unit suite (test_viz_markers_v112.py).
"""

from __future__ import annotations

import pytest

from sherlock import Sherlock


class _CapturingMain:
    """Sync main callable: records system prompts, replies with a fixed body."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system_prompts: list[str] = []

    def __call__(self, messages):
        for m in messages:
            role = m["role"] if isinstance(m, dict) else m.role
            content = m["content"] if isinstance(m, dict) else m.content
            if role == "system":
                self.system_prompts.append(content)
                break
        return self.reply


_TWO_MARKERS = (
    "First quarter:\n<<sherlock-viz: bar chart of Q1 sales | A 1, B 2>>\n"
    "And the trend:\n<<sherlock-viz: line chart of the trend | X 3, Y 4>>\nThat's it."
)


def _agent(tmp_path, name, *, main, visualization=True):
    return Sherlock.with_callable(
        main_chat=main,
        summary_chat=lambda m: "{}",
        inference_chat=lambda m: "{}",
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        background=False,
        companions_mode="off",
        visualization=visualization,
    )


# ---------------------------------------------------------------- enabled (sync)


def test_enabled_two_markers_become_placeholders_sync(tmp_path):
    main = _CapturingMain(_TWO_MARKERS)
    events: list[dict] = []
    agent = _agent(tmp_path, "sync", main=main, visualization=True)
    agent.set_event_sink(events.append)

    reply = agent.chat("show me the numbers")

    # exactly two placeholders, no raw markers survive
    assert reply.count("⟦viz:") == 2
    assert "<<sherlock-viz" not in reply

    # two viz.pending events, each carrying the required fields
    pending = [e for e in events if e["type"] == "viz.pending"]
    assert len(pending) == 2
    for e in pending:
        assert e["actor"] == "llm4"
        d = e["data"]
        assert set(("turn", "viz_id", "anchor", "description")).issubset(d)
        assert d["anchor"] in reply  # the emitted anchor is the one in the text

    # jobs stashed with the split descriptions/data hints intact
    jobs = agent._pending_viz_jobs
    assert len(jobs) == 2
    assert jobs[0]["description"] == "bar chart of Q1 sales"
    assert jobs[0]["data_hint"] == "A 1, B 2"
    assert jobs[1]["description"] == "line chart of the trend"
    # event viz_ids match the stashed jobs
    assert {e["data"]["viz_id"] for e in pending} == {j["viz_id"] for j in jobs}

    # TIER-1 guidance block was injected into the system prompt LLM-1 saw
    assert main.system_prompts
    assert any("Inline visualizations" in sp and "sherlock-viz" in sp for sp in main.system_prompts)


def test_enabled_stash_is_bounded_to_32(tmp_path):
    # Feed many turns, each emitting 1 marker; the pending stash must not grow
    # past 32 (oldest dropped). cap markers/reply is 3 by default, 1 per turn.
    main = _CapturingMain("<<sherlock-viz: tiny chart | n 1>>")
    agent = _agent(tmp_path, "bound", main=main, visualization=True)
    for i in range(40):
        agent.chat(f"turn {i}")
    assert len(agent._pending_viz_jobs) == 32


# --------------------------------------------------------------- enabled (achat)


@pytest.mark.asyncio
async def test_enabled_two_markers_achat_parity(tmp_path):
    class _AsyncMain(_CapturingMain):
        async def __call__(self, messages):  # type: ignore[override]
            return _CapturingMain.__call__(self, messages)

    main = _AsyncMain(_TWO_MARKERS)
    events: list[dict] = []
    agent = _agent(tmp_path, "async", main=main, visualization=True)
    agent.set_event_sink(events.append)

    reply = await agent.achat("show me the numbers")

    assert reply.count("⟦viz:") == 2
    assert "<<sherlock-viz" not in reply
    pending = [e for e in events if e["type"] == "viz.pending"]
    assert len(pending) == 2
    assert len(agent._pending_viz_jobs) == 2
    assert any("Inline visualizations" in sp for sp in main.system_prompts)


# ------------------------------------------------------------------- playground


def test_role_actor_maps_viz_to_llm4():
    from playground.providers import _ROLE_ACTOR

    assert _ROLE_ACTOR["viz"] == "llm4"


def _build_playground(monkeypatch, tmp_home, models, settings):
    """Call the REAL build_agent with fake role callables + HOME redirected,
    recording which roles a callable was requested for."""
    pytest.importorskip("fastapi")
    import playground.providers as providers
    import playground.session as session_mod

    requested_roles: list[str] = []

    def _fake_make(role, sess, emit):
        requested_roles.append(role)
        return lambda m: "ok"

    monkeypatch.setattr(providers, "make_role_callable", _fake_make)
    monkeypatch.setenv("HOME", str(tmp_home))
    sess = session_mod.Session(sid="s", models=models, loop=None, queue=None)
    hermetic = {"embedding": "fake", "search_engine": "off", "background": False}
    agent = session_mod.build_agent(sess, "sys", {**hermetic, **settings})
    return agent, requested_roles


def test_build_agent_builds_viz_callable_when_model_selected(monkeypatch, tmp_path):
    agent, roles = _build_playground(
        monkeypatch,
        tmp_path,
        models={"main": {"provider": "p", "model": "m"}, "viz": {"provider": "p", "model": "v"}},
        settings={"visualization": True},
    )
    assert "viz" in roles  # a viz callable was requested
    assert agent._viz_provider is not None  # and wired onto the agent
    assert agent.config.visualization.enabled is True


def test_build_agent_builds_viz_callable_even_without_model(monkeypatch, tmp_path):
    # v1.12 F1: the viz callable is ALWAYS built, even with no viz model selected,
    # so viz generation runs under role="viz" (non-streaming / actor llm4) instead
    # of leaking into the MAIN streaming path via the library's _viz_llm fallback.
    agent, roles = _build_playground(
        monkeypatch,
        tmp_path,
        models={"main": {"provider": "p", "model": "m"}},  # no viz entry
        settings={},  # visualization off
    )
    assert "viz" in roles  # a viz callable is requested regardless of viz model
    assert agent._viz_provider is not None  # wired onto the agent (uses main model)
    assert agent._viz_llm() is agent._viz_provider  # NOT the main provider
    # Off-state stays byte-identical: visualization disabled → the marker protocol
    # is dormant and the viz callable is never actually invoked.
    assert agent.config.visualization.enabled is False


def test_viz_callable_without_viz_model_is_nonstreaming_llm4(monkeypatch):
    # v1.12 F1 confirmation: a viz callable with NO dedicated viz model resolves to
    # the MAIN model but under role="viz" → the NON-streaming branch. It must NOT
    # emit any llm.delta (that would pollute the live chat bubble), and its llm.call
    # must carry role="viz" + actor="llm4" (so L4 spend is booked correctly).
    import playground.providers as prov

    class _Usage:
        prompt_tokens, completion_tokens, total_tokens = 5, 9, 14

    class _Msg:
        content = "<svg>chart</svg>"
        reasoning_content = None

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]
        usage = _Usage()

    nonstream_calls: list = []

    def _fake_call(model, messages, **extra):
        nonstream_calls.append(model)
        return _Resp()

    def _fake_stream(*a, **k):  # streaming path must never be taken for viz
        raise AssertionError("viz role must not stream")

    monkeypatch.setattr(prov, "_call_litellm", _fake_call)
    monkeypatch.setattr(prov, "_call_litellm_stream", _fake_stream)

    class _Sess:
        models = {"main": {"provider": "gemini", "model": "g-main"}}  # NO viz entry
        providers = {"gemini": {"api_key": "k"}}
        settings: dict = {}
        turn = 3
        agent = None  # _stopped() reads agent._stop_event → False when absent

    emitted: list[dict] = []
    call = prov.make_role_callable("viz", _Sess(), emitted.append)
    resp = call([{"role": "user", "content": "draw it"}])

    assert resp.text == "<svg>chart</svg>"
    assert nonstream_calls == ["gemini/g-main"]  # used the MAIN model, non-streaming
    assert not [e for e in emitted if e["type"] == "llm.delta"]  # no live-bubble leak
    calls = [e for e in emitted if e["type"] == "llm.call"]
    assert len(calls) == 1
    assert calls[0]["actor"] == "llm4"
    assert calls[0]["data"]["role"] == "viz"
