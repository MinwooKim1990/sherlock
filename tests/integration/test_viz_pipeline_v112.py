"""v1.12 Stage B2 — LLM-4 VISUALIZER render pipeline (end-to-end on the agent).

The async render job driven through a real ``Sherlock`` with a SCRIPTED fake
viz LLM: generation → static lint → self-review / repair rounds → validated
artifact + ``viz.rendered`` (or ``viz.failed``). Covers fence unwrapping, the
timeout/repair accounting, fire-and-forget async safety (chat/achat return
before the render finishes; the bg future is untouched), the job CONTEXT slice,
and the disabled kill switch. The pure lint table lives in the unit suite
(test_viz_lint_v112.py).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from sherlock import Sherlock
from sherlock.viz import VALIDATED_META

# --- artifact HTML fixtures (12 and 19 trace to the marker below) ---------- #

VALID = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>Q1 12</span><span>Q2 19</span></div>\n"
    "<script>window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');"
    "parent.postMessage({sherlockViz:'ready'}, '*');</script></body></html>"
)

# missing the ready signal → fails the static lint
BROKEN = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>Q1 12</span><span>Q2 19</span></div></body></html>"
)

MARKER_REPLY = (
    "Sales trend:\n<<sherlock-viz: bar chart of sales | Q1 12, Q2 19>>\nThat's the picture."
)


class _ScriptViz:
    """Scripted, thread-safe fake viz LLM. Records the prompt it saw each call
    and returns the next scripted response (or raises ``raise_exc``)."""

    def __init__(self, *responses, raise_exc=None, gate=None):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.raise_exc = raise_exc
        self._gate = gate
        self._lock = threading.Lock()

    def __call__(self, messages):
        if self._gate is not None:
            self._gate.wait(timeout=5)
        prompt = "\n".join((m["content"] if isinstance(m, dict) else m.content) for m in messages)
        with self._lock:
            self.prompts.append(prompt)
            if self.raise_exc is not None:
                raise self.raise_exc
            return self._responses.pop(0) if self._responses else "NO MORE"


def _agent(tmp_path, name, *, main, viz_chat=None, visualization=True):
    return Sherlock.with_callable(
        main_chat=main,
        summary_chat=lambda m: "{}",
        inference_chat=lambda m: "{}",
        viz_chat=viz_chat,
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        background=False,
        companions_mode="off",
        visualization=visualization,
    )


def _events_of(events, type_):
    return [e for e in events if e["type"] == type_]


# ----------------------------------------------------------- (a) valid first try


def test_valid_first_try_renders_and_persists(tmp_path):
    viz = _ScriptViz(VALID)
    events: list[dict] = []
    agent = _agent(tmp_path, "a", main=lambda m: MARKER_REPLY, viz_chat=viz)
    agent.set_event_sink(events.append)

    reply = agent.chat("show me")
    assert "⟦viz:t1-1⟧" in reply
    assert agent.wait_for_viz(timeout=5) is True

    rendered = _events_of(events, "viz.rendered")
    assert len(rendered) == 1
    d = rendered[0]["data"]
    assert d["viz_id"] == "t1-1"
    assert d["validated"] == "static"
    assert d["anchor"] == "⟦viz:t1-1⟧"
    assert d["turn"] == 1
    assert VALIDATED_META in d["html"]  # stamped before emit
    assert d["bytes"] == len(d["html"].encode("utf-8"))
    # valid first try short-circuits: exactly ONE viz LLM call (no self-review)
    assert len(viz.prompts) == 1
    # artifact persisted with the validated meta
    path = Path(d["path"])
    assert path.exists()
    assert path.name == "t1-1.html"
    assert VALIDATED_META in path.read_text(encoding="utf-8")
    # no failure event
    assert _events_of(events, "viz.failed") == []


# ------------------------------------------------------ (b) self-review recovers


def test_broken_then_self_review_fixes(tmp_path):
    viz = _ScriptViz(BROKEN, VALID)  # gen broken → self-review returns valid
    events: list[dict] = []
    agent = _agent(
        tmp_path,
        "b",
        main=lambda m: MARKER_REPLY,
        viz_chat=viz,
        visualization={"self_review_rounds": 1, "max_repair_rounds": 2},
    )
    agent.set_event_sink(events.append)

    agent.chat("show me")
    assert agent.wait_for_viz(timeout=5) is True

    assert len(_events_of(events, "viz.rendered")) == 1
    assert _events_of(events, "viz.failed") == []
    assert len(viz.prompts) == 2  # generation + one self-review
    # the self-review round is NOT a "repairing" event (that's for lint-error rounds)
    assert _events_of(events, "viz.repairing") == []


# --------------------------------------------------------- (c) all rounds broken


def test_all_broken_fails_no_artifact(tmp_path):
    viz = _ScriptViz(BROKEN, BROKEN, BROKEN)  # gen + repair1 + repair2 all broken
    events: list[dict] = []
    agent = _agent(
        tmp_path,
        "c",
        main=lambda m: MARKER_REPLY,
        viz_chat=viz,
        visualization={"self_review_rounds": 0, "max_repair_rounds": 2},
    )
    agent.set_event_sink(events.append)

    agent.chat("show me")
    assert agent.wait_for_viz(timeout=5) is True

    assert _events_of(events, "viz.rendered") == []
    failed = _events_of(events, "viz.failed")
    assert len(failed) == 1
    assert "ready signal" in failed[0]["data"]["reason"]
    # generation + 2 repair rounds = 3 calls; 2 repairing events
    assert len(viz.prompts) == 3
    assert len(_events_of(events, "viz.repairing")) == 2
    # NO artifact written on failure
    assert not (tmp_path / "c" / "viz" / "t1-1.html").exists()


# ------------------------------------------------------------ (d) provider raises


def test_provider_exception_fails_gracefully(tmp_path):
    viz = _ScriptViz(raise_exc=RuntimeError("boom"))
    events: list[dict] = []
    agent = _agent(tmp_path, "d", main=lambda m: MARKER_REPLY, viz_chat=viz)
    agent.set_event_sink(events.append)

    reply = agent.chat("show me")  # must not raise into the turn
    assert "⟦viz:t1-1⟧" in reply
    assert agent.wait_for_viz(timeout=5) is True

    failed = _events_of(events, "viz.failed")
    assert len(failed) == 1
    assert "RuntimeError" in failed[0]["data"]["reason"]
    assert _events_of(events, "viz.rendered") == []
    assert not (tmp_path / "d" / "viz" / "t1-1.html").exists()


# --------------------------------------------------------- (e) fence unwrapping


def test_fence_wrapped_output_unwrapped(tmp_path):
    viz = _ScriptViz(f"```html\n{VALID}\n```")
    events: list[dict] = []
    agent = _agent(tmp_path, "e", main=lambda m: MARKER_REPLY, viz_chat=viz)
    agent.set_event_sink(events.append)

    agent.chat("show me")
    assert agent.wait_for_viz(timeout=5) is True

    rendered = _events_of(events, "viz.rendered")
    assert len(rendered) == 1
    assert "```" not in rendered[0]["data"]["html"]  # fences stripped


# ---------------------------------------------------------- async safety (sync)


def test_submit_does_not_delay_chat_return(tmp_path):
    gate = threading.Event()  # the render blocks until we open the gate
    viz = _ScriptViz(VALID, gate=gate)
    events: list[dict] = []
    agent = _agent(tmp_path, "f", main=lambda m: MARKER_REPLY, viz_chat=viz)
    agent.set_event_sink(events.append)

    reply = agent.chat("show me")  # returns while the render is still gated
    assert "⟦viz:t1-1⟧" in reply
    # render is blocked → no rendered event yet
    assert _events_of(events, "viz.rendered") == []
    assert agent._viz_executor is not None  # dedicated pool spun up
    # the bg machinery is UNTOUCHED (background=False → no bg executor/future)
    assert agent._executor is None
    assert agent._bg_future is None
    assert agent.wait_for_background(timeout=1) is True

    gate.set()  # release the render
    assert agent.wait_for_viz(timeout=5) is True
    assert len(_events_of(events, "viz.rendered")) == 1


# ------------------------------------------------------------ achat parity


@pytest.mark.asyncio
async def test_achat_fire_and_forget_parity(tmp_path):
    class _AsyncMain:
        async def __call__(self, messages):
            return MARKER_REPLY

    viz = _ScriptViz(VALID)
    events: list[dict] = []
    agent = _agent(tmp_path, "g", main=_AsyncMain(), viz_chat=viz)
    agent.set_event_sink(events.append)

    reply = await agent.achat("show me")
    assert "⟦viz:t1-1⟧" in reply
    assert agent.wait_for_viz(timeout=5) is True
    assert len(_events_of(events, "viz.rendered")) == 1
    # achat sends viz renders to the dedicated pool, never the bg future
    assert agent._bg_future is None


# --------------------------------------------------------------- job context


def test_job_carries_context_and_prompt_uses_it(tmp_path):
    viz = _ScriptViz(VALID)
    agent = _agent(tmp_path, "h", main=lambda m: MARKER_REPLY, viz_chat=viz)

    agent.chat("show me")
    # the stashed job carries the ±context slice of the reply
    job = agent._pending_viz_jobs[0]
    assert "context" in job
    assert "Sales trend" in job["context"]
    assert "That's the picture" in job["context"]

    assert agent.wait_for_viz(timeout=5) is True
    gen_prompt = viz.prompts[0]
    assert "bar chart of sales" in gen_prompt  # description
    assert "Q1 12, Q2 19" in gen_prompt  # data hint
    assert "Sales trend" in gen_prompt  # context slice
    assert "SAME language" in gen_prompt  # the language rule marker


# --------------------------------------------------------------- kill switch


def test_disabled_no_pool_no_events(tmp_path):
    marker = "<<sherlock-viz: a chart | A 1, B 2>>"
    events: list[dict] = []
    agent = _agent(
        tmp_path, "off", main=lambda m: f"here: {marker}", viz_chat=None, visualization=None
    )
    agent.set_event_sink(events.append)

    reply = agent.chat("show me")
    assert marker in reply  # verbatim, no placeholder
    assert "⟦viz:" not in reply
    assert agent._pending_viz_jobs == []
    assert agent._viz_executor is None  # no pool ever created
    for t in ("viz.pending", "viz.rendered", "viz.failed", "viz.repairing"):
        assert _events_of(events, t) == []


# ----------------------------------------------- submitted-once across turns


def test_job_dispatched_once_across_turns(tmp_path):
    viz = _ScriptViz(VALID, VALID, VALID)
    events: list[dict] = []
    agent = _agent(tmp_path, "once", main=lambda m: MARKER_REPLY, viz_chat=viz)
    agent.set_event_sink(events.append)

    agent.chat("turn one")
    agent.chat("turn two")  # a fresh job (t2-1); t1-1 must NOT re-dispatch
    assert agent.wait_for_viz(timeout=5) is True

    rendered = _events_of(events, "viz.rendered")
    ids = sorted(e["data"]["viz_id"] for e in rendered)
    assert ids == ["t1-1", "t2-1"]  # each turn's job rendered exactly once
