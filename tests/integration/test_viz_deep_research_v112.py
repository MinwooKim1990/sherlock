"""v1.12 Stage B3 — LLM-4 VISUALIZER in the deep-research report.

Two seams, both GATED on ``config.visualization.enabled``:

1. The shared synthesis/editor PRESENTATION guide gains an inline-marker
   one-liner ONLY when enabled (off → byte-identical DR prompts).
2. A post-final hook (``_apply_deep_research_viz``) runs AFTER the whole verify
   chain and right BEFORE the ``deep_research.done`` emit — in BOTH the inline
   path (``_execute_deep_research``) and the background runner
   (``_run_deep_research_bg``) — turning report markers into ⟦viz:…⟧
   placeholders + async LLM-4 renders keyed with ``research_id``.

The full DR loop is out of scope here (covered elsewhere); ``_run_deep_research``
is stubbed to a canned report so the hook + guide are tested in isolation.
"""

from __future__ import annotations

import threading

from sherlock import Sherlock

# A valid artifact whose numbers (12, 19) trace to the marker data below, so the
# data-fidelity lint passes on the first try.
VALID = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>Q1 12</span><span>Q2 19</span></div>\n"
    "<script>parent.postMessage('viz-ready', '*');</script></body></html>"
)


class _ScriptViz:
    """Thread-safe scripted fake viz LLM; records each prompt it saw."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, messages):
        prompt = "\n".join((m["content"] if isinstance(m, dict) else m.content) for m in messages)
        with self._lock:
            self.prompts.append(prompt)
            return self._responses.pop(0) if self._responses else "NO MORE"


class _RecMain:
    """Recording main provider callable — returns a fixed reply, keeps prompts."""

    def __init__(self, reply: str = "edited report."):
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, messages):
        self.prompts.append(
            "\n".join((m["content"] if isinstance(m, dict) else m.content) for m in messages)
        )
        return self.reply


def _agent(tmp_path, name, *, main, viz_chat=None, viz=None):
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
        visualization=viz,
    )


def _events_of(events, type_):
    return [e for e in events if e["type"] == type_]


def _done_answer(events):
    dones = _events_of(events, "deep_research.done")
    assert len(dones) == 1
    return dones[0]["data"]["answer"]


# --- fixtures: canned reports the stubbed _run_deep_research returns ---------- #

REPORT_2 = (
    "## Sales\nQ1 was 12 and Q2 was 19.\n"
    "<<sherlock-viz: bar chart of sales | Q1 12, Q2 19>>\n"
    "More prose about the quarters.\n"
    "<<sherlock-viz: second chart | Q1 12, Q2 19>>\n"
    "## Sources\n- http://example.com"
)

# One literal marker — used to prove the OFF path leaves it verbatim.
REPORT_1_LITERAL = "## Findings\nA point.\n<<sherlock-viz: a chart | Q1 12>>\nDone."


# ============================================================ (1) guide gating


def test_synthesis_guide_gated_on_enabled(tmp_path):
    """The editor (verify) prompt carries the inline-marker instruction ONLY when
    visualization is enabled; the raw presentation guide is otherwise unchanged."""
    for enabled in (False, True):
        rec = _RecMain("edited.")
        agent = _agent(
            tmp_path,
            f"g{int(enabled)}",
            main=rec,
            viz=({"enabled": True} if enabled else None),
        )
        # direct: the shared guide toggles the viz one-liner
        guide = agent._presentation_guide()
        assert ("<<sherlock-viz:" in guide) is enabled
        # and it actually reaches a real DR prompt (the editor pass)
        state = {"confirmed_facts": [{"content": "Q1 was 12", "sources": ["http://a"]}]}
        agent._verify_research_report("Q1 was 12.", state, "sales")
        prompt = rec.prompts[-1]
        assert ("<<sherlock-viz:" in prompt) is enabled


def test_presentation_guide_off_is_byte_identical(tmp_path):
    from sherlock.agent import _PRESENTATION_GUIDE

    agent = _agent(tmp_path, "bid", main=lambda m: "ok", viz=None)
    assert agent._presentation_guide() == _PRESENTATION_GUIDE


# ============================================================ (2) OFF: verbatim


def test_dr_off_marker_stays_verbatim(tmp_path):
    """Default (viz off): a report containing a literal marker reaches
    deep_research.done byte-identical — no placeholder, no viz.pending."""
    agent = _agent(tmp_path, "off", main=lambda m: "ok", viz=None)
    events: list[dict] = []
    agent.set_event_sink(events.append)
    agent._run_deep_research = lambda *a, **k: REPORT_1_LITERAL

    out = agent._execute_deep_research("c1", "sales", 1, background=False)

    assert out == REPORT_1_LITERAL
    assert _done_answer(events) == REPORT_1_LITERAL
    assert "<<sherlock-viz: a chart | Q1 12>>" in _done_answer(events)
    assert _events_of(events, "viz.pending") == []
    assert _events_of(events, "viz.rendered") == []


# ============================================================ (3) ON: inline path


def test_dr_on_inline_placeholders_and_renders(tmp_path):
    viz = _ScriptViz(VALID, VALID)
    agent = _agent(
        tmp_path,
        "on",
        main=lambda m: "ok",
        viz_chat=viz,
        viz={"enabled": True, "self_review_rounds": 0, "max_repair_rounds": 0},
    )
    events: list[dict] = []
    agent.set_event_sink(events.append)
    agent._run_deep_research = lambda *a, **k: REPORT_2

    out = agent._execute_deep_research("c1", "sales", 3, background=False)

    # both markers → placeholders in the done answer; no raw marker survives
    answer = _done_answer(events)
    assert out == answer
    assert "<<sherlock-viz" not in answer
    assert answer.count("⟦viz:") == 2

    # research_id was minted as dr1 (first run); placeholders use that prefix,
    # which can never collide with chat's t{turn}-n ids.
    assert "⟦viz:dr1-1⟧" in answer
    assert "⟦viz:dr1-2⟧" in answer

    pending = _events_of(events, "viz.pending")
    assert len(pending) == 2
    for e in pending:
        assert e["data"]["research_id"] == "dr1"
        assert e["data"]["viz_id"].startswith("dr1-")
        assert not e["data"]["viz_id"].startswith("t")

    assert agent.wait_for_viz(timeout=5) is True
    rendered = _events_of(events, "viz.rendered")
    assert len(rendered) == 2
    for e in rendered:
        assert e["data"]["research_id"] == "dr1"
        assert e["data"]["turn"] == 3
        assert e["data"]["validated"] == "static"
    assert _events_of(events, "viz.failed") == []


# ============================================================ (4) ON: bg parity


def test_dr_on_bg_runner_parity(tmp_path):
    """The background runner (_run_deep_research_bg) placeholders the report and
    submits renders exactly like the inline path."""
    viz = _ScriptViz(VALID, VALID)
    agent = _agent(
        tmp_path,
        "bg",
        main=lambda m: "ok",
        viz_chat=viz,
        viz={"enabled": True, "self_review_rounds": 0, "max_repair_rounds": 0},
    )
    events: list[dict] = []
    agent.set_event_sink(events.append)
    agent._run_deep_research = lambda *a, **k: REPORT_2

    # a REAL conversation row so the persisted assertion below is meaningful
    # (add_message is FK-guarded; a bare "c1" would silently no-op).
    cid = agent._storage.create_conversation("test").id
    agent._run_deep_research_bg(cid, "sales", 5, "dr7")

    answer = _done_answer(events)
    assert "<<sherlock-viz" not in answer
    assert "⟦viz:dr7-1⟧" in answer
    assert "⟦viz:dr7-2⟧" in answer

    pending = _events_of(events, "viz.pending")
    assert len(pending) == 2
    assert all(e["data"]["research_id"] == "dr7" for e in pending)

    assert agent.wait_for_viz(timeout=5) is True
    rendered = _events_of(events, "viz.rendered")
    assert len(rendered) == 2
    assert all(e["data"]["research_id"] == "dr7" for e in rendered)

    # v1.12 F4(c): the PERSISTED message must equal the emitted done answer —
    # stored ↔ emitted consistency, both carrying the ⟦viz:…⟧ placeholders.
    msgs = agent._storage.list_messages(cid)
    assert msgs[-1].role == "assistant"
    assert msgs[-1].content == answer


# ============================================================ (5) cap enforced


def test_dr_report_marker_cap_enforced(tmp_path):
    """Markers beyond max_markers_report are stripped entirely (no placeholder,
    no job) — here cap=2 with 4 markers → 2 placeholders, 3rd/4th dropped."""
    report = (
        "Q1 12, Q2 19.\n"
        "<<sherlock-viz: c1 | Q1 12, Q2 19>>\n"
        "<<sherlock-viz: c2 | Q1 12, Q2 19>>\n"
        "<<sherlock-viz: c3 | Q1 12, Q2 19>>\n"
        "<<sherlock-viz: c4 | Q1 12, Q2 19>>\n"
    )
    viz = _ScriptViz(VALID, VALID)
    agent = _agent(
        tmp_path,
        "cap",
        main=lambda m: "ok",
        viz_chat=viz,
        viz={
            "enabled": True,
            "max_markers_report": 2,
            "self_review_rounds": 0,
            "max_repair_rounds": 0,
        },
    )
    events: list[dict] = []
    agent.set_event_sink(events.append)
    agent._run_deep_research = lambda *a, **k: report

    out = agent._execute_deep_research("c1", "sales", 1, background=False)

    assert out.count("⟦viz:") == 2
    assert "<<sherlock-viz" not in out  # the over-cap markers are stripped, not left raw
    assert len(_events_of(events, "viz.pending")) == 2
    assert agent.wait_for_viz(timeout=5) is True


# ============================================================ (6) kill switch


def test_dr_disabled_emits_nothing_even_with_markers(tmp_path):
    """enabled=False: even a report full of markers produces no viz activity and
    the answer is untouched (the existing DR contract holds)."""
    agent = _agent(tmp_path, "kill", main=lambda m: "ok", viz=None)
    events: list[dict] = []
    agent.set_event_sink(events.append)
    agent._run_deep_research = lambda *a, **k: REPORT_2

    out = agent._execute_deep_research("c1", "sales", 1, background=False)

    assert out == REPORT_2
    assert _events_of(events, "viz.pending") == []
    assert _events_of(events, "viz.rendered") == []
    assert agent._pending_viz_jobs == []


# ============================================================ (7) F1 best-effort


def test_dr_viz_hook_failure_still_persists_and_emits_done(tmp_path):
    """v1.12 F1: the post-final viz hook is the ONLY step between the DR answer and
    persist + deep_research.done. If it raises (here _submit_viz_jobs blows up, the
    way a pool.submit RuntimeError does at interpreter shutdown), the run must STILL
    persist the report and emit deep_research.done with the answer returned
    unchanged — never the 'stuck at Starting deep research…' + lost-report class."""
    viz = _ScriptViz(VALID, VALID)
    agent = _agent(
        tmp_path,
        "f1",
        main=lambda m: "ok",
        viz_chat=viz,
        viz={"enabled": True, "self_review_rounds": 0, "max_repair_rounds": 0},
    )
    events: list[dict] = []
    agent.set_event_sink(events.append)
    agent._run_deep_research = lambda *a, **k: REPORT_2

    def _boom():
        raise RuntimeError("cannot schedule new futures after shutdown")

    agent._submit_viz_jobs = _boom  # simulate the swallowed raise in the hook

    cid = agent._storage.create_conversation("test").id
    out = agent._execute_deep_research(cid, "sales", 3, background=False)

    # answer returned UNCHANGED (original report, markers intact) on hook failure
    assert out == REPORT_2
    # deep_research.done still fired, exactly once, with that same answer
    assert _done_answer(events) == REPORT_2
    # and the report was actually persisted (not lost)
    msgs = agent._storage.list_messages(cid)
    assert msgs[-1].role == "assistant"
    assert msgs[-1].content == REPORT_2
