"""v1.12 Stage B1 — LLM-4 VISUALIZER marker protocol (parser + kill switch).

The pure ``_parse_viz_tags`` behaviour (placeholder substitution, description|
data split, cap enforcement, CJK/multiline/malformed handling, exact placeholder
format) plus the enabled=False (default) KILL SWITCH end-to-end through chat():
a stray ``<<sherlock-viz: ...>>`` marker must survive VERBATIM, emit no
``viz.pending`` event, stash no jobs, and add no system-prompt guidance. The
with_callable plumbing (viz provider / _viz_llm fallback / config flip) is here
too; the enabled render-pipeline e2e + playground wiring live in the integration
suite (test_viz_plumbing_v112.py).
"""

from __future__ import annotations

import time

from sherlock import Sherlock
from sherlock.agent import _parse_viz_tags, _viz_marker_guidance

# ------------------------------------------------------------------ pure parser


def test_single_marker_placeholder_and_job():
    text = "Here is the trend:\n<<sherlock-viz: line chart of sales | Jan 1, Feb 2>>\nDone."
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="t1")
    assert len(jobs) == 1
    job = jobs[0]
    assert job["viz_id"] == "t1-1"
    assert job["anchor"] == "⟦viz:t1-1⟧"
    assert job["description"] == "line chart of sales"
    assert job["data_hint"] == "Jan 1, Feb 2"
    # marker replaced by its placeholder, no raw marker left
    assert "⟦viz:t1-1⟧" in new_text
    assert "<<sherlock-viz" not in new_text


def test_description_data_split_first_pipe_only():
    # A data hint that itself contains pipes: only the FIRST '|' splits.
    _, jobs = _parse_viz_tags(
        "<<sherlock-viz: table of scores | A|1, B|2, C|3>>", cap=3, id_prefix="t2"
    )
    assert jobs[0]["description"] == "table of scores"
    assert jobs[0]["data_hint"] == "A|1, B|2, C|3"


def test_no_pipe_empty_data_hint():
    _, jobs = _parse_viz_tags("<<sherlock-viz: a simple flow diagram>>", cap=3, id_prefix="t3")
    assert jobs[0]["description"] == "a simple flow diagram"
    assert jobs[0]["data_hint"] == ""


def test_cap_enforcement_strips_extra():
    text = (
        "<<sherlock-viz: one>>\n<<sherlock-viz: two>>\n"
        "<<sherlock-viz: three>>\n<<sherlock-viz: four>>"
    )
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="t4")
    # exactly cap jobs; the 4th marker stripped with NO placeholder
    assert len(jobs) == 3
    assert [j["viz_id"] for j in jobs] == ["t4-1", "t4-2", "t4-3"]
    assert new_text.count("⟦viz:") == 3
    assert "⟦viz:t4-4⟧" not in new_text
    assert "<<sherlock-viz" not in new_text  # the 4th is gone entirely, not left raw
    # description "four" never became a job
    assert all(j["description"] != "four" for j in jobs)


def test_cjk_description_and_data_survive():
    text = "매출 추이:\n<<sherlock-viz: 분기별 매출 막대그래프 | 1분기 12, 2분기 19>>"
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="t5")
    assert jobs[0]["description"] == "분기별 매출 막대그래프"
    assert jobs[0]["data_hint"] == "1분기 12, 2분기 19"
    assert "⟦viz:t5-1⟧" in new_text


def test_multiline_payload():
    text = "<<sherlock-viz: a diagram spanning\nmultiple lines | node A -> node B\nB -> C>>"
    _, jobs = _parse_viz_tags(text, cap=3, id_prefix="t6")
    assert len(jobs) == 1
    assert "multiple lines" in jobs[0]["description"]
    assert "B -> C" in jobs[0]["data_hint"]


def test_malformed_unclosed_left_untouched():
    text = "trailing off <<sherlock-viz: never closed with no end bracket"
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="t7")
    assert jobs == []
    assert new_text == text  # byte-identical — nothing matched


def test_placeholder_token_exact_format():
    # U+27E6 (⟦) LEFT WHITE SQUARE BRACKET / U+27E7 (⟧) RIGHT WHITE SQUARE BRACKET.
    _, jobs = _parse_viz_tags("<<sherlock-viz: x>>", cap=2, id_prefix="tK")
    assert jobs[0]["anchor"] == "⟦viz:tK-1⟧"


def test_no_markers_returns_original_and_empty_jobs():
    text = "Just a normal reply with no visuals at all."
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="t8")
    assert jobs == []
    assert new_text == text


def test_capped_removal_collapses_blank_lines():
    # cap=1: markers 2 and 3 (each on their own line) are stripped; the gap
    # left behind must not become a run of 3+ newlines.
    text = "intro\n\n<<sherlock-viz: keep>>\n\n<<sherlock-viz: drop2>>\n\n<<sherlock-viz: drop3>>\n\nend"
    new_text, jobs = _parse_viz_tags(text, cap=1, id_prefix="t9")
    assert len(jobs) == 1
    assert "\n\n\n" not in new_text


def test_guidance_includes_cap_number():
    g = _viz_marker_guidance(2)
    assert "sherlock-viz" in g
    assert "2 marker" in g  # the cap made it into the block


def test_unclosed_flood_parses_under_time_bound():
    # F1 regression: a flood of unclosed ``<<sherlock-viz:`` openers followed by a
    # long tail must NOT trigger quadratic backtracking. The ``{1,2000}`` bound
    # caps the scan per opener, so this parses near-instantly. Bound is a generous
    # 2s so a slow/loaded CI box can't flake (the fix runs it in ~0.2s; the old
    # ``.+?`` took tens of seconds on inputs this shape).
    text = "<<sherlock-viz:" * 2000 + ("x" * 100_000)
    start = time.monotonic()
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="tT")
    elapsed = time.monotonic() - start
    assert elapsed < 2.0
    # nothing is closed → nothing matched → left VERBATIM, no jobs
    assert jobs == []
    assert new_text == text


def test_close_bracket_in_payload_truncates_at_first():
    # A literal ``>>`` inside the payload closes the marker early — the parser is
    # not ``>>``-aware. CURRENT semantics: a placeholder is minted for the head
    # ("a chart") and the trailing " junk>>" is LEFT VERBATIM in the text. Pinned
    # so any future change to ``>>`` handling is a deliberate decision, not drift.
    new_text, jobs = _parse_viz_tags("<<sherlock-viz: a chart >> junk>>", cap=3, id_prefix="tC")
    assert len(jobs) == 1
    assert jobs[0]["description"] == "a chart"
    assert jobs[0]["anchor"] in new_text
    assert new_text.endswith(" junk>>")  # trailing garbage remains after truncation


def test_empty_payload_stripped_no_job_no_placeholder():
    # F4: ``<<sherlock-viz: >>`` (whitespace-only payload) is stripped entirely —
    # no job, no placeholder token, and no cap budget consumed. The blank-line gap
    # left where it sat on its own line is collapsed.
    new_text, jobs = _parse_viz_tags("before\n\n<<sherlock-viz: >>\n\nafter", cap=3, id_prefix="tE")
    assert jobs == []
    assert "⟦viz:" not in new_text
    assert "<<sherlock-viz" not in new_text
    assert "\n\n\n" not in new_text


def test_marker_inside_code_fence_still_replaced():
    # The parser is NOT markdown-aware: a marker inside a ``` code fence ``` is
    # still replaced by a placeholder. Accepted as cosmetics for B1 — pinned here
    # so the (rare) in-fence case stays a known, deliberate behavior.
    text = "```\n<<sherlock-viz: a chart | A 1>>\n```"
    new_text, jobs = _parse_viz_tags(text, cap=3, id_prefix="tF")
    assert len(jobs) == 1
    assert "⟦viz:tF-1⟧" in new_text
    assert "<<sherlock-viz" not in new_text


# ------------------------------------------------------------------ hermetic agent


def _agent(tmp_path, name, *, main, visualization=None, viz_chat=None):
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


class _CapturingMain:
    """A main callable that records the system prompt it received each call."""

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


# ------------------------------------------------------------------ kill switch


def test_kill_switch_marker_verbatim_no_events_no_guidance(tmp_path):
    marker = "<<sherlock-viz: a bar chart | A 1, B 2>>"
    main = _CapturingMain(f"Here you go: {marker}")
    events: list[dict] = []
    agent = _agent(tmp_path, "off", main=main, visualization=None)  # OFF (default)
    agent.set_event_sink(events.append)

    reply = agent.chat("show me a chart")

    # marker survives VERBATIM; no placeholder token was ever introduced
    assert marker in reply
    assert "⟦viz:" not in reply
    # no jobs stashed, no viz.pending emitted
    assert agent._pending_viz_jobs == []
    assert [e for e in events if e["type"] == "viz.pending"] == []
    # no visualizer guidance leaked into the system prompt
    assert main.system_prompts
    assert all("sherlock-viz" not in sp for sp in main.system_prompts)


def test_new_session_resets_pending_viz_stash(tmp_path):
    # F2: viz jobs are keyed off the session-local turn index (which resets to 0
    # on a new session), so a stale stash from the prior session would let the new
    # session's "t1-1" collide with an old "t1-1". new_session() must clear it.
    main = _CapturingMain("Here you go: <<sherlock-viz: a chart | A 1>>")
    agent = _agent(tmp_path, "reset", main=main, visualization=True)

    agent.chat("draw it")
    assert agent._pending_viz_jobs  # marker turn populated the stash

    agent.new_session()
    assert agent._pending_viz_jobs == []  # cleared on the session boundary


# ------------------------------------------------------------------ plumbing


def test_with_callable_viz_chat_sets_provider(tmp_path):
    from sherlock.providers.callable_provider import CallableProvider

    agent = _agent(tmp_path, "vp", main=lambda m: "ok", viz_chat=lambda m: "<svg/>")
    assert isinstance(agent._viz_provider, CallableProvider)
    # the resolver uses the dedicated provider, not the main one
    assert agent._viz_llm() is agent._viz_provider
    assert agent._viz_llm() is not agent._provider


def test_unset_viz_falls_back_to_main(tmp_path):
    agent = _agent(tmp_path, "fb", main=lambda m: "ok", viz_chat=None)
    assert agent._viz_provider is None
    assert agent._viz_llm() is agent._provider


def test_visualization_true_flips_config(tmp_path):
    off = _agent(tmp_path, "cfgoff", main=lambda m: "ok", visualization=None)
    assert off.config.visualization.enabled is False  # default byte-identical off

    on = _agent(tmp_path, "cfgon", main=lambda m: "ok", visualization=True)
    assert on.config.visualization.enabled is True
    # defaults preserved
    assert on.config.visualization.max_markers_chat == 3


def test_visualization_dict_overrides(tmp_path):
    agent = _agent(
        tmp_path,
        "cfgdict",
        main=lambda m: "ok",
        visualization={"max_markers_chat": 1, "max_html_bytes": 1000},
    )
    assert agent.config.visualization.enabled is True  # dict implies enabled
    assert agent.config.visualization.max_markers_chat == 1
    assert agent.config.visualization.max_html_bytes == 1000
