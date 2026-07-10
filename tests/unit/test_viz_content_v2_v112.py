"""v1.12 Stage V2 — LLM-4 VISUALIZER content quality: context enrichment,
content-type palette, format awareness, frame.

Pure-function coverage of ``detect_context_flags`` (markdown/HTML table sniff),
the enriched ``build_generation_user`` (reader's question + don't-duplicate-the-
table note, both OPTIONAL keys so pre-V2 stashed jobs build the pre-V2 prompt),
the FORM palette + FRAME contract in the generation system prompt, and the
question/context_flags plumbing through ``_parse_viz_tags`` /
``_extract_viz_jobs`` / ``_apply_deep_research_viz``. The e2e proof that a
number appearing ONLY in the question survives the fidelity lint lives in the
integration suite (test_viz_pipeline_v112.py).
"""

from __future__ import annotations

from sherlock import Sherlock
from sherlock.agent import (
    _DR_VIZ_GUIDANCE_TEMPLATE,
    _parse_viz_tags,
    _viz_marker_guidance,
)
from sherlock.viz import (
    LANGUAGE_RULE,
    VIZ_GENERATION_SYSTEM,
    build_csp_meta,
    build_generation_system,
    build_generation_user,
    detect_context_flags,
)

# ------------------------------------------------------------- format sniffing


def test_detect_gfm_table_with_pipes():
    ctx = "Results:\n| Model | Score |\n|---|---|\n| A | 12 |\n| B | 19 |\n"
    assert detect_context_flags(ctx) == ("table",)


def test_detect_gfm_table_without_edge_pipes():
    ctx = "Model | Score\n--- | ---\nA | 12\n"
    assert detect_context_flags(ctx) == ("table",)


def test_detect_gfm_table_with_alignment_colons():
    ctx = "| a | b | c |\n|:---|:---:|---:|\n| 1 | 2 | 3 |\n"
    assert detect_context_flags(ctx) == ("table",)


def test_detect_html_table_tag():
    assert detect_context_flags("some text <TABLE><tr><td>x</td></tr></table>") == ("table",)


def test_fenced_example_table_not_flagged():
    # A table the reply merely QUOTES inside a code fence is not a table the
    # reply shows — flagging it would make the injected note lie to LLM-4.
    ctx = "GFM syntax example:\n```markdown\n| a | b |\n|---|---|\n| 1 | 2 |\n```\nprose"
    assert detect_context_flags(ctx) == ()


def test_unclosed_fence_swallows_to_end():
    # The ±slice can cut the closing fence off — everything after the opener is
    # still code, not shown content.
    ctx = "example:\n```\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert detect_context_flags(ctx) == ()


def test_backticked_table_mention_not_flagged():
    assert detect_context_flags("use the `<table>` element for layout") == ()
    assert detect_context_flags("use `| a | b |` and `|---|` rows") == ()


def test_prose_tablex_not_flagged():
    # substring "<table" without a tag terminator is not a table tag
    assert detect_context_flags("see <tablex> docs") == ()


def test_single_column_table_flagged():
    assert detect_context_flags("| score |\n|---|\n| 12 |") == ("table",)


def test_separator_at_slice_start_over_data_row_flagged():
    # the slice cut the header off: separator first, |-bearing data row below
    assert detect_context_flags("|---|---|\n| A | 12 |\n| B | 19 |") == ("table",)


def test_whitespace_flood_line_is_fast():
    import time

    ctx = "| a | b |\n" + (" " * 5000) + "-|\nrest"
    start = time.monotonic()
    detect_context_flags(ctx)
    assert time.monotonic() - start < 0.2  # linear scan — no regex backtracking


def test_horizontal_rule_is_not_a_table():
    # ``---`` under a pipe-free line is an <hr>, and even under a line WITH a
    # pipe the separator itself carries no pipe → not a table.
    assert detect_context_flags("intro\n---\nbody") == ()
    assert detect_context_flags("uses a | b syntax\n---\nbody") == ()


def test_prose_pipes_without_separator_not_a_table():
    assert detect_context_flags("either A | B | C — pick one\nno separator here") == ()


def test_separator_line_with_prose_not_a_table():
    # dashes+pipe embedded in prose must not match the whole-line separator.
    assert detect_context_flags("a | b\nfoo -- | bar --\n") == ()


def test_empty_context_no_flags():
    assert detect_context_flags("") == ()
    assert detect_context_flags(None) == ()  # type: ignore[arg-type]


# ------------------------------------------------------- generation user prompt


def _job(**extra) -> dict:
    base = {
        "description": "bar chart of quarterly sales",
        "data_hint": "Q1 12, Q2 19",
        "context": "Sales rose across the year. Q1 12, Q2 19.",
    }
    base.update(extra)
    return base


def test_question_rides_the_prompt_when_present():
    user = build_generation_user(_job(question="how did sales do in 2024?"))
    assert "THE READER'S QUESTION" in user
    assert "how did sales do in 2024?" in user
    # audit hardening: labelled untrusted + emphasis-only, never a data source
    assert "untrusted" in user
    assert "NOT a data source" in user
    assert "NEVER chart a number that appears only in the question" in user
    # existing fields intact
    assert "bar chart of quarterly sales" in user
    assert "Q1 12, Q2 19" in user


def test_no_question_no_marker_line():
    # A pre-V2 stashed job (no ``question``/``context_flags`` keys) must build
    # the exact pre-V2 prompt shape — no new sections appear.
    user = build_generation_user(_job())
    assert "READER'S QUESTION" not in user
    assert "ALREADY shows a table" not in user


def test_blank_question_treated_as_absent():
    user = build_generation_user(_job(question="   "))
    assert "READER'S QUESTION" not in user


def test_table_flag_adds_no_duplicate_table_note():
    user = build_generation_user(_job(context_flags=("table",)))
    assert "ALREADY shows a table" in user
    assert "Do NOT render another table" in user


def test_unknown_flags_add_nothing():
    user = build_generation_user(_job(context_flags=("something-else",)))
    assert "ALREADY shows a table" not in user


# ------------------------------------------------- generation system: FORM+FRAME


def test_system_prompt_has_form_palette_and_table_ban():
    s = VIZ_GENERATION_SYSTEM
    assert "FORM — pick the graphical form" in s
    assert "line or area chart" in s
    assert "step/flow diagram" in s
    assert "NOT a visualization — never output one" in s


def test_system_prompt_has_frame_contract():
    s = VIZ_GENERATION_SYSTEM
    assert "FRAME — every artifact is ONE self-contained card" in s
    assert "prefers-color-scheme: dark" in s
    assert "system-ui" in s


def test_system_prompt_keeps_v1_contracts():
    # V2 adds quality sections but must not disturb the sandbox/CSP/language
    # contracts V1 and B2 pinned.
    s = VIZ_GENERATION_SYSTEM
    assert build_csp_meta(()) in s
    assert LANGUAGE_RULE in s
    assert "sherlockViz:'ready'" in s
    assert "sherlockViz:'error'" in s
    assert build_generation_system(()) == s
    url = "https://img.example.com/chart.png"
    assert url in build_generation_system((url,))


def test_system_prompt_carries_no_lintable_numbers():
    # Audit invariant: the (empty-allowlist) system prompt must contain ZERO
    # significant-number tokens. Any pixel size / hex color worded into the
    # prompt is a number the model was shown but the fidelity lint would
    # reject as "invented" if echoed into visible text — a class of
    # unrepairable failures that must never exist.
    from sherlock.viz import _significant_numbers

    assert _significant_numbers(VIZ_GENERATION_SYSTEM) == []
    # and the prompt explicitly forbids echoing styling values into text
    assert "never echo a pixel size or color code into visible text" in VIZ_GENERATION_SYSTEM


# --------------------------------------------------------------- marker parsing


def test_parse_viz_tags_sets_context_flags_table():
    text = (
        "| Q | Rev |\n|---|---|\n| Q1 | 12 |\n| Q2 | 19 |\n\n"
        "<<sherlock-viz: bar chart of revenue | Q1 12, Q2 19>>\ndone"
    )
    _, jobs = _parse_viz_tags(text, cap=3, id_prefix="t1")
    assert jobs[0]["context_flags"] == ("table",)


def test_parse_viz_tags_no_table_empty_flags():
    _, jobs = _parse_viz_tags("prose only\n<<sherlock-viz: a chart | A 1>>", cap=3, id_prefix="t2")
    assert jobs[0]["context_flags"] == ()


# ---------------------------------------------------------------- agent plumbing


def _agent(tmp_path, name, *, main, viz_chat=None):
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
        visualization=True,
    )


def test_chat_threads_user_question_into_job(tmp_path):
    agent = _agent(tmp_path, "q", main=lambda m: "Here: <<sherlock-viz: a chart | A 12, B 19>>")
    agent.chat("compare A and B for me")
    assert agent._pending_viz_jobs
    assert agent._pending_viz_jobs[-1]["question"] == "compare A and B for me"


def test_question_trimmed_to_600(tmp_path):
    agent = _agent(tmp_path, "trim", main=lambda m: "Here: <<sherlock-viz: a chart | A 12>>")
    agent.chat("x" * 2000)
    assert len(agent._pending_viz_jobs[-1]["question"]) == 600


def test_deep_research_hook_threads_topic_as_question(tmp_path):
    agent = _agent(tmp_path, "dr", main=lambda m: "ok")
    out = agent._apply_deep_research_viz(
        "Report.\n<<sherlock-viz: trend line | 2020 12, 2021 19>>\nEnd.",
        "dr1",
        3,
        question="global widget demand",
    )
    assert "⟦viz:dr1-1⟧" in out
    job = agent._pending_viz_jobs[-1]
    assert job["question"] == "global widget demand"
    assert job["research_id"] == "dr1"


# ------------------------------------------------------------- marker guidance


def test_chat_guidance_redirects_tables_to_markdown():
    g = _viz_marker_guidance(3)
    assert "small table" not in g
    assert "markdown table" in g
    assert "GRAPHICAL" in g


def test_dr_guidance_redirects_tables_to_markdown():
    g = _DR_VIZ_GUIDANCE_TEMPLATE.format(N=2)
    assert "markdown table" in g
    assert "GRAPHICAL" in g
