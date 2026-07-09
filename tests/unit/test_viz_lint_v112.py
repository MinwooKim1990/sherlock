"""v1.12 Stage B2 — LLM-4 VISUALIZER static lint (sherlock/viz.py).

Pure red/green table for ``_viz_static_lint`` (the sandbox-contract enforcer)
plus the fence-stripping + validated-meta helpers and the prompt builders. The
async render pipeline (generation → lint → repair → artifact/event) lives in the
integration suite (test_viz_pipeline_v112.py).
"""

from __future__ import annotations

from sherlock.config import VisualizationConfig
from sherlock.viz import (
    LANGUAGE_RULE,
    VALIDATED_META,
    _viz_static_lint,
    build_generation_user,
    build_repair_user,
    build_self_review_user,
    inject_validated_meta,
    strip_code_fences,
)

CFG = VisualizationConfig(enabled=True)

# A minimal, valid artifact skeleton. Renders "Q1 12 / Q2 19" — both numbers
# trace to SRC, so data fidelity holds.
VALID = """<!DOCTYPE html><html><head>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; \
script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<style>body{font-family:sans-serif}</style></head>
<body><div id="c"><span>Q1 12</span><span>Q2 19</span></div>
<script>window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');
parent.postMessage({sherlockViz:'ready'}, '*');</script></body></html>"""

SRC = "bar chart of Q1 sales | Q1 12, Q2 19"


def _lint(html, src=SRC, cfg=CFG):
    return _viz_static_lint(html, src, cfg)


# ------------------------------------------------------------------ green


def test_valid_skeleton_passes():
    ok, errors = _lint(VALID)
    assert ok is True
    assert errors == []


def test_ready_signal_passes():
    # the {sherlockViz:'ready'} postMessage is the accepted ready signal
    ok, _ = _lint(VALID)
    assert ok is True
    assert "parent.postMessage({sherlockViz:'ready'}" in VALID


def test_number_present_in_data_hint_passes():
    html = VALID.replace("Q2 19", "Q2 3.5")
    ok, errors = _lint(html, src="chart | Q1 12, growth 3.5")
    assert ok is True, errors


def test_year_2025_exempt():
    html = VALID.replace("Q2 19", "Q2 19 in 2025")  # 2025 is a year label
    ok, errors = _lint(html)
    assert ok is True, errors


def test_cjk_labels_fine():
    html = VALID.replace("Q1 12", "일분기 12").replace("Q2 19", "이분기 19")
    ok, errors = _lint(html, src="분기 차트 | 일분기 12, 이분기 19")
    assert ok is True, errors


def test_thousands_comma_number_normalizes():
    html = VALID.replace("Q2 19", "Q2 1,234")
    ok, errors = _lint(html, src="chart | Q1 12, Q2 1234")  # source has no comma
    assert ok is True, errors


def test_inline_svg_xmlns_not_flagged_as_external():
    html = VALID.replace(
        '<div id="c">',
        '<div id="c"><svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
    )
    ok, errors = _lint(html)
    assert ok is True, errors


def test_data_uri_image_allowed():
    html = VALID.replace(
        '<div id="c">',
        '<div id="c"><img src="data:image/png;base64,AAAA">',
    )
    ok, errors = _lint(html)
    assert ok is True, errors


# -------------------------------------------------------------------- red


def test_empty_document_fails():
    ok, errors = _lint("   ")
    assert ok is False
    assert any("empty" in e for e in errors)


def test_missing_csp_meta_fails():
    html = VALID.replace(
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">",
        "",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("Content-Security-Policy" in e for e in errors)


def test_missing_viz_ready_fails():
    html = VALID.replace("parent.postMessage({sherlockViz:'ready'}, '*');", "")
    ok, errors = _lint(html)
    assert ok is False
    assert any("ready signal" in e for e in errors)


def test_external_script_src_fails():
    html = VALID.replace(
        "<style>body{font-family:sans-serif}</style>",
        '<script src="https://cdn.example.com/x.js"></script>',
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("external resource" in e for e in errors)


def test_protocol_relative_ref_fails():
    html = VALID.replace(
        "<style>body{font-family:sans-serif}</style>",
        '<link href="//cdn.example.com/x.css">',
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("external resource" in e for e in errors)


def test_fetch_call_fails():
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "fetch('/data.json');parent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("fetch(" in e for e in errors)


def test_window_parent_non_ready_fails():
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "var z = window.parent.location;parent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("window.parent" in e for e in errors)


def test_disallowed_postmessage_fails():
    # a parent.postMessage that is NOT a sherlockViz ready/error signal
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "parent.postMessage('exfil', '*');parent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("postMessage" in e for e in errors)


def test_localStorage_fails():
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "localStorage.setItem('a','b');parent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("localStorage" in e for e in errors)


def test_iframe_fails():
    html = VALID.replace('<div id="c">', '<iframe src="data:text/html,x"></iframe><div id="c">')
    ok, errors = _lint(html)
    assert ok is False
    assert any("forbidden element" in e for e in errors)


def test_javascript_url_fails():
    html = VALID.replace('<div id="c">', '<a href="javascript:alert(1)">x</a><div id="c">')
    ok, errors = _lint(html)
    assert ok is False
    assert any("javascript" in e for e in errors)


def test_import_statement_fails():
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "\nimport x from 'y';\nparent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("import" in e for e in errors)


def test_missing_error_handler_fails():
    # v1.12 B4: the runtime harness contract requires the window.onerror →
    # {sherlockViz:'error'} handler; a doc with only the ready signal fails.
    html = VALID.replace(
        "window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');\n",
        "",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("error handler" in e for e in errors)
    assert not any("ready signal" in e for e in errors)  # ready is still present


def test_unbalanced_div_fails():
    html = VALID.replace("</div>", "")  # drop the div close
    ok, errors = _lint(html)
    assert ok is False
    assert any("nested" in e or "unclosed" in e or "balanc" in e for e in errors)


def test_oversize_fails():
    small = VisualizationConfig(enabled=True, max_html_bytes=50)
    ok, errors = _lint(VALID, cfg=small)
    assert ok is False
    assert any("too large" in e for e in errors)


def test_invented_number_fails():
    html = VALID.replace("Q2 19", "Q2 42.7")  # 42.7 not in SRC
    ok, errors = _lint(html)
    assert ok is False
    assert any("invented number: 42.7" in e for e in errors)


def test_multiple_errors_all_collected():
    # missing ready signal AND an invented number → both surface (no short-circuit)
    html = VALID.replace("parent.postMessage({sherlockViz:'ready'}, '*');", "").replace(
        "Q2 19", "Q2 88.8"
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("ready signal" in e for e in errors)
    assert any("invented number: 88.8" in e for e in errors)


# ------------------------------------------------------------- helpers


def test_strip_code_fences_html_lang():
    assert strip_code_fences("```html\n<div>x</div>\n```") == "<div>x</div>"


def test_strip_code_fences_bare():
    assert strip_code_fences("```\n<b>y</b>\n```") == "<b>y</b>"


def test_strip_code_fences_untouched():
    assert strip_code_fences("<div>z</div>") == "<div>z</div>"


def test_inject_validated_meta_before_head():
    out = inject_validated_meta(VALID)
    assert VALIDATED_META in out
    assert out.index(VALIDATED_META) < out.lower().index("</head>")


def test_inject_validated_meta_idempotent():
    once = inject_validated_meta(VALID)
    twice = inject_validated_meta(once)
    assert twice.count(VALIDATED_META) == 1


def test_inject_validated_meta_no_head_falls_back_to_csp():
    # a document without a </head> still gets the meta after the CSP meta
    body_only = (
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">"
        "<div>x</div>"
    )
    out = inject_validated_meta(body_only)
    assert VALIDATED_META in out


# --------------------------------------------------------------- prompts


def test_generation_prompt_contains_job_and_language_rule():
    job = {
        "description": "bar chart of quarterly sales",
        "data_hint": "Q1 12, Q2 19",
        "context": "Sales rose across the year. Q1 12, Q2 19, ...",
    }
    user = build_generation_user(job)
    assert "bar chart of quarterly sales" in user
    assert "Q1 12, Q2 19" in user
    assert "Sales rose across the year" in user  # the surrounding context slice
    # the language rule marker rides the system prompt
    from sherlock.viz import VIZ_GENERATION_SYSTEM

    assert LANGUAGE_RULE in VIZ_GENERATION_SYSTEM
    assert "SAME language" in VIZ_GENERATION_SYSTEM


def test_repair_prompt_feeds_back_errors():
    user = build_repair_user("<div>x</div>", ["missing ready signal", "invented number: 5.5"])
    assert "missing ready signal" in user
    assert "invented number: 5.5" in user
    assert "<div>x</div>" in user


def test_self_review_prompt_carries_html_and_checklist():
    user = build_self_review_user("<div>y</div>")
    assert "<div>y</div>" in user
    assert "checklist" in user.lower()


# ------------------------------------------------ v1.12 B4 ready/error protocol


def test_ready_error_protocol_self_consistent():
    """The B4 runtime protocol must be threaded consistently: the generation prompt
    instructs BOTH the {sherlockViz:'ready'} post and the window.onerror →
    {sherlockViz:'error'} handler, the same two literals the lint requires — a doc
    that emits both passes, and dropping either one fails with a matching message."""
    from sherlock.viz import ERROR_HANDLER, READY_SIGNAL, VIZ_GENERATION_SYSTEM, _REVIEW_CHECKLIST

    # the canonical literals ride the generation prompt + the self-review checklist
    assert "sherlockViz:'ready'" in READY_SIGNAL
    assert "sherlockViz:'error'" in ERROR_HANDLER and "window.onerror" in ERROR_HANDLER
    assert "sherlockViz:'ready'" in VIZ_GENERATION_SYSTEM
    assert "sherlockViz:'error'" in VIZ_GENERATION_SYSTEM
    assert "window.onerror" in VIZ_GENERATION_SYSTEM
    assert "sherlockViz:'ready'" in _REVIEW_CHECKLIST

    # a doc carrying BOTH signals passes; dropping either one fails specifically
    assert _lint(VALID)[0] is True
    no_error = VALID.replace(
        "window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');\n", ""
    )
    ok, errs = _lint(no_error)
    assert ok is False and any("error handler" in e for e in errs)
    no_ready = VALID.replace("parent.postMessage({sherlockViz:'ready'}, '*');", "")
    ok, errs = _lint(no_ready)
    assert ok is False and any("ready signal" in e for e in errs)


# ------------------------------------------------ v1.12 audit regressions (F1)


def test_f1_adjacent_bare_number_text_nodes_not_merged():
    # F1: adjacent bare-number <text> nodes (12, 19) must be joined with a SPACE,
    # not "", so they don't merge into a fake "1219" that fails as invented.
    html = VALID.replace(
        '<div id="c"><span>Q1 12</span><span>Q2 19</span></div>',
        '<div id="c"><svg><text>12</text><text>19</text></svg></div>',
    )
    ok, errors = _lint(html)  # SRC contains both 12 and 19
    assert ok is True, errors
    assert not any("invented number" in e for e in errors)


def test_f1_genuinely_invented_number_still_fails():
    # F1 must NOT weaken invented-number detection: a bare number absent from the
    # source still fails.
    html = VALID.replace(
        '<div id="c"><span>Q1 12</span><span>Q2 19</span></div>',
        '<div id="c"><svg><text>12</text><text>777</text></svg></div>',
    )
    ok, errors = _lint(html)  # 777 is not in SRC
    assert ok is False
    assert any("invented number: 777" in e for e in errors)


# ------------------------------------------------ v1.12 audit regressions (F2)


def test_f2_meta_refresh_navigation_fails():
    html = VALID.replace(
        "<style>body{font-family:sans-serif}</style>",
        '<meta http-equiv="refresh" content="5">',
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("meta refresh" in e for e in errors)


def test_f2_location_href_navigation_fails():
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "location.href='#x';parent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("navigation via location" in e for e in errors)


def test_f2_window_open_navigation_fails():
    html = VALID.replace(
        "parent.postMessage({sherlockViz:'ready'}, '*');",
        "window.open('#x');parent.postMessage({sherlockViz:'ready'}, '*');",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("window.open" in e for e in errors)


# ------------------------------------------------ v1.12 audit regressions (F3)


def test_f3_external_srcset_fails():
    html = VALID.replace(
        '<div id="c">',
        '<div id="c"><img srcset="http://evil.example/x.png 1x">',
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("srcset" in e for e in errors)


def test_f3_external_form_action_fails():
    html = VALID.replace(
        '<div id="c">',
        '<div id="c"><form action="http://evil.example/collect"></form>',
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("form action" in e for e in errors)


def test_f3_external_css_url_fails():
    html = VALID.replace(
        "<style>body{font-family:sans-serif}</style>",
        "<style>body{background:url(http://evil.example/bg.png)}</style>",
    )
    ok, errors = _lint(html)
    assert ok is False
    assert any("url()" in e for e in errors)
