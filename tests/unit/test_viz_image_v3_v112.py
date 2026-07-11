"""v1.12 Stage V3 — text→image modality: marker kind, deterministic wrapper,
adapter normalisation, guidance gating.

The pipeline e2e (fake image callable through chat() → viz.rendered) lives in
tests/integration/test_viz_image_pipeline_v112.py.
"""

from __future__ import annotations

import base64

import pytest

from sherlock import Sherlock
from sherlock.agent import _parse_viz_tags, _viz_marker_guidance
from sherlock.config import VisualizationConfig
from sherlock.viz import (
    _viz_static_lint,
    build_image_artifact,
    sanitize_image_allowlist,
)

CFG = VisualizationConfig()
DATA_URI = "data:image/png;base64," + base64.b64encode(b"fakepng").decode()
URL = "https://img.example.com/gen/abc123.png"


# ------------------------------------------------------------------ marker kind


def test_image_prefix_sets_kind_and_strips():
    _, jobs = _parse_viz_tags(
        "<<sherlock-viz: image: a lighthouse in a storm, watercolor>>", cap=3, id_prefix="t1"
    )
    assert jobs[0]["kind"] == "image"
    assert jobs[0]["description"] == "a lighthouse in a storm, watercolor"


def test_image_prefix_case_insensitive():
    _, jobs = _parse_viz_tags("<<sherlock-viz: IMAGE: a cat mascot>>", cap=3, id_prefix="t2")
    assert jobs[0]["kind"] == "image"
    assert jobs[0]["description"] == "a cat mascot"


def test_no_prefix_kind_html():
    _, jobs = _parse_viz_tags("<<sherlock-viz: bar chart | A 1>>", cap=3, id_prefix="t3")
    assert jobs[0]["kind"] == "html"


def test_imagey_word_is_not_a_prefix():
    _, jobs = _parse_viz_tags("<<sherlock-viz: imagery of sales growth>>", cap=3, id_prefix="t4")
    assert jobs[0]["kind"] == "html"
    assert jobs[0]["description"] == "imagery of sales growth"


# ----------------------------------------------------------- wrapper + lint


def test_data_uri_artifact_passes_lint():
    html = build_image_artifact("a cat mascot", DATA_URI, ())
    ok, errors = _viz_static_lint(html, "a cat mascot", CFG, ())
    assert ok, errors
    assert DATA_URI in html
    assert "sherlockViz:'ready'" in html
    assert "prefers-color-scheme: dark" in html  # V2 frame card


def test_allowlisted_url_artifact_passes_lint_and_pins_csp():
    allow = sanitize_image_allowlist((URL,))
    html = build_image_artifact("a cat mascot", URL, allow)
    ok, errors = _viz_static_lint(html, "a cat mascot", CFG, allow)
    assert ok, errors
    assert f"img-src data: {URL}" in html  # CSP pin


def test_url_artifact_fails_lint_without_allowlist():
    html = build_image_artifact("a cat mascot", URL, ())
    ok, errors = _viz_static_lint(html, "a cat mascot", CFG, ())
    assert not ok
    assert any("external" in e for e in errors)


def test_title_is_escaped():
    html = build_image_artifact('x "<script>" y', DATA_URI, ())
    assert '<script>"' not in html  # escaped, not raw
    assert "&lt;script&gt;" in html


def test_oversize_data_uri_fails_image_cap():
    big = "data:image/png;base64," + ("A" * (CFG.max_image_html_bytes + 100))
    html = build_image_artifact("big", big, ())
    ok, errors = _viz_static_lint(html, "big", CFG, ())
    assert not ok
    assert any("too large" in e for e in errors)


# ----------------------------------------------------------- guidance gating


def test_guidance_without_image_is_byte_identical():
    from sherlock.agent import _VIZ_MARKER_GUIDANCE_TEMPLATE

    assert "image:" not in _viz_marker_guidance(3)
    # pin against the TEMPLATE (not the function against itself) so the
    # unconfigured guidance is provably the pre-V3 block, byte for byte.
    assert _viz_marker_guidance(3, image=False) == _VIZ_MARKER_GUIDANCE_TEMPLATE.format(N=3)


def test_guidance_with_image_teaches_prefix():
    g = _viz_marker_guidance(3, image=True)
    assert "image:" in g
    assert "NOT a data" in g


# ----------------------------------------------------------- adapter normalise


def _agent(tmp_path, name, *, image_gen=None, image_model=None):
    viz = {"enabled": True}
    if image_model:
        viz["image_model"] = image_model
    return Sherlock.with_callable(
        main_chat=lambda m: "ok",
        summary_chat=lambda m: "{}",
        inference_chat=lambda m: "{}",
        viz_image_gen=image_gen,
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        background=False,
        companions_mode="off",
        visualization=viz,
    )


def test_available_flags(tmp_path):
    assert _agent(tmp_path, "n")._viz_image_available() is False
    assert _agent(tmp_path, "c", image_gen=lambda p: {"b64": "x"})._viz_image_available() is True
    assert _agent(tmp_path, "m", image_model="dall-e-3")._viz_image_available() is True


@pytest.mark.parametrize(
    "ret,expect",
    [
        ({"b64": "QUJD"}, ("data:image/png;base64,QUJD", None)),
        ({"b64_json": "QUJD"}, ("data:image/png;base64,QUJD", None)),
        ({"url": URL}, (None, URL)),
        (URL, (None, URL)),
        ("QUJD", ("data:image/png;base64,QUJD", None)),
        (DATA_URI, (DATA_URI, None)),
    ],
)
def test_generate_normalises_shapes(tmp_path, ret, expect):
    agent = _agent(tmp_path, "norm", image_gen=lambda p: ret)
    assert agent._viz_image_generate("x") == expect


def test_generate_raises_on_empty(tmp_path):
    agent = _agent(tmp_path, "empty", image_gen=lambda p: {})
    with pytest.raises(ValueError):
        agent._viz_image_generate("x")
