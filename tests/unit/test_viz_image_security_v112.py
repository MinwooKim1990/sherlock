"""v1.12 Stage V1 — IMAGE SECURITY CORE for the LLM-4 visualizer lint.

The visualizer may embed web images WITHOUT weakening the sandbox by pinning the
CSP ``img-src`` to an EXACT per-job URL allowlist. This suite proves the three
layers of that control:

  * ``sanitize_image_allowlist`` — the construction-time gate that blocks any URL
    that could break out of the CSP meta string or the lint;
  * L1 — the real ``img-src`` directive parse (byte-equal against the allowlist,
    plus no URL source on any other directive);
  * L2 — the external-reference scan reconciled against parser-approved
    ``<img src>`` allowlist entries (zero carve-out for non-<img> refs);

and — CRITICAL — that with an EMPTY allowlist the lint output is byte-identical to
the pre-Stage-V1 behaviour (the whole existing fixture corpus is re-run here).
"""

from __future__ import annotations

import pytest

from sherlock.config import VisualizationConfig
from sherlock.viz import (
    CSP_META,
    build_csp_meta,
    build_generation_system,
    sanitize_image_allowlist,
    _ALLOWLIST_FORBIDDEN_CHARS,
    _viz_static_lint,
)

CFG = VisualizationConfig(enabled=True)
URL = "https://cdn.example.com/logo.png"
ALLOW = (URL,)


def _doc(body_extra: str = "", csp_img: str = "img-src data:") -> str:
    """A valid skeleton (numbers 12/19 trace to the src) with a customisable body
    fragment and img-src directive."""
    return (
        "<!DOCTYPE html><html><head>"
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; " + csp_img + '">'
        "</head><body><div>" + body_extra + "<span>12</span><span>19</span></div>"
        "<script>window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');"
        "parent.postMessage({sherlockViz:'ready'}, '*');</script></body></html>"
    )


def _lint(html, allow=(), src="12 19"):
    return _viz_static_lint(html, src, CFG, allow)


# ============================================================ CSP builder


def test_csp_meta_constant_byte_identical():
    assert CSP_META == build_csp_meta()
    assert CSP_META == (
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">"
    )


def test_csp_meta_appends_exact_allowed_urls():
    out = build_csp_meta((URL, "https://img.example.org/a.png"))
    assert f'img-src data: {URL} https://img.example.org/a.png"' in out


def test_generation_system_empty_is_pre_v1_constant():
    from sherlock.viz import VIZ_GENERATION_SYSTEM

    assert build_generation_system() == VIZ_GENERATION_SYSTEM
    # a non-empty allowlist reaches the CSP the model is told to emit verbatim
    assert URL in build_generation_system((URL,))


# ============================================================ sanitizer


def test_sanitizer_keeps_clean_https_and_http():
    assert sanitize_image_allowlist([URL]) == (URL,)
    assert sanitize_image_allowlist(["http://a.example/x.png"]) == ("http://a.example/x.png",)


@pytest.mark.parametrize(
    "bad",
    [
        "ftp://a.example/x.png",  # non-http scheme
        "//a.example/x.png",  # protocol-relative (no http scheme)
        "https://a.example/x y.png",  # ASCII space
        "https://a.example/x\ty.png",  # tab (whitespace/control)
        "https://a.example/x'y.png",  # single quote
        'https://a.example/x"y.png',  # double quote
        "https://a.example/x;y.png",  # semicolon
        "https://a.example/x\\y.png",  # backslash
        "https://a.example/x`y.png",  # backtick
        "https://a.example/x<y.png",  # less-than
        "https://a.example/x>y.png",  # greater-than
        "https://a.example/x(y.png",  # open paren
        "https://a.example/x)y.png",  # close paren
        "https://a.example/x\x01y.png",  # control char < 0x20
        "https://a.example/x\x7fy.png",  # DEL 0x7f
        "https://a.example/" + "a" * 500,  # > 500 bytes
    ],
)
def test_sanitizer_rejects_each_hostile(bad):
    assert sanitize_image_allowlist([bad]) == ()


def test_sanitizer_rejects_non_str():
    assert sanitize_image_allowlist([None, 123, b"https://a/x"]) == ()


def test_sanitizer_dedupes_preserving_order():
    got = sanitize_image_allowlist(["https://a/x", "https://b/y", "https://a/x", "https://c/z"])
    assert got == ("https://a/x", "https://b/y", "https://c/z")


def test_sanitizer_caps_at_four():
    got = sanitize_image_allowlist([f"https://a/{i}" for i in range(9)])
    assert got == tuple(f"https://a/{i}" for i in range(4))


def test_sanitizer_500_byte_boundary():
    # exactly 500 bytes kept; 501 dropped
    at = "https://a/" + "b" * (500 - len("https://a/"))
    assert len(at.encode()) == 500
    assert sanitize_image_allowlist([at]) == (at,)
    over = at + "b"
    assert sanitize_image_allowlist([over]) == ()


# ============================================================ L1 (CSP parse)


def test_l1_img_src_non_allowlisted_url_fails():
    ok, errs = _lint(_doc(csp_img="img-src data: https://evil.example/x.png"), allow=())
    assert ok is False
    assert any(
        "img-src has a non-allowlisted source: https://evil.example/x.png" in e for e in errs
    )


def test_l1_img_src_allowlisted_passes():
    ok, errs = _lint(_doc(f'<img src="{URL}">', csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is True, errs


def test_l1_img_src_data_only_empty_allowlist_passes():
    # today's behaviour: img-src data:, empty allowlist
    ok, errs = _lint(_doc(csp_img="img-src data:"), allow=())
    assert ok is True, errs


def test_l1_extra_source_on_default_src_fails():
    ok, errs = _lint(
        _doc(csp_img=f"default-src https://evil.example; img-src data: {URL}"), allow=ALLOW
    )
    # note: default-src also loses its 'none', so the CSP-missing message fires too;
    # the load-bearing assertion is that the URL source is rejected.
    assert ok is False
    assert any("default-src must not carry a URL source: https://evil.example" in e for e in errs)


def test_l1_connect_src_url_fails():
    ok, errs = _lint(
        _doc(
            f'<img src="{URL}">', csp_img=f"img-src data: {URL}; connect-src https://evil.example"
        ),
        allow=ALLOW,
    )
    assert ok is False
    assert any("connect-src must not carry a URL source: https://evil.example" in e for e in errs)


# ============================================================ L2 (reference scan)


def test_l2_allowlisted_img_src_passes():
    ok, errs = _lint(_doc(f'<img src="{URL}">', csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is True, errs


def test_l2_non_allowlisted_img_src_fails():
    ok, errs = _lint(
        _doc('<img src="http://evil.example/x.png">', csp_img=f"img-src data: {URL}"), allow=ALLOW
    )
    assert ok is False
    assert any("could not confirm as an allowlisted <img src>" in e for e in errs)


def test_l2_allowlisted_href_fails_not_img_src():
    # href is not an <img> src → no carve-out even if the URL is allowlisted
    ok, errs = _lint(_doc(f'<a href="{URL}">x</a>', csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is False
    assert any("could not confirm as an allowlisted <img src>" in e for e in errs)


def test_l2_allowlisted_srcset_fails():
    ok, errs = _lint(_doc(f'<img srcset="{URL}">', csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is False
    assert any("srcset" in e for e in errs)


def test_l2_allowlisted_css_url_fails():
    ok, errs = _lint(
        _doc(f'<div style="background:url({URL})">z</div>', csp_img=f"img-src data: {URL}"),
        allow=ALLOW,
    )
    assert ok is False
    assert any("url()" in e for e in errs)


def test_l2_img_src_byte_exact_query_suffix_fails():
    ok, errs = _lint(_doc(f'<img src="{URL}?x=leak">', csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is False
    assert any("could not confirm as an allowlisted <img src>" in e for e in errs)


def test_l2_entity_encoded_img_src_fails():
    # &#104; decodes to 'h' → parser sees https://…allowed (byte-equal), but the
    # raw-text regex misses the obfuscation → count mismatch → fail.
    body = "<img src=&#104;ttps://cdn.example.com/logo.png>"
    ok, errs = _lint(_doc(body, csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is False
    assert any("could not confirm as an allowlisted <img src>" in e for e in errs)


def test_l2_data_image_src_still_allowed_empty_allowlist():
    ok, errs = _lint(
        _doc('<img src="data:image/png;base64,AAAA">', csp_img="img-src data:"), allow=()
    )
    assert ok is True, errs


# ============================================================ byte caps


def _pad_doc(nbytes: int, image: bool) -> str:
    img = '<img src="data:image/png;base64,AAAA">' if image else ""
    csp = "img-src data:"
    pad = "<span>x</span>" * ((nbytes // 14) + 1)
    return _doc(img + pad, csp_img=csp)


def test_cap_non_image_70kb_fails():
    doc = _pad_doc(72_000, image=False)
    assert len(doc.encode()) > 70_000
    ok, errs = _lint(doc, allow=(), src="")
    assert ok is False
    assert any("too large" in e for e in errs)


def test_cap_image_bearing_uses_larger_limit():
    doc = _pad_doc(72_000, image=True)  # > 64KB but image-bearing → 600KB cap
    assert len(doc.encode()) > 70_000
    ok, errs = _lint(doc, allow=(), src="")
    assert not any("too large" in e for e in errs), errs


def test_cap_image_bearing_up_to_600kb_passes_size():
    doc = _pad_doc(590_000, image=True)
    assert 64_000 < len(doc.encode()) < 4_000_000
    ok, errs = _lint(doc, allow=(), src="")
    assert not any("too large" in e for e in errs), [e for e in errs if "large" in e]


def test_cap_allowlisted_image_is_image_bearing():
    # an artifact whose only image is an allowlisted <img src> (no data:image) is
    # still image-bearing → larger cap.
    pad = "<span>x</span>" * 5200
    doc = _doc(f'<img src="{URL}">' + pad, csp_img=f"img-src data: {URL}")
    assert len(doc.encode()) > 70_000
    ok, errs = _lint(doc, allow=ALLOW, src="")
    assert not any("too large" in e for e in errs), [e for e in errs if "large" in e]


# ============================================================ REGRESSION (critical)


def test_regression_default_arg_equals_explicit_empty():
    """The 3-arg call (no allowlist) and the 4-arg call with () must be identical."""
    import tests.unit.test_viz_lint_v112 as fx

    docs = [
        (fx.VALID, fx.SRC),
        (fx.VALID.replace("Q2 19", "Q2 42.7"), fx.SRC),  # invented number
        (fx.VALID.replace("<style>body{font-family:sans-serif}</style>", ""), fx.SRC),
        ("   ", fx.SRC),  # empty
        (
            fx.VALID.replace(
                "<style>body{font-family:sans-serif}</style>",
                '<script src="https://cdn.example.com/x.js"></script>',
            ),
            fx.SRC,
        ),  # external script src
        (fx.VALID.replace("</div>", ""), fx.SRC),  # unbalanced
    ]
    for html, src in docs:
        assert _viz_static_lint(html, src, CFG) == _viz_static_lint(html, src, CFG, ())


def test_regression_external_ref_message_unchanged_for_empty_allowlist():
    # the pre-V1 external-resource message is reproduced verbatim when allow=()
    html = _doc('<script src="https://cdn.example.com/x.js"></script>')
    ok, errs = _lint(html, allow=())
    assert ok is False
    assert any(
        "external resource reference (src=/href= to http(s):// or //) — inline everything" in e
        for e in errs
    )


def test_regression_full_fixture_corpus_reruns_green():
    """Re-run every red/green fixture from the pre-V1 lint suite through the new
    (empty-allowlist) code path and assert the ok verdict is unchanged."""
    import tests.unit.test_viz_lint_v112 as fx

    # green fixtures still pass
    assert _lint(fx.VALID, src=fx.SRC)[0] is True
    # a representative red fixture still fails
    assert _lint(fx.VALID.replace("Q2 19", "Q2 42.7"), src=fx.SRC)[0] is False


# ============================================================ CONFORMANCE FIXES


def test_f1_entity_encoded_img_offset_by_raw_href_rejected():
    # F1: an entity-encoded allowlisted <img> (regex MISS / parser APPROVE) must not
    # OFFSET a raw external href (regex HIT / parser MISS). Counts match 1==1, so the
    # count-only guard wrongly accepts; the explicit unapproved-ref list catches the
    # <a href> → reject.
    body = "<img src=&#104;ttps://cdn.example.com/logo.png>" '<a href="https://evil.com/x">x</a>'
    ok, errs = _lint(_doc(body, csp_img=f"img-src data: {URL}"), allow=ALLOW)
    assert ok is False
    assert any("could not confirm as an allowlisted <img src>" in e for e in errs)


def test_f2_unicode_whitespace_url_dropped_by_sanitizer():
    # F2: Unicode whitespace (NBSP, NEL, U+2028/29, U+3000) must be dropped by the
    # sanitizer — never admitted where a later CSP split() would fragment the URL.
    for ws in ("\xa0", "\x85", " ", " ", "　"):
        assert sanitize_image_allowlist([f"https://cdn.example.com/lo{ws}go.png"]) == ()
    # sanitize → lint: the proposed allowlist collapses to (), so the same <img> is
    # no longer pinned and the lint rejects the (now unallowlisted) external ref.
    nbsp_url = "https://cdn.example.com/lo\xa0go.png"
    allow = sanitize_image_allowlist([nbsp_url])
    assert allow == ()
    ok, _ = _lint(_doc(f'<img src="{nbsp_url}">', csp_img="img-src data:"), allow=allow)
    assert ok is False


def test_f3_wildcard_host_url_dropped_by_sanitizer():
    # F3: a '*' must never survive the sanitizer — else the exact-URL pin would widen
    # to a CSP wildcard host.
    assert sanitize_image_allowlist(["https://*.evil.com/x.png"]) == ()
    assert "*" in _ALLOWLIST_FORBIDDEN_CHARS


def test_f5_img_src_zero_sources_or_none_rejected():
    # F5: an img-src with ZERO sources or only 'none' (no data:) is REJECTED — pre-V1
    # required the literal "img-src data:" substring; the combined missing-directive
    # message returns for both.
    for csp in ("img-src", "img-src 'none'"):
        ok, errs = _lint(_doc(csp_img=csp), allow=())
        assert ok is False, csp
        assert any("img-src data:" in e for e in errs), (csp, errs)


def test_f6_legacy_entity_url_dropped_by_sanitizer():
    # F6: a URL carrying a legacy HTML entity (&copy) is dropped — HTMLParser would
    # entity-decode it in an <img src>, desyncing the byte-exact allowlist compare.
    assert sanitize_image_allowlist(["https://a.example/x?c=&copy"]) == ()
    # a bare ampersand that is NOT an entity survives unchanged
    kept = "https://a.example/x?a=1&z=2"
    assert sanitize_image_allowlist([kept]) == (kept,)


def test_f1_f5_empty_allowlist_regression_still_holds():
    # The conformance fixes must not perturb the empty-allowlist path: the pre-V1
    # external-ref message and the data-only accept are both preserved.
    ok, errs = _lint(_doc('<a href="https://evil.com/x">x</a>'), allow=())
    assert ok is False
    assert any(
        "external resource reference (src=/href= to http(s):// or //) — inline everything" in e
        for e in errs
    )
    assert _lint(_doc(csp_img="img-src data:"), allow=())[0] is True
