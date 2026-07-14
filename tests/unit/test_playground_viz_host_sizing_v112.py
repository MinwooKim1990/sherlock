"""Browser-host regression guards for seamless inline visualization sizing."""

from pathlib import Path

APP_JS = Path(__file__).parents[2] / "playground" / "static" / "app.js"


def test_visualization_iframe_fits_content_without_internal_scroll() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "const VIZ_BOOT_H = 1" in source
    assert "VIZ_HARD_MAX_H = 12000" in source
    assert "VIZ_MAX_H" not in source
    assert 'iframe.setAttribute("scrolling", "no")' in source
    assert "withVizFitRuntime(html)" in source
    assert "new ResizeObserver(queue)" in source
    assert 'background:" + vizFrameBg() + ";overflow:hidden"' in source
