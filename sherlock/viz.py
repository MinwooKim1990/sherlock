"""v1.12 Stage B2: LLM-4 VISUALIZER — static lint + prompt builders.

Pure, side-effect-free helpers for the visualizer pipeline. The async
orchestration (render job, pool, dispatch) lives on the ``Sherlock`` agent
(``agent.py``); everything here is import-light (stdlib only) and unit-testable
in isolation.

The lint is the SANDBOX CONTRACT enforcer: an LLM-4 artifact is a single
self-contained HTML document rendered inside a locked-down sandboxed iframe
(the playground supplies ``sandbox="allow-scripts"`` + this inline CSP). The
lint statically rejects anything that would (a) escape the sandbox, (b) phone
home, or (c) invent data the material never contained. It is deliberately
conservative — a false reject just costs a repair round; a false accept ships
an unsafe or lying artifact.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# --------------------------------------------------------------------------- #
# Required skeleton                                                            #
# --------------------------------------------------------------------------- #

# The CSP meta LLM-4 must emit verbatim (the prompt asks for exactly this). The
# lint checks the individual directives (normalised, substring) rather than a
# byte-exact match, so trivial whitespace/quote drift doesn't force a needless
# repair round while still proving every directive is present.
CSP_META = (
    '<meta http-equiv="Content-Security-Policy" '
    "content=\"default-src 'none'; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; img-src data:\">"
)

# v1.12 Stage B4: the iframe→parent signalling protocol. The sandboxed artifact
# runs at an OPAQUE origin (sandbox="allow-scripts", NO allow-same-origin), so the
# host cannot read window.onerror across the boundary — instead the artifact posts
# a structured message the host correlates by ``event.source``. TWO signals, one
# shape ``{sherlockViz: 'ready'|'error', message?}``:
#   * READY — posted as the LAST thing after the visual paints (optionally carrying
#     the content height). The host waits for it (behind a ~4s runtime harness)
#     before it un-hides the frame; a missing ready means the visual never appears.
#   * ERROR — a top-of-script ``window.onerror`` handler forwards any runtime throw
#     so the host can repair immediately instead of waiting out the timeout.
# ``parent.postMessage`` is allowed ONLY as one of these two signals.
READY_SIGNAL = "parent.postMessage({sherlockViz:'ready'}, '*')"
ERROR_HANDLER = (
    "window.onerror = (e) => parent.postMessage({sherlockViz:'error', message:String(e)}, '*')"
)

# Stamped into the persisted artifact after a successful static lint so a
# re-hydrated artifact records HOW it was validated (Stage B3+ may add runtime).
VALIDATED_META = '<meta name="sherlock-viz-validated" content="static">'

# The CSP directives that must all be present (normalised substring match).
_CSP_REQUIRED = (
    "content-security-policy",
    "default-src 'none'",
    "script-src 'unsafe-inline'",
    "style-src 'unsafe-inline'",
    "img-src data:",
)

# Structural tags whose open/close balance is checked. Everything else (incl.
# SVG shape elements and void HTML elements) is exempt — void/self-closing tags
# never push onto the stack (see ``_VizHTMLParser``).
_STRUCTURAL = frozenset({"html", "body", "div", "svg", "script", "style"})


# --------------------------------------------------------------------------- #
# Forbidden patterns                                                          #
# --------------------------------------------------------------------------- #

# (compiled regex, human message). Case-insensitive. Kept as a table so the
# repair prompt can feed the exact message back to LLM-4.
_FORBIDDEN: list[tuple[re.Pattern[str], str]] = [
    # External resource refs: src=/href= pointing at http(s):// or a
    # protocol-relative //. Data URIs (img-src data:), fragment refs (#id) and
    # single-slash local paths are NOT matched. Inline SVG xmlns="http://..."
    # is an ``xmlns=`` attribute, not src/href, so it is left alone.
    (
        re.compile(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", re.IGNORECASE),
        "external resource reference (src=/href= to http(s):// or //) — inline everything",
    ),
    # F3 (defense-in-depth): further external-reference forms the src/href scan
    # above misses. The sandbox CSP already contains these (default-src 'none'),
    # but the lint is itself a security control, so we reject them statically.
    # OUT OF SCOPE (documented as contained): entity-decoded attribute values
    # (e.g. src=&#104;ttp…) — the sandbox CSP blocks the fetch regardless.
    (
        re.compile(r"""srcset\s*=\s*["']?\s*(?:https?:)?//""", re.IGNORECASE),
        "external resource reference (srcset= to http(s):// or //) — inline everything",
    ),
    (
        re.compile(r"""<form[^>]+action\s*=\s*["']?\s*(?:https?:)?//""", re.IGNORECASE),
        "external form action (form action= to http(s):// or //) — no navigation",
    ),
    (
        re.compile(r"""url\(\s*["']?\s*(?:https?:)?//""", re.IGNORECASE),
        "external resource reference (CSS url() to http(s):// or //) — inline everything",
    ),
    (re.compile(r"\bfetch\s*\(", re.IGNORECASE), "network call: fetch("),
    (re.compile(r"XMLHttpRequest", re.IGNORECASE), "network call: XMLHttpRequest"),
    (re.compile(r"WebSocket", re.IGNORECASE), "network call: WebSocket"),
    (re.compile(r"EventSource", re.IGNORECASE), "network call: EventSource"),
    (re.compile(r"navigator\.sendBeacon", re.IGNORECASE), "network call: navigator.sendBeacon"),
    (re.compile(r"\bimport\s*\(", re.IGNORECASE), "dynamic import()"),
    (re.compile(r"(?m)^\s*import\s+\S", re.IGNORECASE), "module import statement"),
    (re.compile(r"window\.top", re.IGNORECASE), "frame-busting: window.top"),
    (re.compile(r"window\.parent", re.IGNORECASE), "frame-busting: window.parent"),
    # F2 (navigation): the sandbox permits frame self-navigation and the inner
    # CSP has no navigation directive, so meta-refresh / location self-navigation
    # / window.open can issue a GET to an attacker URL (data in the query string).
    # The generation prompt already promises "no navigation"; the lint enforces it.
    (
        re.compile(r"<meta[^>]+http-equiv\s*=\s*[\"']?\s*refresh", re.IGNORECASE),
        "meta refresh navigation",
    ),
    (
        re.compile(r"location\s*\.\s*(?:href|assign|replace)", re.IGNORECASE),
        "navigation via location.*",
    ),
    (re.compile(r"\bwindow\.open\s*\(", re.IGNORECASE), "navigation via window.open"),
    (re.compile(r"document\.cookie", re.IGNORECASE), "storage access: document.cookie"),
    (re.compile(r"localStorage", re.IGNORECASE), "storage access: localStorage"),
    (re.compile(r"indexedDB", re.IGNORECASE), "storage access: indexedDB"),
    (re.compile(r"<(?:iframe|object|embed|base)\b", re.IGNORECASE), "forbidden element"),
    (re.compile(r"javascript:", re.IGNORECASE), "javascript: URL"),
]

# ``parent.postMessage(`` call sites — every one must be a sherlockViz ready/error
# signal (whitespace/quote-tolerant so trivial drift never forces a repair round).
_POSTMESSAGE_CALL = re.compile(r"parent\.postMessage\s*\(\s*", re.IGNORECASE)
_POSTMESSAGE_READY = re.compile(
    r"parent\.postMessage\s*\(\s*\{\s*sherlockViz\s*:\s*['\"]ready['\"]", re.IGNORECASE
)
_POSTMESSAGE_ERROR = re.compile(
    r"parent\.postMessage\s*\(\s*\{\s*sherlockViz\s*:\s*['\"]error['\"]", re.IGNORECASE
)

# Numeric tokens in TEXT content: integers/decimals with optional thousands
# commas and an optional trailing percent. Bare single digits and years are
# filtered downstream, not here.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


class _VizHTMLParser(HTMLParser):
    """Tracks structural-tag balance AND collects visible text (skipping the
    contents of ``<script>``/``<style>``). One pass serves both the balance
    check and the data-fidelity number extraction.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.balance_errors: list[str] = []
        self._suppress = 0  # >0 while inside script/style (text ignored)
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _STRUCTURAL:
            self.stack.append(tag)
        if tag in ("script", "style"):
            self._suppress += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._suppress > 0:
            self._suppress -= 1
        if tag not in _STRUCTURAL:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            # A structural tag deeper in the stack is closing before its inner
            # structural children were closed → mis-nested / unclosed inner tag.
            self.balance_errors.append(f"mis-nested or unclosed structural tag before </{tag}>")
            while self.stack:
                if self.stack.pop() == tag:
                    break
        else:
            self.balance_errors.append(f"unexpected close tag </{tag}>")

    def handle_data(self, data: str) -> None:
        if self._suppress == 0:
            self.text_parts.append(data)


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace runs to a single space."""
    return re.sub(r"\s+", " ", s).lower()


def _norm_num(s: str) -> str:
    """Drop commas/spaces/percent so ``1,234`` == ``1234`` and ``19%`` == ``19``."""
    return s.replace(",", "").replace(" ", "").replace("%", "")


def _significant_numbers(text: str) -> list[str]:
    """Numbers from TEXT worth checking for fidelity: ≥2 significant digits,
    excluding pure years (1900-2100) and the trivial 0/1/100. Returns the raw
    tokens (as they appeared) so the error message is legible."""
    out: list[str] = []
    for raw in _NUMBER_RE.findall(text):
        norm = _norm_num(raw)
        digits = norm.replace(".", "")
        if len(digits) < 2:
            continue  # single-digit / trivial
        try:
            value = float(norm)
        except ValueError:
            continue
        if value in (0.0, 1.0, 100.0):
            continue
        # pure integer year in [1900, 2100] is a label, not data
        if "." not in norm and 1900 <= value <= 2100:
            continue
        out.append(raw)
    return out


def _viz_static_lint(html: str, source_text: str, cfg) -> tuple[bool, list[str]]:  # noqa: ANN001
    """Statically validate an LLM-4 HTML artifact.

    Returns ``(ok, errors)`` where ``errors`` are precise English strings the
    repair prompt feeds back verbatim. All checks run (no short-circuit) so a
    single repair round sees every problem at once.

    Checks: non-empty; ``len(utf-8) <= cfg.max_html_bytes``; required CSP meta +
    the ``{sherlockViz:'ready'}`` ready signal + the ``window.onerror`` →
    ``{sherlockViz:'error'}`` handler present; structural tags balanced (stdlib
    ``html.parser``); no forbidden patterns (external refs / network / storage /
    frame-busting / disallowed elements / ``javascript:``); and DATA FIDELITY —
    every significant number rendered in the artifact's TEXT must appear in
    ``source_text`` (the marker description + data hint + surrounding reply
    slice), comma/space-insensitive.
    """
    errors: list[str] = []

    if not html or not html.strip():
        return False, ["empty document"]

    # F4: reserve headroom for the validated-meta injected in the finalize path.
    # ``_finalize_viz_render`` runs ``inject_validated_meta`` (~54 bytes) and emits
    # ``bytes=len(validated)``; capping the pre-injection HTML at
    # ``max_html_bytes - 128`` guarantees the emitted payload stays under the cap.
    max_bytes = int(getattr(cfg, "max_html_bytes", 64_000)) - 128
    n_bytes = len(html.encode("utf-8"))
    if n_bytes > max_bytes:
        errors.append(f"document too large: {n_bytes} bytes exceeds {max_bytes}")

    norm_html = _norm(html)
    missing = [d for d in _CSP_REQUIRED if d not in norm_html]
    if missing:
        errors.append(
            "missing Content-Security-Policy meta directive(s): "
            + ", ".join(missing)
            + f" — emit exactly: {CSP_META}"
        )

    if not _POSTMESSAGE_READY.search(html):
        errors.append(
            "missing ready signal — end the script with "
            "parent.postMessage({sherlockViz:'ready'}, '*') after render"
        )
    if not _POSTMESSAGE_ERROR.search(html):
        errors.append(
            "missing error handler — start the script with "
            "window.onerror = (e) => parent.postMessage({sherlockViz:'error', "
            "message:String(e)}, '*')"
        )

    # Structural balance + text extraction.
    parser = _VizHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - html.parser rarely raises
        errors.append(f"HTML did not parse: {type(exc).__name__}")
        return False, errors
    errors.extend(parser.balance_errors)
    if parser.stack:
        errors.append("unbalanced tags: unclosed " + ", ".join(f"<{t}>" for t in parser.stack))

    # Forbidden patterns (scan the whole document).
    for pattern, message in _FORBIDDEN:
        if pattern.search(html):
            errors.append(f"forbidden: {message}")
    # parent.postMessage only as a sherlockViz ready/error signal.
    n_calls = len(_POSTMESSAGE_CALL.findall(html))
    n_ok = len(_POSTMESSAGE_READY.findall(html)) + len(_POSTMESSAGE_ERROR.findall(html))
    if n_calls > n_ok:
        errors.append(
            "forbidden: parent.postMessage call other than the sherlockViz ready/error signal"
        )

    # Data fidelity: significant numbers in the rendered text must trace to the
    # source material.
    haystack = _norm_num(_norm(source_text or ""))
    # F1: join text nodes with a SPACE, not "". Adjacent bare-number labels in
    # SVG/DOM charts (``<text>12</text><text>19</text>``) would otherwise merge
    # into a fake token ``1219`` that fails as an invented number, failing valid
    # charts and burning every repair round → permanent viz.failed.
    text = " ".join(parser.text_parts)
    for token in _significant_numbers(text):
        if _norm_num(token) not in haystack:
            errors.append(f"invented number: {token} (not present in the source material)")

    return (not errors), errors


# --------------------------------------------------------------------------- #
# Fence stripping                                                             #
# --------------------------------------------------------------------------- #


def strip_code_fences(text: str) -> str:
    """Unwrap a ```html … ``` (or bare ``` … ```) fence if the model wrapped its
    output in one. Leaves un-fenced HTML untouched."""
    if not text:
        return ""
    t = text.strip()
    if not t.startswith("```"):
        return t
    nl = t.find("\n")
    if nl != -1:
        t = t[nl + 1 :]  # drop the ```lang opener line
    else:
        t = t[3:]
    t = t.rstrip()
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


# --------------------------------------------------------------------------- #
# Validated-meta injection                                                    #
# --------------------------------------------------------------------------- #


def inject_validated_meta(html: str) -> str:
    """Insert the ``sherlock-viz-validated`` meta before ``</head>`` (or, absent
    a head, right after the CSP meta; else prepend). Idempotent."""
    if VALIDATED_META in html:
        return html
    m = re.search(r"</head\s*>", html, re.IGNORECASE)
    if m:
        return html[: m.start()] + VALIDATED_META + "\n" + html[m.start() :]
    idx = html.lower().find("content-security-policy")
    if idx != -1:
        end = html.find(">", idx)
        if end != -1:
            return html[: end + 1] + "\n" + VALIDATED_META + html[end + 1 :]
    return VALIDATED_META + "\n" + html


# --------------------------------------------------------------------------- #
# LLM-4 prompts (English-internal; visible OUTPUT pinned to the user's language)#
# --------------------------------------------------------------------------- #

# The language rule marker — a stable substring the render pipeline (and tests)
# can assert is present in every LLM-4 prompt.
LANGUAGE_RULE = (
    "LANGUAGE: every visible label, title, legend and annotation must be in the "
    "SAME language as the material below — never translate the data's language."
)

VIZ_GENERATION_SYSTEM = (
    "You are a visualization coder. Produce ONE self-contained HTML document that "
    "renders a single small data visualization for embedding in a locked-down "
    "sandboxed iframe. Output ONLY the HTML document — no prose, no markdown, no "
    "code fences.\n"
    "\n"
    "REQUIRED SKELETON (exactly):\n"
    "- Start with <!DOCTYPE html>.\n"
    "- In <head>, include this inline meta VERBATIM:\n"
    f"  {CSP_META}\n"
    "- ALL CSS in inline <style>; ALL JS in inline <script>. No external "
    "stylesheets, scripts, fonts, images, or imports of any kind.\n"
    "- At the TOP of your <script>, install an error handler so a runtime throw is "
    "reported to the host:\n"
    "  window.onerror = (e) => parent.postMessage({sherlockViz:'error', "
    "message:String(e)}, '*');\n"
    "- After the visual has painted, signal readiness as the LAST thing your script "
    "does (optionally carry the content height):\n"
    "  parent.postMessage({sherlockViz:'ready', height: document.documentElement."
    "scrollHeight}, '*');\n"
    "\n"
    "TECHNIQUE: use inline SVG, <canvas>, or plain styled DOM — your choice. "
    "Interactivity (hover, click, tooltips) is allowed as long as it stays "
    "inside the document. Keep the whole document under the size cap.\n"
    "\n"
    "DATA FIDELITY: visualize ONLY the numbers and labels present in the material "
    "below — invent nothing. Do NOT introduce axis ticks, totals, percentages, or "
    "rounded values that are not in the material. If a number is not in the "
    "material, it must not appear in the output.\n"
    "\n"
    f"{LANGUAGE_RULE}\n"
    "\n"
    "SANDBOX: no network (no fetch/XMLHttpRequest/WebSocket/EventSource/"
    "sendBeacon), no storage (cookie/localStorage/indexedDB), no navigation, no "
    "window.top/window.parent access — signal ONLY via parent.postMessage("
    "{sherlockViz:'ready'|'error'}), no "
    "<iframe>/<object>/<embed>/<base>, no javascript: URLs, no external URLs."
)


def build_generation_user(job: dict) -> str:
    """The per-marker generation request: description, optional data hint, and the
    surrounding reply slice (context)."""
    parts = [
        "Render this visualization as one self-contained HTML document.",
        "",
        f"DESCRIPTION: {job.get('description', '')}",
    ]
    hint = job.get("data_hint") or ""
    if hint:
        parts.append(f"DATA HINT (the exact series/labels): {hint}")
    context = job.get("context") or ""
    if context:
        parts.append("")
        parts.append("SURROUNDING MATERIAL (the reply text around this visual):")
        parts.append(context)
    parts.append("")
    parts.append(
        "Output ONLY the HTML document. Use only the numbers/labels above; keep all "
        "visible text in the material's language."
    )
    return "\n".join(parts)


_REVIEW_CHECKLIST = (
    "Review your HTML below against this checklist:\n"
    "- Is the HTML syntactically valid and are all tags balanced?\n"
    "- Is the Content-Security-Policy meta present exactly as required?\n"
    "- Does the script install window.onerror → parent.postMessage({sherlockViz:"
    "'error', …}) at the top AND post parent.postMessage({sherlockViz:'ready'}, "
    "'*') AFTER the visual paints?\n"
    "- Are there zero external references, network calls, storage accesses, and "
    "imports?\n"
    "- Are all visible labels in the SAME language as the material?\n"
    "- Do all numbers come ONLY from the material (nothing invented)?\n"
    "Reply with the corrected FULL HTML document only — no prose, no fences."
)


def build_self_review_user(html: str) -> str:
    """Self-critique round: hand the model its own HTML + the checklist."""
    return f"{_REVIEW_CHECKLIST}\n\n---\n{html}"


def build_repair_user(html: str, errors: list[str]) -> str:
    """Repair round: the HTML + the exact validation errors to fix."""
    bullet = "\n".join(f"- {e}" for e in errors)
    return (
        "Your HTML document below FAILED validation with these exact errors:\n"
        f"{bullet}\n\n"
        "Fix every listed problem without changing the data. Reply with the "
        "corrected FULL HTML document only — no prose, no fences.\n\n"
        f"---\n{html}"
    )
