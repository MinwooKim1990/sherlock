"""Pure-stdlib perception layer (Stage 1).

Deterministic, per-turn observations about the user's latest message — the
class of things a *small* model routinely gets wrong but code computes for
free: date arithmetic (day-delta / weekday / business-days), script &
locale, structural spans (URLs / emails / IPs / UUIDs / paths), arithmetic,
and freshness/staleness cues. Split into two channels so a probabilistic
guess can never read as a hard fact:

  OBSERVED  — deterministic facts (no confidence; the code is certain).
  PRIOR     — probabilistic cues (explicit confidence; clearly *not* a fact).

Design rules (locked — see plan "잠금 원칙"):
  * Pure stdlib only — ``unicodedata, datetime, urllib.parse, ipaddress,
    uuid, decimal, re, ast, operator``. No heavy deps, ever. This module
    imports NOTHING from ``sherlock`` so it stays a dependency-free leaf
    (re-implementing tiny helpers like the negation set rather than
    importing ``agent.py`` and creating an import cycle into the hot path).
  * Flag-don't-guess — when a thing is not *certainly* computable, emit a
    low-confidence PRIOR or nothing at all; never fabricate an OBSERVED.
  * No network, no clock surprises — ``now`` is injected by the caller (the
    same instant the slot clock uses), never sampled here; host/IP checks
    are syntactic only (``ipaddress`` literals, ``localhost``), never DNS.
"""

from __future__ import annotations

import ast
import ipaddress
import operator
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, getcontext
from urllib.parse import urlparse

getcontext().prec = 34


@dataclass(frozen=True)
class Observation:
    """One perception result.

    channel: ``"observed"`` (deterministic fact) | ``"prior"`` (probabilistic).
    kind:    short machine key, e.g. ``"date_delta"`` / ``"script"`` / ``"url"``.
    text:    human-readable one-liner rendered into the slot.
    confidence: only set on PRIOR observations (None on OBSERVED).
    span:    the verbatim source substring, when one applies.
    """

    channel: str
    kind: str
    text: str
    confidence: float | None = None
    span: str | None = None


def perceive(
    message: str,
    *,
    now: datetime | None = None,
    history: list | None = None,
    config: object | None = None,
) -> list[Observation]:
    """Run the deterministic perception primitives over ``message``.

    Returns a (possibly empty) list of :class:`Observation`. ``now`` should be
    the same instant injected into the slot clock; defaults to UTC now. Each
    primitive is isolated so one raising never sinks the rest. ``config`` is an
    optional object with boolean per-primitive toggles (``dates``, ``scripts``,
    ``arithmetic``, ``spans``, ``code``, ``discourse``, ``freshness``); any
    missing attribute defaults to on.
    """
    msg = message or ""
    if not msg.strip():
        return []
    if now is None:
        now = datetime.now(timezone.utc)

    def _on(attr: str) -> bool:
        return config is None or bool(getattr(config, attr, True))

    scripts = _scripts(msg)  # cheap; gates the CJK-fragile primitives below
    out: list[Observation] = []
    plan = [
        ("scripts", lambda: _observe_scripts(scripts)),
        ("dates", lambda: _observe_dates(msg, now.date())),
        ("arithmetic", lambda: _observe_arithmetic(msg)),
        ("spans", lambda: _observe_spans(msg)),
        ("code", lambda: _observe_code(msg)),
        ("discourse", lambda: _observe_discourse(msg, scripts)),
        ("freshness", lambda: _observe_freshness(msg)),
    ]
    for attr, fn in plan:
        if not _on(attr):
            continue
        try:
            out.extend(fn())
        except Exception:
            # A single primitive failing must never break slot assembly.
            continue
    return out


# --------------------------------------------------------------------------
# script / locale
# --------------------------------------------------------------------------
def _classify_char(ch: str) -> str | None:
    """Return a script name for a single *letter*, or None for non-letters."""
    o = ord(ch)
    if ch.isascii():
        return "Latin" if ch.isalpha() else None
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "Hangul"
    if 0x3040 <= o <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "Katakana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "Han"
    if 0x0400 <= o <= 0x04FF:
        return "Cyrillic"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
        return "Arabic"
    if 0x0590 <= o <= 0x05FF:
        return "Hebrew"
    if 0x0E00 <= o <= 0x0E7F:
        return "Thai"
    if 0x0900 <= o <= 0x097F:
        return "Devanagari"
    if 0x0370 <= o <= 0x03FF:
        return "Greek"
    if unicodedata.category(ch).startswith("L"):
        return "Other"
    return None


def _scripts(msg: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in msg[:4000]:
        s = _classify_char(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


def _locale_hint(scripts: dict[str, int]) -> str:
    if "Hangul" in scripts:
        return "Korean"
    if "Hiragana" in scripts or "Katakana" in scripts:
        return "Japanese"
    if "Han" in scripts:
        return "Han (Chinese, or Japanese kanji)"
    if "Cyrillic" in scripts:
        return "Cyrillic script (e.g. Russian)"
    if "Arabic" in scripts:
        return "Arabic script"
    if "Hebrew" in scripts:
        return "Hebrew script"
    if "Thai" in scripts:
        return "Thai"
    if "Devanagari" in scripts:
        return "Devanagari (e.g. Hindi)"
    if "Greek" in scripts:
        return "Greek"
    return ""


def _observe_scripts(scripts: dict[str, int]) -> list[Observation]:
    # Only notable when a non-Latin script is present (plain English → silent).
    non_latin = {k: v for k, v in scripts.items() if k != "Latin"}
    if not non_latin:
        return []
    present = sorted(scripts, key=lambda k: -scripts[k])
    hint = _locale_hint(scripts)
    mixed = "Latin" in scripts and non_latin
    text = "message script: " + ", ".join(present)
    if hint:
        text += f" → {hint}"
    if mixed:
        text += " (mixed with Latin)"
    return [Observation("observed", "script", text)]


# --------------------------------------------------------------------------
# dates: day-delta / weekday / business-days (top small-model failure)
# --------------------------------------------------------------------------
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
# English month words that are also common ordinary words — only treat them as
# a month when Capitalized or paired with an explicit year (flag-don't-guess:
# "I may 5 times" must NOT read as a May 5th date).
_AMBIGUOUS_MONTH = {"may", "march", "august"}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# Use digit lookarounds, NOT \b: a date glued to a CJK particle ("2026-12-27에")
# has no \b between the ASCII '7' and the Korean word-char '에', so \b would
# silently miss the most important case. The trailing lookahead rejects ASCII
# word chars and version/path continuations ("2025.06.30.1234", "2026/12/27/x")
# but ALLOWS a non-ASCII particle ("2026-12-27에"). The separator must repeat
# (\2) so a mixed "2026-12/27" is not read as a date.
_ISO_RE = re.compile(r"(?<![\w./-])(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(?![A-Za-z0-9_]|[./-]\d|/)")
# A dot-separated number in an explicit version/build context is NOT a date.
_VERSION_CTX = re.compile(r"(?i)(?:\b(?:v|ver|version|rel|release|build|patch|sdk|api)\.?|#)\s*$")
_KO_RE = re.compile(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_JA_RE = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_EN1_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", re.I
)
_EN2_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?(?:,?\s+(\d{{4}}))?\b", re.I
)


def _business_days(d0: date, d1: date) -> int:
    """Weekdays (Mon–Fri) in the half-open interval (d0, d1] — absolute value."""
    if d1 == d0:
        return 0
    lo, hi = (d0, d1) if d1 > d0 else (d1, d0)
    days = (hi - lo).days
    full, extra = divmod(days, 7)
    bd = full * 5
    wd = lo.weekday()
    for i in range(1, extra + 1):
        if (wd + i) % 7 < 5:
            bd += 1
    return bd


def _resolve_date(today: date, month: int, day: int, year: int | None):
    """Return (date, explicit_year) or (None, None) when invalid."""
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None, None
    if year is not None:
        try:
            return date(year, month, day), True
        except ValueError:
            return None, None
    # No year given: resolve to the nearest *future* occurrence (this year, else
    # next year). The day-delta/weekday are deterministic given that rule; the
    # assumption is stated in the rendered text so it never over-claims.
    for y in (today.year, today.year + 1):
        try:
            cand = date(y, month, day)
        except ValueError:
            continue
        if cand >= today:
            return cand, False
    try:
        return date(today.year, month, day), False
    except ValueError:
        return None, None


def _fmt_date(today: date, target: date, explicit_year: bool) -> str:
    d = (target - today).days
    wd = target.strftime("%A")
    if d == 0:
        when = "today"
    elif d == 1:
        when = "tomorrow"
    elif d == -1:
        when = "yesterday"
    elif d > 0:
        when = f"{d} days from today"
    else:
        when = f"{abs(d)} days ago"
    bd_note = ""
    if abs(d) > 1:
        bd_note = f"; {_business_days(today, target)} business days"
    yr_note = "" if explicit_year else " (year assumed — nearest upcoming)"
    return f"{target.isoformat()} is a {wd} — {when}{bd_note}{yr_note}."


def _observe_dates(msg: str, today: date) -> list[Observation]:
    found: list[tuple[int, int, int | None]] = []  # (month, day, year|None)
    for m in _ISO_RE.finditer(msg):
        y, sep, mo, da = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if not (1900 <= y <= 2100):  # implausible "year" → a version like 9001.2.3
            continue
        if sep == "." and _VERSION_CTX.search(msg[max(0, m.start() - 12) : m.start()]):
            continue  # "release 2024.2.29" is a version, not a date
        found.append((mo, da, y))
    for rx in (_KO_RE, _JA_RE):
        for m in rx.finditer(msg):
            y = int(m.group(1)) if m.group(1) else None
            found.append((int(m.group(2)), int(m.group(3)), y))
    for m in _EN1_RE.finditer(msg):
        word, yr = m.group(1), (int(m.group(3)) if m.group(3) else None)
        if word.lower() in _AMBIGUOUS_MONTH and word[:1].islower() and yr is None:
            continue
        found.append((_MONTHS[word.lower()], int(m.group(2)), yr))
    for m in _EN2_RE.finditer(msg):
        word, yr = m.group(2), (int(m.group(3)) if m.group(3) else None)
        if word.lower() in _AMBIGUOUS_MONTH and word[:1].islower() and yr is None:
            continue
        found.append((_MONTHS[word.lower()], int(m.group(1)), yr))

    out: list[Observation] = []
    seen: set[tuple] = set()
    for mo, da, yr in found:
        target, explicit = _resolve_date(today, mo, da, yr)
        if target is None:
            continue
        key = (target, explicit)
        if key in seen:
            continue
        seen.add(key)
        out.append(Observation("observed", "date_delta", _fmt_date(today, target, explicit)))
        if len(out) >= 3:
            break
    return out


# --------------------------------------------------------------------------
# arithmetic (exact, Decimal — beats float-fuzzy small models)
# --------------------------------------------------------------------------
_ARITH_SPAN = re.compile(r"[\d\s().+\-*/%^,]+")
_DATEISH = re.compile(r"^\s*\d{1,4}([-/.])\d{1,2}\1\d{1,4}\s*$")
_IPISH = re.compile(r"^\s*\d{1,3}(\.\d{1,3}){2,3}\s*$")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _is_calc_expr(raw: str) -> bool:
    """High-precision gate: only fire on expressions the user clearly wants
    *computed*, never on idioms/ratios/versions/dates/times.

    Qualifies only when there is a whitespace-flanked operator ("100 - 37"),
    a parenthesised expression, or an exponent ("2 ** 10"). A bare "N op N"
    with no spaces (24/7, 9/10, 50/50, 12-27) is NOT arithmetic.
    """
    if _DATEISH.match(raw) or _IPISH.match(raw):
        return False
    s = raw.strip()
    if "**" in s:
        return True
    if ("(" in s or ")" in s) and any(op in s for op in "+-*/%^"):
        return True
    # whitespace-flanked operator is the strong "please compute" signal
    return bool(re.search(r"\d\s+[-+*/%^]\s+[\d(]", s))


def _boundary_ok(msg: str, start: int, end: int) -> bool:
    """Reject a span that is a *fragment* of a larger token — the char just
    before/after its non-space content is a letter or token-continuation char,
    meaning the run was sliced out of sci-notation ("1e10"→"10"), a time
    ("9:00"→"00"), or a digit-grouped/identifier token ("1_000"→"000"). The
    span's own leading/trailing whitespace is ignored (else "what is 1+1" trips
    on the 's' of 'is')."""
    s = start
    while s < end and msg[s].isspace():
        s += 1
    e = end
    while e > s and msg[e - 1].isspace():
        e -= 1
    before = msg[s - 1] if s > 0 else ""
    after = msg[e] if e < len(msg) else ""

    def bad(c: str) -> bool:
        return bool(c) and (c.isalpha() or c in "_:")

    return not (bad(before) or bad(after))


def _sane_expr(s: str) -> bool:
    """Reject malformed fragments: leading binary operator, trailing operator,
    or a double-operator run ("--5 + 1") that only a truncation produces."""
    if not s or s[0] in "*/%^" or s[-1] in "+-*/%^":
        return False
    probe = s.replace("**", "  ")
    return not re.search(r"[+\-*/%^]{2,}", probe)


def _eval_ast(n) -> Decimal:
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
            raise ValueError("non-numeric constant")
        return Decimal(str(n.value))
    if isinstance(n, ast.BinOp):
        op = _BINOPS.get(type(n.op))
        if op is None:
            raise ValueError("unsupported binop")
        lhs, rhs = _eval_ast(n.left), _eval_ast(n.right)
        if isinstance(n.op, ast.Pow):
            if abs(rhs) > 64 or rhs != rhs.to_integral_value():
                raise ValueError("pow out of range")
            return lhs ** int(rhs)
        if isinstance(n.op, (ast.Div, ast.Mod)) and rhs == 0:
            raise ValueError("division by zero")
        return op(lhs, rhs)
    if isinstance(n, ast.UnaryOp):
        op = _UNOPS.get(type(n.op))
        if op is None:
            raise ValueError("unsupported unaryop")
        return op(_eval_ast(n.operand))
    raise ValueError("unsupported node")


def _fmt_num(val: Decimal) -> str:
    if val == val.to_integral_value():
        return str(int(val))
    q = val.quantize(Decimal("0.0000000001"))
    return format(q.normalize(), "f")


def _observe_arithmetic(msg: str) -> list[Observation]:
    out: list[Observation] = []
    for m in _ARITH_SPAN.finditer(msg):
        if not _boundary_ok(msg, m.start(), m.end()):
            continue  # span is a fragment of a larger token (1e10, 9:00, 1_000)
        raw = m.group(0).strip()
        if not (3 <= len(raw) <= 120) or not _is_calc_expr(raw) or not _sane_expr(raw):
            continue
        expr = raw.replace(",", "").replace("^", "**")
        if len(_NUM.findall(expr)) < 2:
            continue
        try:
            val = _eval_ast(ast.parse(expr, mode="eval").body)
        except Exception:
            continue
        out.append(Observation("observed", "arithmetic", f"{raw} = {_fmt_num(val)}", span=raw))
        if len(out) >= 3:
            break
    return out


# --------------------------------------------------------------------------
# structural spans: URLs / emails / IPs / UUIDs / paths  (+ SSRF flag)
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"')]+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_IPV4_RE = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
_PATH_RE = re.compile(r"(?:^|\s)((?:/[\w.\-]+){2,}/?|[A-Za-z]:\\[\w.\\\-]+)")


def _ascii_host(host: str) -> str:
    """Truncate a host/domain at the first non-ASCII char — a hostname is ASCII
    (or punycode), so a glued CJK particle ("example.com에서") is not part of it."""
    cut = re.split(r"[^\x00-\x7f]", host, maxsplit=1)[0]
    return cut.rstrip(".,;")


def _host_flag(host: str) -> str | None:
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return "private/internal address (non-public)"
        return None
    except ValueError:
        h = host.lower()
        if h == "localhost" or h.endswith((".local", ".internal", ".localhost")):
            return "non-public host (localhost/internal)"
        return None


def _observe_spans(msg: str) -> list[Observation]:
    out: list[Observation] = []
    url_spans: list[tuple[int, int]] = []
    for m in _URL_RE.finditer(msg):
        url_spans.append(m.span())
        u = m.group(0).rstrip(".,);]")
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.netloc:
            continue
        host = _ascii_host(p.hostname or p.netloc)
        if not host:
            continue
        flag = _host_flag(host)
        text = f"URL present: {host}" + (f" — {flag}" if flag else "")
        ch = "observed"
        out.append(Observation(ch, "url", text, span=u))
        if sum(1 for o in out if o.kind == "url") >= 3:
            break

    for m in _EMAIL_RE.finditer(msg):
        dom = _ascii_host(m.group(0).rsplit("@", 1)[-1])
        if not dom or "." not in dom:
            continue
        out.append(
            Observation(
                "observed", "email", f"email address present (domain: {dom})", span=m.group(0)
            )
        )
        if sum(1 for o in out if o.kind == "email") >= 2:
            break

    for m in _IPV4_RE.finditer(msg):
        s, e = m.span()
        if any(a <= s < b for (a, b) in url_spans):
            continue  # part of a URL already reported
        try:
            ip = ipaddress.ip_address(m.group(0))
        except ValueError:
            continue
        kind = (
            "private/internal"
            if (ip.is_private or ip.is_loopback or ip.is_link_local)
            else "public"
        )
        out.append(
            Observation("observed", "ip", f"IPv4 literal {m.group(0)} ({kind})", span=m.group(0))
        )
        if sum(1 for o in out if o.kind == "ip") >= 3:
            break

    for m in _UUID_RE.finditer(msg):
        try:
            u = uuid.UUID(m.group(0))
        except ValueError:
            continue
        ver = f" (v{u.version})" if u.version else ""
        out.append(Observation("observed", "uuid", f"UUID present{ver}", span=m.group(0)))
        if sum(1 for o in out if o.kind == "uuid") >= 2:
            break

    for m in _PATH_RE.finditer(msg):
        p = m.group(1)
        out.append(
            Observation(
                "prior", "path", f"looks like a filesystem path: {p}", confidence=0.6, span=p
            )
        )
        if sum(1 for o in out if o.kind == "path") >= 2:
            break

    return out


# --------------------------------------------------------------------------
# code signals
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```([\w+#.\-]*)")
_FILE_LINE_RE = re.compile(r'File ".+", line \d+')
_LANG_HINTS = (
    (
        "Python",
        (re.compile(r"\bdef\s+\w+\s*\("), re.compile(r"\bimport\s+\w"), re.compile(r"\bprint\(")),
    ),
    (
        "JavaScript/TypeScript",
        (
            re.compile(r"\bfunction\s*\w*\s*\("),
            re.compile(r"=>"),
            re.compile(r"\bconsole\.log\("),
            re.compile(r"\b(?:const|let)\s+\w+\s*="),
        ),
    ),
    (
        "Java",
        (re.compile(r"\bpublic\s+(?:static\s+)?(?:class|void)\b"), re.compile(r"\bSystem\.out\.")),
    ),
    ("C/C++", (re.compile(r"#include\s*<"), re.compile(r"\bint\s+main\s*\("))),
    ("SQL", (re.compile(r"\bSELECT\b.+\bFROM\b", re.I | re.S),)),
    ("HTML/XML", (re.compile(r"</\w+>"),)),
)


def _guess_lang(msg: str) -> str | None:
    for lang, patterns in _LANG_HINTS:
        if any(p.search(msg) for p in patterns):
            return lang
    return None


def _observe_code(msg: str) -> list[Observation]:
    out: list[Observation] = []
    fences = _FENCE_RE.findall(msg)
    if fences:
        lang = next((f for f in fences if f), "")
        text = "fenced code block present" + (f" (lang: {lang})" if lang else "")
        out.append(Observation("observed", "code_block", text))
    if "Traceback (most recent call last)" in msg:
        out.append(
            Observation(
                "observed", "traceback", "Python traceback present → user is debugging an error."
            )
        )
    elif _FILE_LINE_RE.search(msg):
        out.append(
            Observation(
                "observed", "traceback", 'code traceback frame present (File "...", line N).'
            )
        )
    if not fences:
        lang = _guess_lang(msg)
        if lang:
            out.append(
                Observation(
                    "prior",
                    "code_signal",
                    f"message looks like code/config ({lang}).",
                    confidence=0.5,
                )
            )
    return out[:3]


# --------------------------------------------------------------------------
# discourse: anaphora / hedging  (negation handled in agent.py consistency)
# --------------------------------------------------------------------------
_ANAPHORA_KO = ("그거", "이거", "저거", "그것", "이것", "저것", "해당", "걔", "쟤")
# Fire English anaphora only on *pronominal* demonstratives (clause-final or
# before a verb) — NOT determiners ("this afternoon", "that file"). "it" is
# dropped entirely: dummy/expletive "it" ("it's 24/7", "it works") is too noisy.
_ANAPHORA_EN_RE = re.compile(
    r"\b(?:this|that|these|those)\b"
    r"(?=\s+(?:is|was|are|were|does|do|did|isn'?t|wasn'?t|won'?t|doesn'?t|don'?t|looks?|"
    r"seems?|works?|worked|broke|broken|fail(?:s|ed)?|happened|means?)\b|\s*[.?!,]|\s*$)",
    re.I,
)
_HEDGE = (
    "maybe",
    "perhaps",
    "probably",
    "i think",
    "i guess",
    "not sure",
    "might be",
    "아마",
    "아마도",
    "혹시",
    "글쎄",
    "잘 모르",
)


def _observe_discourse(msg: str, scripts: dict[str, int]) -> list[Observation]:
    out: list[Observation] = []
    words = re.findall(r"\w+", msg.lower(), flags=re.UNICODE)
    short = len(words) <= 6
    # Korean demonstratives are strongly anaphoric → fire on presence.
    if any(a in msg for a in _ANAPHORA_KO):
        out.append(
            Observation(
                "prior",
                "anaphora",
                "demonstrative/anaphora ('그거/이거' …) → likely refers to earlier conversation; weight history/RAG.",
                confidence=0.6,
            )
        )
    elif short and _ANAPHORA_EN_RE.search(msg):
        out.append(
            Observation(
                "prior",
                "anaphora",
                "short message anchored on a demonstrative (this/that/these/those) → likely refers to earlier context.",
                confidence=0.5,
            )
        )
    low = msg.lower()
    if any(h in low for h in _HEDGE):
        out.append(
            Observation(
                "prior",
                "hedge",
                "user is hedging/uncertain → they may want options or confirmation, not a single hard answer.",
                confidence=0.5,
            )
        )
    return out[:2]


# --------------------------------------------------------------------------
# freshness: live/time-sensitive request (strong keywords only → high precision)
# --------------------------------------------------------------------------
_FRESH_STRONG = (
    "stock price",
    "share price",
    "exchange rate",
    "real-time",
    "real time",
    "latest",
    "newest",
    "breaking news",
    "headlines",
    "weather forecast",
    "주가",
    "현재가",
    "시세",
    "환율",
    "최신",
    "실시간",
    "속보",
    "뉴스",
    "날씨",
    "今",
    "株価",
    "為替",
    "最新",
)
# "latest/newest <software-word>" is a version query, not a live-data request.
_SOFTWARE_AFTER = re.compile(
    r"(?i)\b(?:latest|newest)\s+(?:version|release|update|build|patch|driver|sdk|stable)\b"
)


def _observe_freshness(msg: str) -> list[Observation]:
    low = msg.lower()
    hit = []
    for kw in _FRESH_STRONG:
        probe = kw.lower()
        if (probe if probe.isascii() else kw) in (low if probe.isascii() else msg):
            hit.append(kw)
    if _SOFTWARE_AFTER.search(low):
        hit = [k for k in hit if k not in ("latest", "newest")]
    if not hit:
        return []
    # de-dup while preserving order, cap the listing
    seen = []
    for k in hit:
        if k not in seen:
            seen.append(k)
    listed = ", ".join(f"'{k}'" for k in seen[:4])
    return [
        Observation(
            "observed",
            "freshness",
            f"live/time-sensitive request — keyword(s): {listed} → answer needs CURRENT data; "
            "don't rely on training-cutoff knowledge.",
        )
    ]
