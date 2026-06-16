"""Secret / PII redaction (v0.5.0).

Applied on the *memory/RAG write path only* — the raw conversation
transcript is never redacted. The goal is to prevent credentials and
obvious personal identifiers from being embedded into long-term memory
and resurfacing in a later system prompt's RAG block.

This is best-effort pattern matching, not a guarantee. It deliberately
errs toward redacting credential-shaped strings (high value, low
false-positive cost) and is conservative about generic PII.
"""

from __future__ import annotations

import re

_PLACEHOLDER = "[REDACTED:{}]"

# Order matters: more specific patterns first. Each entry is (label, regex).
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Provider API keys (OpenAI/Anthropic/Google/xAI/OpenRouter/Tavily/Brave…)
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b")),
    ("api_key", re.compile(r"\bsk-or-[A-Za-z0-9_\-]{16,}\b")),
    ("api_key", re.compile(r"\btvly-[A-Za-z0-9_\-]{12,}\b")),
    ("api_key", re.compile(r"\bBSA[A-Za-z0-9_\-]{12,}\b")),
    ("api_key", re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}\b")),  # Google
    ("api_key", re.compile(r"\bxai-[A-Za-z0-9_\-]{16,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    # Bearer / JWT
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b", re.IGNORECASE)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    # Generic key=value secret assignments
    (
        "secret_kv",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token|access[_-]?key)\b\s*[=:]\s*"
            r"['\"]?([A-Za-z0-9._\-/+]{8,})['\"]?"
        ),
    ),
    # Credit-card-like (13–16 digits, optional separators)
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # Email
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]


def redact_findings(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, labels_found). Never raises."""
    if not text:
        return text, []
    found: list[str] = []
    out = text
    for label, pat in _PATTERNS:

        def _sub(m, _label=label):
            found.append(_label)
            return _PLACEHOLDER.format(_label)

        try:
            # For secret_kv keep the key name, redact only the value group.
            if label == "secret_kv":

                def _sub_kv(m):
                    found.append("secret_kv")
                    whole = m.group(0)
                    val = m.group(1)
                    return whole.replace(val, _PLACEHOLDER.format("secret"))

                out = pat.sub(_sub_kv, out)
            else:
                out = pat.sub(_sub, out)
        except Exception:
            continue
    return out, found


def redact(text: str) -> str:
    """Return text with secrets/PII replaced by ``[REDACTED:…]`` markers."""
    return redact_findings(text)[0]
