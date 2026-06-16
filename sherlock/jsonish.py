"""Lenient JSON recovery + one-shot retry for companion-model output (v1.0).

Small models slip: trailing commas, prose around the JSON, markdown fences.
The recovery ladder here fixes most of that for free (pure code). When even
the ladder fails, ONE retry that feeds the parse error back costs a fraction
of the call that would otherwise be wasted entirely — and the second failure
falls through to each caller's existing degraded fallback, so behavior after
two attempts is identical to before.
"""

from __future__ import annotations

import json
import re

from sherlock.providers.base import ChatMessage

# Telemetry: how many calls needed the retry (and how many were rescued by it).
RETRY_STATS = {"retries": 0, "rescued": 0}


def loads_lenient(s: str) -> object:
    """json.loads with a trailing-comma repair pass (a common small-model slip)."""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
    except Exception:
        return None


def extract_balanced(text: str, open_ch: str, close_ch: str) -> object:
    """Return the LONGEST parseable balanced {...}/[...] span (string-aware).
    Longest wins so bracketed prose like "[1] cite" before the real payload
    can't shadow it."""
    best: object = None
    best_len = 0
    start = text.find(open_ch)
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    span = text[start : i + 1]
                    if len(span) > best_len:
                        val = loads_lenient(span)
                        if val is not None:
                            best, best_len = val, len(span)
                    break
        start = text.find(open_ch, start + 1)
    return best


def safe_parse_json(text: str) -> object:
    text = (text or "").strip()
    if not text:
        return None
    val = loads_lenient(text)
    if val is not None:
        return val
    if "```" in text:
        body = text.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            val = loads_lenient(inner.strip())
            if val is not None:
                return val
    # Objects first (round answers / reviews), then arrays (question lists);
    # balanced + string-aware so prose like "[1] source" before the JSON is skipped.
    val = extract_balanced(text, "{", "}")
    if val is not None:
        return val
    return extract_balanced(text, "[", "]")


_RETRY_INSTRUCTION = (
    "Your previous reply was not parseable as JSON ({err}). "
    "Return ONLY the JSON — no prose, no code fences, no trailing commas."
)


def _chat_maybe_json_mode(provider, messages: list[ChatMessage], json_mode: bool):
    """v1.1 R6: constrained-decoding passthrough. When the provider route
    supports OpenAI-style ``response_format={"type": "json_object"}`` it gets
    it (litellm forwards/translates); routes that reject it are memoized OFF
    on the provider instance after one failure, so the cost is at most one
    failed API handshake per provider — never a lost completion."""
    if json_mode and getattr(provider, "_sherlock_json_mode_ok", True):
        try:
            return provider.chat(messages, response_format={"type": "json_object"})
        except Exception:
            try:
                provider._sherlock_json_mode_ok = False
            except Exception:
                pass
    return provider.chat(messages)


def chat_json_with_retry(
    provider,
    messages: list[ChatMessage],
    *,
    want=dict,
    on_usage=None,
    json_mode: bool = True,
):
    """Call ``provider.chat`` expecting JSON; on parse failure retry ONCE with
    the error fed back. Returns ``(parsed_or_None, last_response)``.

    ``want``: required parsed type (``dict``/``list``), or ``None`` for any
    non-None JSON value. ``on_usage(resp)`` fires per attempt (best-effort) so
    callers keep their token accounting. ``json_mode`` requests constrained
    JSON decoding where the provider supports it (R6) — a no-op elsewhere.
    """

    def _ok(p) -> bool:
        return isinstance(p, want) if want is not None else p is not None

    resp = _chat_maybe_json_mode(provider, messages, json_mode)
    if on_usage:
        try:
            on_usage(resp)
        except Exception:
            pass
    parsed = safe_parse_json(getattr(resp, "text", "") or "")
    if _ok(parsed):
        return parsed, resp

    RETRY_STATS["retries"] += 1
    err = (
        "no JSON found"
        if parsed is None
        else f"got {type(parsed).__name__}, expected {getattr(want, '__name__', 'JSON')}"
    )
    retry_messages = list(messages) + [
        ChatMessage(role="assistant", content=(getattr(resp, "text", "") or "")[:2000]),
        ChatMessage(role="user", content=_RETRY_INSTRUCTION.format(err=err)),
    ]
    resp2 = _chat_maybe_json_mode(provider, retry_messages, json_mode)
    if on_usage:
        try:
            on_usage(resp2)
        except Exception:
            pass
    parsed2 = safe_parse_json(getattr(resp2, "text", "") or "")
    if _ok(parsed2):
        RETRY_STATS["rescued"] += 1
        return parsed2, resp2
    return None, resp2
