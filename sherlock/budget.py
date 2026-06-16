"""Slot-budget infrastructure (v0.4.0).

The v0.4.0 slot rebuild allocates each block in the LLM-1 prompt a
*reserved budget* up front, then derives the K-turn raw-conversation tail
from the leftover. Each component (LLM-2, LLM-3, RAG) is told its budget
and must self-truncate to fit. This keeps total prompt size predictable
per call, friendly to prompt caching, and avoids context-window
overflow regardless of conversation length.

Public surface:

* :class:`SlotBudget` — per-block byte allocations + ``k_turn_budget``
  derivation.
* :data:`DEFAULT_PROFILE` / :data:`SMALL_MODEL_PROFILE` — sensible
  starting points for large vs small context windows.
* :func:`select_profile_for_window` — auto-pick a profile from a model's
  context-window size.
* :data:`CONTEXT_WINDOW_REGISTRY` — best-guess context-window sizes by
  model prefix; consumed by :func:`resolve_context_window`.
* :func:`count_tokens` — tiktoken-first, chars/4 fallback token count.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, asdict, replace
from typing import Optional

# ---------------------------------------------------------------------------
# Context-window registry (best-guess; users can override via ModelConfig)
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_REGISTRY: dict[str, int] = {
    # Anthropic Claude 4 family
    "claude-haiku-4-5*": 200_000,
    "claude-sonnet-4-6*": 200_000,
    "claude-opus-4-7*": 1_000_000,
    "claude-opus-4-5*": 200_000,
    "claude-3-5*": 200_000,
    "claude-3-7*": 200_000,
    # OpenAI
    "gpt-5.4*": 400_000,
    "gpt-5*": 400_000,
    "gpt-4o*": 128_000,
    "gpt-4-turbo*": 128_000,
    "gpt-4*": 32_000,
    # Google
    "gemini-2.5*": 1_000_000,
    "gemini-3.0*": 1_000_000,
    "gemini-3.1*": 2_000_000,
    "gemini-1.5*": 1_000_000,
    # Misc / local
    "ollama/*": 32_000,
    "*": 128_000,  # safe fallback for unknown
}


def resolve_context_window(model_id: str, override: Optional[int] = None) -> int:
    """Return the context-window size for ``model_id``.

    ``override`` (the user's ``ModelConfig.context_window``) wins outright.
    Otherwise we glob-match against :data:`CONTEXT_WINDOW_REGISTRY` —
    most-specific pattern first.
    """
    if override is not None and override > 0:
        return int(override)
    # Strip litellm-style provider prefix (anthropic/, openai/, etc.) for matching
    bare = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    candidates = [bare, model_id]
    # Sort patterns by specificity (longer pattern = more specific).
    patterns = sorted(CONTEXT_WINDOW_REGISTRY.keys(), key=len, reverse=True)
    for pat in patterns:
        for c in candidates:
            if fnmatch.fnmatchcase(c, pat):
                return CONTEXT_WINDOW_REGISTRY[pat]
    return CONTEXT_WINDOW_REGISTRY["*"]


# ---------------------------------------------------------------------------
# Slot budget
# ---------------------------------------------------------------------------


@dataclass
class SlotBudget:
    """Per-block token reservations for the LLM-1 slot.

    All values are *maxes*. Components are expected to self-truncate to
    fit. The K-turn tail (`k_turn_budget`) gets whatever's left.

    Defaults assume a ≥200K context window. For smaller models use
    :data:`SMALL_MODEL_PROFILE` (auto-selected via
    :func:`select_profile_for_window`).
    """

    sherlock_system_max: int = 5_000  # Tier 1 — Sherlock internal protocol
    tool_prompt_max: int = 3_000  # Tier 1 — tool docs (companion + sherlock-tool)
    user_system_max: int = 5_000  # Tier 1 — user's persona/role prompt
    compacted_memory_max: int = 30_000  # Tier 2 — pinned + persona summary + compact highlights
    inference_data_max: int = 30_000  # Tier 3 — LLM-3 hypotheses (when present)
    rag_max: int = 8_000  # Tier 3 — RAG retrieval + cached search
    output_reserve: int = 30_000  # response space (max_tokens upper bound)
    floor_k_turn_budget: int = 8_000  # never let K-turn dip below this
    # v1.2: cap the raw K-turn tail at this fraction of the context window so a
    # large window can't let raw history crowd out compaction (the user's
    # "fixed blocks + variable x% tail" design). 0.5 = tail ≤ half the window.
    k_turn_max_fraction: float = 0.5

    def total_reserved(self) -> int:
        return (
            self.sherlock_system_max
            + self.tool_prompt_max
            + self.user_system_max
            + self.compacted_memory_max
            + self.inference_data_max
            + self.rag_max
            + self.output_reserve
        )

    def k_turn_budget(self, ctx_window: int) -> int:
        """Tokens available for the rolling K-turn raw tail + current input."""
        remaining = int(ctx_window) - self.total_reserved()
        return max(remaining, self.floor_k_turn_budget)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# Sensible defaults for ≥200K context.
DEFAULT_PROFILE = SlotBudget()

# For 128K and smaller contexts, scale every reservation down by ~3×.
SMALL_MODEL_PROFILE = SlotBudget(
    sherlock_system_max=3_000,
    tool_prompt_max=2_000,
    user_system_max=3_000,
    compacted_memory_max=10_000,
    inference_data_max=10_000,
    rag_max=4_000,
    output_reserve=15_000,
    floor_k_turn_budget=4_000,
)

# v1.0: honest profiles for genuinely small windows (local 8K-32K models).
# Invariant: total_reserved() < window, so the K-turn tail is never starved
# by reservations alone (the SMALL profile's 15K output_reserve exceeded an
# 8K window outright — k_pool collapsed to 0).
PROFILE_8K = SlotBudget(
    sherlock_system_max=900,
    tool_prompt_max=400,
    user_system_max=800,
    compacted_memory_max=800,
    inference_data_max=400,
    rag_max=400,
    output_reserve=1_024,
    floor_k_turn_budget=1_200,
)
PROFILE_16K = SlotBudget(
    sherlock_system_max=1_200,
    tool_prompt_max=600,
    user_system_max=1_200,
    compacted_memory_max=2_000,
    inference_data_max=1_000,
    rag_max=800,
    output_reserve=2_048,
    floor_k_turn_budget=2_000,
)
PROFILE_32K = SlotBudget(
    sherlock_system_max=2_000,
    tool_prompt_max=1_000,
    user_system_max=2_000,
    compacted_memory_max=4_500,
    inference_data_max=2_500,
    rag_max=1_500,
    output_reserve=4_096,
    floor_k_turn_budget=3_000,
)


def select_profile_for_window(ctx_window: int) -> SlotBudget:
    """Pick a sane default profile from the model's context-window size."""
    if ctx_window < 12_000:
        return PROFILE_8K
    if ctx_window < 24_000:
        return PROFILE_16K
    if ctx_window < 48_000:
        return PROFILE_32K
    if ctx_window < 200_000:
        return SMALL_MODEL_PROFILE
    return DEFAULT_PROFILE


def apply_overrides(profile: SlotBudget, overrides: Optional[dict] = None) -> SlotBudget:
    """Return ``profile`` with per-field overrides applied (None-tolerant)."""
    if not overrides:
        return profile
    clean = {k: v for k, v in overrides.items() if v is not None and hasattr(profile, k)}
    return replace(profile, **clean)


# ---------------------------------------------------------------------------
# Tokenization (tiktoken-first, chars/4 fallback)
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODING = None
_TIKTOKEN_TRIED = False


def _get_tiktoken():
    """Lazy-load tiktoken's cl100k_base encoding.

    We pick ``cl100k_base`` even for Anthropic / Gemini because (a) it's
    a close approximation and (b) the slot budget already has a healthy
    margin. Real Claude / Gemini tokenizers are slower and require
    network or large model downloads — overkill for budget math.

    Returns ``None`` if tiktoken isn't available; callers must fall
    back to a heuristic.
    """
    global _TIKTOKEN_ENCODING, _TIKTOKEN_TRIED
    if _TIKTOKEN_TRIED:
        return _TIKTOKEN_ENCODING
    _TIKTOKEN_TRIED = True
    try:
        import tiktoken

        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TIKTOKEN_ENCODING = None
    return _TIKTOKEN_ENCODING


def count_tokens(text: str | None) -> int:
    """Best-effort token count for slot-budgeting.

    Uses tiktoken cl100k_base when available; otherwise approximates via
    ``len(text) / 4``. Either result is precise enough for the
    slot-budget walk-backward.
    """
    if not text:
        return 0
    enc = _get_tiktoken()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: 1 token ≈ 4 chars (English-ish average).
    return max(1, len(text) // 4)
