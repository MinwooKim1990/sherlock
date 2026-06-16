"""Optional LLMLingua-2 compression for fetched web content (roadmap R17).

Multi-round fetches are deep research's largest single token sink, and
fetched pages are mostly boilerplate/navigation — genuinely wasted tokens.
This module drops low-information tokens from fetched page text *before*
it reaches the research prompt. It is quality-preserving compression of
waste, NOT result restriction: every search result and every fetched page
still flows through — only the boilerplate inside each page is squeezed.
That honors the locked principle: no routing, no result caps — savings
come from waste only. Scope explicitly excludes user memories, pinned
facts, and protocol prompts (compression there risks meaning loss and
conflicts with provenance guarantees).

The dependency is optional (same pattern as fastembed in
``sherlock/memory/embeddings.py``). Install with::

    pip install "sherlock[compress]"

to get the BERT-size, CPU-friendly local LLMLingua-2 compressor (2-5x
typical compression on fetched pages). Without the extra installed,
``maybe_compress`` returns exactly the legacy ``text[:target_chars]``
truncation the caller would have done — zero behavior change.
"""

from __future__ import annotations

import inspect
import warnings

# LLMLingua-2 small multilingual model (BERT-size, CPU-friendly, local).
LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Lazy singleton — the model load is expensive, so it happens at most once.
_compressor = None
# One-time-warning latch: warn only the FIRST time compression was
# explicitly requested but llmlingua is missing.
_warned_missing = False


def is_available() -> bool:
    """True only when the optional ``llmlingua`` package imports."""
    try:
        import llmlingua  # noqa: F401  # lazy; optional dependency

        return True
    except Exception:
        return False


def _get_compressor():
    """Lazily build (once) and return the shared PromptCompressor."""
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor  # lazy; optional dependency

        _compressor = PromptCompressor(
            model_name=LLMLINGUA2_MODEL,
            use_llmlingua2=True,
        )
    return _compressor


def maybe_compress(
    text: str,
    *,
    target_chars: int,
    query: str = "",
    requested: bool = False,
) -> str:
    """Compress ``text`` toward ``target_chars`` if llmlingua is installed.

    Fallback contract (never raises): when llmlingua is unavailable, when
    the text is already short, or when compression fails for ANY reason,
    return exactly the legacy truncation ``text[:target_chars]``.

    ``requested=True`` means the caller explicitly asked for compression
    (e.g. via config); the first such call without llmlingua installed
    emits a one-time ``RuntimeWarning``. Default ``False`` = silent
    fallback.
    """
    global _warned_missing
    if not is_available():
        if requested and not _warned_missing:
            _warned_missing = True
            warnings.warn(
                "Compression was requested but llmlingua is not installed; "
                "falling back to plain truncation. Install with "
                '`pip install "sherlock[compress]"` to enable LLMLingua-2 '
                "compression of fetched web pages.",
                RuntimeWarning,
                stacklevel=2,
            )
        return text[:target_chars]
    if len(text) < target_chars * 1.2:
        # Not enough headroom for compression to pay for itself.
        return text[:target_chars]
    try:
        comp = _get_compressor()
        # Aim roughly at target_chars: rate is the kept fraction.
        rate = max(0.05, min(1.0, target_chars / max(1, len(text))))
        kwargs: dict = {"rate": rate}
        if query:
            # Pass the query as context when the API supports it (the
            # `question` parameter), so query-relevant tokens are kept.
            try:
                params = inspect.signature(comp.compress_prompt).parameters
                if "question" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    kwargs["question"] = query
            except (TypeError, ValueError):
                pass
        result = comp.compress_prompt(text, **kwargs)
        compressed = result.get("compressed_prompt") if isinstance(result, dict) else None
        if isinstance(compressed, str) and compressed:
            return compressed
        return text[:target_chars]
    except Exception:
        # ANY failure (model load, OOM, API drift, …) → legacy truncation.
        return text[:target_chars]
