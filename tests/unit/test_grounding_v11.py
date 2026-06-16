"""v1.1 grounding changes — R5 (LLM-2 few-shot anchor) + R35 (span-grounded facts).

Covers:
  - R5: DEFAULT_LLM2_PROMPT ends with ONE valid few-shot example that parses
    as JSON, matches the current schema (incl. the new "quote" subfield) and
    OMITS the optional "corrections" field
  - R35 grounded: quote verifies against the transcript → confidence kept
    as-is, quote appended to the entry's evidence JSON list, pin honored
  - R35 fuzzy fallback: non-verbatim quote with ≥80% of its non-stopword
    tokens in the transcript still counts as grounded
  - R35 ungrounded: quote NOT in transcript → confidence capped at 0.5,
    pin_recommended overridden to False, "ungrounded" tag set
  - quote omitted → byte-identical legacy behavior (confidence as given,
    pin honored, evidence untouched, no tags)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.summarizer import (
    DEFAULT_LLM2_PROMPT,
    SummarizerEngine,
    _quote_grounded,
)
from sherlock.providers.base import ChatMessage
from sherlock.providers.fake import FakeProvider
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    storage = Storage(tmp_path / "test.db")
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=build_embedding_provider(_FakeEmbedConfig()),
        vector_path=tmp_path / "vectors",
    )


class _RecordingProvider(FakeProvider):
    """FakeProvider that captures the messages each chat() call received."""

    def __init__(self, canned_reply: str) -> None:
        super().__init__(canned_reply=canned_reply)
        self.seen: list[list[ChatMessage]] = []

    def chat(self, messages, **kwargs):
        self.seen.append(list(messages))
        return super().chat(messages, **kwargs)


def _llm2_payload(facts: list[dict], **extra) -> str:
    payload = {
        "summary": "",
        "facts": facts,
        "topic_label": "test",
        "topic_changed_from_previous": False,
        "retrieval_keywords": [],
    }
    payload.update(extra)
    return json.dumps(payload)


_TRANSCRIPT_TURNS = [
    ChatMessage(
        role="user",
        content="quick note for you — yujin's allergy is BUCKWHEAT (메밀), peanuts are fine",
    ),
    ChatMessage(role="assistant", content="Got it — buckwheat, not peanuts."),
]


def _run_summarizer(
    store: MemoryStore,
    payload: str,
    *,
    recent_turns: list[ChatMessage] | None = None,
    turn_index: int = 3,
) -> dict:
    engine = SummarizerEngine(provider=_RecordingProvider(canned_reply=payload), store=store)
    return engine.run(
        conversation_id="c",
        recent_turns=recent_turns if recent_turns is not None else _TRANSCRIPT_TURNS,
        turn_index=turn_index,
    )


# ---------- (a) R5: few-shot example in DEFAULT_LLM2_PROMPT ------------------


def test_prompt_example_is_valid_json_and_matches_schema():
    head, tail = DEFAULT_LLM2_PROMPT.split("VALID EXAMPLE", 1)
    block = tail[tail.index("{") : tail.rindex("}") + 1]
    example = json.loads(block)  # must parse — small models anchor on it

    # The new "quote" subfield is both documented in the schema (above the
    # example) and exercised inside the example itself.
    assert '"quote"' in head
    fact = example["facts"][0]
    assert fact["quote"]
    for subfield in (
        "content",
        "type",
        "source",
        "confidence",
        "semantic_triple",
        "evidence",
        "pin_recommended",
        "let_fade",
    ):
        assert subfield in fact, f"few-shot fact is missing {subfield!r}"

    # Optional "corrections" is OMITTED in the example (explicit
    # contradictions only); every other top-level field is present.
    assert "corrections" not in example
    for key in (
        "summary",
        "facts",
        "topic_label",
        "topic_changed_from_previous",
        "retrieval_keywords",
        "persona_summary",
        "predicted_directions",
        "worth_digging",
    ):
        assert key in example, f"few-shot example is missing {key!r}"
    assert len(example["retrieval_keywords"]) == 1
    assert example["persona_summary"].strip()


# ---------- (b) R35: grounded quote ------------------------------------------


def test_grounded_quote_keeps_confidence_and_lands_in_evidence(store):
    payload = _llm2_payload(
        [
            {
                "content": "Yujin has a buckwheat allergy",
                "type": "fact",
                "source": "user",
                "confidence": 0.95,
                "evidence": ["user correction"],
                # Verbatim modulo whitespace/case — the exact-substring pass.
                "quote": "Yujin's allergy is BUCKWHEAT",
                "pin_recommended": True,
                "let_fade": False,
            }
        ]
    )
    _run_summarizer(store, payload)
    fact = next(r for r in store.list(conversation_id="c") if "buckwheat allergy" in r.content)
    assert fact.confidence == pytest.approx(0.95)
    ev = json.loads(fact.evidence)
    assert "user correction" in ev, "original evidence is preserved"
    assert "Yujin's allergy is BUCKWHEAT" in ev, "grounded quote is appended to evidence"
    assert fact.pinned is True
    assert "ungrounded" not in (fact.tags or "")


def test_fuzzy_fallback_grounds_non_verbatim_quote(store):
    # Not a verbatim substring (the transcript has "(메밀)," in the middle),
    # but every non-stopword token appears → ≥80% → grounded.
    payload = _llm2_payload(
        [
            {
                "content": "Peanuts are safe for Yujin",
                "type": "fact",
                "source": "user",
                "confidence": 0.9,
                "quote": "allergy is buckwheat and peanuts are fine",
                "pin_recommended": False,
            }
        ]
    )
    _run_summarizer(store, payload)
    fact = next(r for r in store.list(conversation_id="c") if "Peanuts are safe" in r.content)
    assert fact.confidence == pytest.approx(0.9)
    assert "allergy is buckwheat and peanuts are fine" in json.loads(fact.evidence)
    assert "ungrounded" not in (fact.tags or "")


def test_quote_grounded_helper_edges():
    assert _quote_grounded("Yujin's   ALLERGY is buckwheat", "yujin's allergy is buckwheat ok")
    assert not _quote_grounded("", "anything at all")
    # All-stopword quote carries zero signal for the FUZZY fallback — it
    # only grounds when it is a verbatim substring.
    assert not _quote_grounded("it is the and of", "totally unrelated words here")


# ---------- (c) R35: ungrounded quote ----------------------------------------


def test_ungrounded_quote_caps_confidence_blocks_pin_and_tags(store):
    payload = _llm2_payload(
        [
            {
                "content": "User's brother Minjun is seventeen",
                "type": "fact",
                "source": "user",
                "confidence": 0.95,
                # Never said anywhere in the transcript — fabricated span.
                "quote": "my brother minjun just turned seventeen",
                "pin_recommended": True,
            }
        ]
    )
    _run_summarizer(store, payload)
    fact = next(r for r in store.list(conversation_id="c") if "Minjun" in r.content)
    assert fact.confidence <= 0.5
    assert fact.pinned is False, "a hallucinated fact must never become protected ground truth"
    assert "ungrounded" in (fact.tags or "").split(",")
    # The fabricated quote is NOT laundered into evidence.
    assert "my brother minjun just turned seventeen" not in json.loads(fact.evidence)


# ---------- (d) quote omitted → byte-identical legacy behavior ---------------


def test_no_quote_is_byte_identical_legacy_behavior(store):
    payload = _llm2_payload(
        [
            {
                "content": "User prefers window seats on flights",
                "type": "fact",
                "source": "user",
                "confidence": 0.8,
                "evidence": ["booking chat"],
                "pin_recommended": True,
            }
        ]
    )
    _run_summarizer(store, payload)
    fact = next(r for r in store.list(conversation_id="c") if "window seats" in r.content)
    assert fact.confidence == pytest.approx(0.8), "no penalty when the quote is omitted"
    assert fact.pinned is True, "pin_recommended is honored exactly as today"
    assert json.loads(fact.evidence) == ["booking chat"], "evidence untouched"
    assert (fact.tags or "") == ""


def test_blank_quote_treated_as_omitted(store):
    payload = _llm2_payload(
        [
            {
                "content": "User adopted a cat named Phoebe",
                "type": "fact",
                "source": "user",
                "confidence": 0.85,
                "quote": "   ",
                "pin_recommended": True,
            }
        ]
    )
    _run_summarizer(store, payload)
    fact = next(r for r in store.list(conversation_id="c") if "Phoebe" in r.content)
    assert fact.confidence == pytest.approx(0.85)
    assert fact.pinned is True
    assert (fact.tags or "") == ""
