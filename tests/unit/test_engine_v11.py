"""v1.1 LLM-3 reliability fixes.

Covers:
  - R5: DEFAULT_LLM3_PROMPT carries a VALID EXAMPLE that json-parses to the
    CURRENT schema (context_to_expand/context_to_exclude stay removed)
  - R12: infer() renders the provenance ledger ONLY when system-persona
    entries exist (no persona → no LEDGER / USER-STATED block at all)
  - R20: plan_search seeds at least one counter-evidence query and
    generate_meta_questions demands a disconfirming question — with the
    frozen "MULTILINGUAL web-search sweep" / "META-COGNITION QUESTIONS"
    markers intact
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock.inference.engine import DEFAULT_LLM3_PROMPT, InferenceEngine
from sherlock.jsonish import extract_balanced
from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.providers.base import ChatMessage
from sherlock.providers.fake import FakeProvider
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


class _RecordingProvider(FakeProvider):
    """FakeProvider that captures the messages each chat() call received."""

    def __init__(self, canned_reply: str) -> None:
        super().__init__(canned_reply=canned_reply)
        self.seen: list[list[ChatMessage]] = []

    def chat(self, messages, **kwargs):
        self.seen.append(list(messages))
        return super().chat(messages, **kwargs)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    storage = Storage(tmp_path / "test.db")
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=build_embedding_provider(_FakeEmbedConfig()),
        vector_path=tmp_path / "vectors",
    )


_INFER_REPLY = json.dumps(
    {
        "hypotheses": [
            {
                "intent": "test intent",
                "probability": 0.5,
                "evidence": ["clue"],
                "search_keywords": [],
                "reasoning_type": "deduction",
            }
        ],
        "tools_recommended": [],
        "freshness_required": [],
        "confidence_overall": 0.5,
        "evolution_signals": {},
    }
)


def _last_user_msg(provider: _RecordingProvider) -> str:
    msgs = provider.seen[-1]
    return next(m.content for m in reversed(msgs) if m.role == "user")


# ---------- R5: few-shot anchor in the LLM-3 system prompt ----------------


def test_prompt_example_parses_to_current_schema():
    marker = "VALID EXAMPLE"
    assert marker in DEFAULT_LLM3_PROMPT
    tail = DEFAULT_LLM3_PROMPT.split(marker, 1)[1]
    example = extract_balanced(tail, "{", "}")
    assert isinstance(example, dict), "example must json-parse to an object"
    assert set(example) == {
        "hypotheses",
        "implied_chain",
        "really_asking",
        "anticipated_next",
        "tools_recommended",
        "freshness_required",
        "confidence_overall",
        "evolution_signals",
    }
    assert len(example["hypotheses"]) == 3
    for h in example["hypotheses"]:
        assert set(h) == {
            "intent",
            "probability",
            "evidence",
            "search_keywords",
            "reasoning_type",
        }
    assert set(example["evolution_signals"]) == {
        "user_pattern_observed",
        "good_inference_candidate",
    }


def test_prompt_has_no_removed_schema_fields():
    assert "context_to_expand" not in DEFAULT_LLM3_PROMPT
    assert "context_to_exclude" not in DEFAULT_LLM3_PROMPT
    # The example sits right before the frozen closing line.
    assert "JSON only. No prose around it." in DEFAULT_LLM3_PROMPT


# ---------- R12: ledger only renders when persona entries exist -----------


def test_infer_without_persona_skips_ledger(store):
    store.add(
        conversation_id="c",
        content="user likes tea",
        type=MemoryType.USER_UTTERANCE,
        source=MemorySource.USER,
    )
    provider = _RecordingProvider(_INFER_REPLY)
    engine = InferenceEngine(provider=provider, store=store)
    out = engine.infer(
        conversation_id="c",
        turn_index=1,
        user_text="should I email my boss tonight?",
        recent_turns=[ChatMessage(role="user", content="should I email my boss tonight?")],
    )
    assert out["hypotheses"], "scripted reply must round-trip"
    user_msg = _last_user_msg(provider)
    assert "LEDGER" not in user_msg
    assert "USER-STATED" not in user_msg
    assert "SYSTEM-PERSONA" not in user_msg
    assert "--- TRANSCRIPT" in user_msg


def test_infer_with_persona_keeps_ledger(store):
    store.add(
        conversation_id="c",
        content="user likes tea",
        type=MemoryType.USER_UTTERANCE,
        source=MemorySource.USER,
    )
    store.add(
        conversation_id="c",
        content="user's name is Jiwon (persona note)",
        type=MemoryType.FACT,
        source=MemorySource.SYSTEM,
    )
    provider = _RecordingProvider(_INFER_REPLY)
    engine = InferenceEngine(provider=provider, store=store)
    engine.infer(
        conversation_id="c",
        turn_index=1,
        user_text="did I tell you my name?",
        recent_turns=[ChatMessage(role="user", content="did I tell you my name?")],
    )
    user_msg = _last_user_msg(provider)
    assert "LEDGER" in user_msg
    assert "USER-STATED" in user_msg
    assert "SYSTEM-PERSONA" in user_msg
    assert "user likes tea" in user_msg
    assert "user's name is Jiwon (persona note)" in user_msg


# ---------- R20: counter-evidence / disconfirming-question seeding --------


def test_plan_search_seeds_counter_evidence_and_keeps_marker(store):
    provider = _RecordingProvider(
        '[{"lang":"en","keywords":"kyoto travel tips"},'
        '{"lang":"ja","keywords":"京都 観光 注意点"}]'
    )
    engine = InferenceEngine(provider=provider, store=store)
    out = engine.plan_search(topic="kyoto travel", user_lang="en")
    assert out, "scripted plan must round-trip"
    prompt = _last_user_msg(provider)
    assert "MULTILINGUAL web-search sweep" in prompt  # FROZEN marker
    assert "counter-evidence" in prompt


def test_generate_meta_questions_demands_disconfirming_and_keeps_marker(store):
    provider = _RecordingProvider('["is the strongest finding wrong?", "who disagrees?"]')
    engine = InferenceEngine(provider=provider, store=store)
    out = engine.generate_meta_questions(
        topic="kyoto travel",
        queries=["kyoto travel tips"],
        findings_digest="kyoto is popular in autumn",
        round_index=3,
    )
    assert out, "scripted questions must round-trip"
    prompt = _last_user_msg(provider)
    assert "META-COGNITION QUESTIONS" in prompt  # FROZEN marker
    assert "DISCONFIRMING" in prompt
