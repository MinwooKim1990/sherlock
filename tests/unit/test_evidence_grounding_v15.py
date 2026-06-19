"""v1.5 Stage 2 — evidence-grounded LLM-3.

Three additive, kill-switched behaviors, all OFF by default:
  * perception OBSERVED block fed into LLM-3's user message,
  * span-grounded evidence cap (a hypothesis with no verifiable verbatim quote
    is down-weighted to ≤0.35),
  * premise_conflict gap-detection field → routed into freshness_required.

OFF must leave the LLM-3 prompt + output byte-identical (the DEFAULT_LLM3_PROMPT
constant and its exact-key schema test in test_engine_v11 stay untouched).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock.inference.engine import (
    DEFAULT_LLM3_PROMPT,
    EVIDENCE_GROUNDING_EXTENSION,
    PREMISE_CONFLICT_EXTENSION,
    InferenceEngine,
    InferenceResult,
)
from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.providers.base import ChatMessage
from sherlock.providers.fake import FakeProvider
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


class _RecordingProvider(FakeProvider):
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


def _last_user_msg(p: _RecordingProvider) -> str:
    return next(m.content for m in reversed(p.seen[-1]) if m.role == "user")


def _hyp(intent, prob, evidence):
    return {
        "intent": intent,
        "probability": prob,
        "evidence": evidence,
        "search_keywords": [],
        "reasoning_type": "deduction",
    }


def _reply(hyps, **extra):
    d = {
        "hypotheses": hyps,
        "tools_recommended": [],
        "freshness_required": [],
        "confidence_overall": 0.5,
        "evolution_signals": {},
    }
    d.update(extra)
    return json.dumps(d)


def _infer(store, reply, **kw):
    p = _RecordingProvider(reply)
    eng = InferenceEngine(provider=p, store=store)
    out = eng.infer(
        conversation_id="c",
        turn_index=1,
        user_text=kw.pop("user_text", "hi"),
        recent_turns=kw.pop("recent_turns", [ChatMessage(role="user", content="hi")]),
        **kw,
    )
    return out, p


# ---------- OFF byte-identity ----------------------------------------------
def test_default_prompt_constant_unchanged():
    # The Stage-2 text lives in SEPARATE constants, never in the base prompt.
    assert "premise_conflict" not in DEFAULT_LLM3_PROMPT
    assert "EVIDENCE GROUNDING" not in DEFAULT_LLM3_PROMPT
    assert "PREMISE / KNOWLEDGE-GAP" not in DEFAULT_LLM3_PROMPT


def test_to_dict_premise_conflict_default_empty():
    assert InferenceResult().to_dict()["premise_conflict"] == []


def test_observations_absent_by_default(store):
    _out, p = _infer(store, _reply([_hyp("x", 0.5, ["clue"])]))
    assert "CODE OBSERVATIONS" not in _last_user_msg(p)


def test_grounding_off_leaves_probabilities(store):
    hyps = [_hyp("uncited", 0.9, ["no quote here"])]
    out, _p = _infer(store, _reply(hyps))  # ground_evidence defaults False
    assert out["hypotheses"][0]["probability"] == 0.9


def test_premise_conflict_absent_leaves_freshness(store):
    out, _p = _infer(store, _reply([_hyp("x", 0.5, ["c"])], freshness_required=["weather"]))
    assert out["premise_conflict"] == []
    assert out["freshness_required"] == ["weather"]


# ---------- observations injection -----------------------------------------
def test_observations_injected_when_passed(store):
    _out, p = _infer(
        store,
        _reply([_hyp("x", 0.5, ["c"])]),
        observations="OBSERVED (code-verified):\n- 2026-12-27 is a Sunday",
    )
    um = _last_user_msg(p)
    assert "CODE OBSERVATIONS" in um and "2026-12-27 is a Sunday" in um


# ---------- span-grounded evidence cap -------------------------------------
def test_span_grounding_caps_uncited_hypotheses(store):
    hyps = [
        _hyp("grounded", 0.8, ['the user said "buy the pass"']),
        _hyp("uncited", 0.9, ["pure speculation, no quote"]),
        _hyp("fabricated", 0.7, ['"I never said this phrase"']),
    ]
    out, _p = _infer(
        store,
        _reply(hyps),
        user_text="should I buy the pass early?",
        recent_turns=[ChatMessage(role="user", content="should I buy the pass early?")],
        ground_evidence=True,
    )
    h = {x["intent"]: x for x in out["hypotheses"]}
    assert h["grounded"]["probability"] == 0.8  # quote IS in the corpus → kept
    assert h["uncited"]["probability"] <= 0.35  # no quote → capped
    assert h["fabricated"]["probability"] <= 0.35  # quote not in corpus → capped
    assert h["uncited"].get("grounding") == "uncited — capped"


def test_span_grounding_trivial_2char_quote_does_not_ground(store):
    # AUDIT BUG-2: a junk 2-char quote ("in") substring-matching the corpus must
    # NOT rescue a speculation from the cap.
    hyps = [_hyp("dodge", 0.95, ['speculation with a junk token "in" inside'])]
    out, _p = _infer(
        store,
        _reply(hyps),
        user_text="should I book the meeting room",
        recent_turns=[ChatMessage(role="user", content="should I book the meeting room")],
        ground_evidence=True,
    )
    assert out["hypotheses"][0]["probability"] <= 0.35  # "in"⊂"meeting" must not save it


def test_span_grounding_word_boundary(store):
    # A real word quote grounds only at word boundaries, not mid-word.
    hyps = [_hyp("real", 0.9, ['the user said "meeting"'])]
    out, _p = _infer(
        store,
        _reply(hyps),
        user_text="should I book the meeting room",
        recent_turns=[ChatMessage(role="user", content="should I book the meeting room")],
        ground_evidence=True,
    )
    assert out["hypotheses"][0]["probability"] == 0.9  # "meeting" is a whole word → grounded


def test_span_grounding_cjk_quote_grounds(store):
    # AUDIT BUG-3: a verbatim CJK quote in native 「」 brackets must ground (not be
    # falsely capped) — query-language i18n.
    hyps = [_hyp("ja", 0.8, ["ユーザーは「メールを送る」と書いた"])]
    out, _p = _infer(
        store,
        _reply(hyps),
        user_text="今夜メールを送るべき？",
        recent_turns=[ChatMessage(role="user", content="今夜メールを送るべき？")],
        ground_evidence=True,
    )
    assert out["hypotheses"][0]["probability"] == 0.8  # 「メールを送る」 is verbatim in corpus


def test_span_grounding_handles_non_dict_hypothesis(store):
    # A non-dict hypothesis element must not crash grounding or the decay pass.
    reply = json.dumps(
        {
            "hypotheses": [_hyp("ok", 0.9, ['"book the meeting"']), "junk-not-a-dict"],
            "tools_recommended": [],
            "freshness_required": [],
            "confidence_overall": 0.5,
            "evolution_signals": {},
        }
    )
    out, _p = _infer(
        store,
        reply,
        user_text="should I book the meeting",
        recent_turns=[ChatMessage(role="user", content="should I book the meeting")],
        ground_evidence=True,
    )
    assert out["hypotheses"][0]["probability"] == 0.9


def test_span_grounding_quote_in_observations_counts(store):
    # A quote that appears in the OBSERVED block (not the transcript) still grounds.
    hyps = [_hyp("obs-grounded", 0.8, ['observation says "is a Sunday"'])]
    out, _p = _infer(
        store,
        _reply(hyps),
        observations="OBSERVED:\n- 2026-12-27 is a Sunday",
        ground_evidence=True,
    )
    assert out["hypotheses"][0]["probability"] == 0.8


# ---------- premise_conflict routing (GATED) -------------------------------
def test_premise_conflict_parsed_and_routed_when_enabled(store):
    out, _p = _infer(
        store,
        _reply(
            [_hyp("x", 0.5, ["c"])],
            premise_conflict=["SpaceX IPO / stock status"],
            freshness_required=["existing topic"],
        ),
        premise_conflict=True,  # kill-switch ON
    )
    assert out["premise_conflict"] == ["SpaceX IPO / stock status"]
    # routed into the same search loop as freshness, original preserved
    assert "SpaceX IPO / stock status" in out["freshness_required"]
    assert "existing topic" in out["freshness_required"]


def test_premise_conflict_NOT_routed_when_off(store):
    # AUDIT BUG-1: a model emitting premise_conflict while the feature is OFF must
    # NOT mutate freshness_required (that would trigger an unasked web search).
    out, _p = _infer(
        store,
        _reply(
            [_hyp("x", 0.5, ["c"])],
            premise_conflict=["SpaceX IPO / stock status"],
            freshness_required=["existing topic"],
        ),
        # premise_conflict defaults False
    )
    assert out["premise_conflict"] == []
    assert out["freshness_required"] == ["existing topic"]  # byte-identical


def test_premise_conflict_malformed_types_safe(store):
    # AUDIT BUG-4: a scalar string must NOT explode into characters; non-str items
    # are dropped — even with the feature ON.
    for bad in ("SpaceX stock", [1, {"x": 1}, None, "real topic"], 42):
        out, _p = _infer(
            store,
            _reply([_hyp("x", 0.5, ["c"])], premise_conflict=bad),
            premise_conflict=True,
        )
        pc = out["premise_conflict"]
        assert all(isinstance(t, str) and len(t) > 1 for t in pc), (bad, pc)
        assert pc in ([], ["real topic"])  # scalar/non-list → [], list → only valid str


# ---------- agent-level prompt augmentation gating -------------------------
def _agent(tmp_path, name, **kw):
    from sherlock import Sherlock

    return Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        **kw,
    )


def test_agent_inferer_prompt_unaugmented_by_default(tmp_path):
    a = _agent(tmp_path, "off")
    assert a._llm3_prompt == DEFAULT_LLM3_PROMPT  # stored base prompt
    assert "EVIDENCE GROUNDING" not in a._inferer._prompt
    assert "premise_conflict" not in a._inferer._prompt


def test_cache_prefix_covers_full_augmented_prompt(store):
    # The LLM-3 system message marks its WHOLE static prompt as cacheable; with
    # the extension appended, the prefix length must still equal the full prompt.
    from sherlock.inference.engine import DEFAULT_LLM3_PROMPT as _base

    aug = _base + "\n\n" + EVIDENCE_GROUNDING_EXTENSION + "\n\n" + PREMISE_CONFLICT_EXTENSION
    p = _RecordingProvider(_reply([_hyp("x", 0.5, ["c"])]))
    eng = InferenceEngine(provider=p, store=store, system_prompt=aug)
    eng.infer(
        conversation_id="c",
        turn_index=1,
        user_text="hi",
        recent_turns=[ChatMessage(role="user", content="hi")],
    )
    sys_msg = p.seen[-1][0]
    assert sys_msg.role == "system"
    assert sys_msg.cache_stable_prefix_chars == len(sys_msg.content) == len(aug)


def test_agent_inferer_prompt_augmented_when_on(tmp_path):
    a = _agent(tmp_path, "on", evidence_grounding=True, premise_conflict=True)
    assert EVIDENCE_GROUNDING_EXTENSION.split("\n", 1)[0] in a._inferer._prompt
    assert PREMISE_CONFLICT_EXTENSION.split("\n", 1)[0] in a._inferer._prompt
    assert "premise_conflict" in a._inferer._prompt
    # the STORED prompt stays the byte-identical base (persisted/inspected)
    assert a._llm3_prompt == DEFAULT_LLM3_PROMPT
