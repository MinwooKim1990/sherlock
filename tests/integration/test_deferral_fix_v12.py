"""v1.2 deferral fix — when LLM-3 hands LLM-1 an unrolled chain, the slot must
push LLM-1 to ANSWER (with assumptions) rather than defer with a clarifying
question. Locks in the four coordinated edits:

  A. LLM-3 prompt grows a NULL HYPOTHESIS / falsifier brake (engine.py).
  B. _format_active_intent rule leads with "ANSWER FIRST" + explicit "do NOT
     ask for their location" (agent.py) — keeps the "answer the END" substring.
  C. TIER-3 label + inference header drop the deferral-licensing "SPECULATIVE —
     verify before relying" wording (agent.py).
  D. PINNED FACTS header permits reasonable assumptions instead of demanding
     perfection (agent.py).

Background: on a small model the prior rule led with "answer the END of the
chain FIRST", and the SPECULATIVE label + "treat as authoritative" pins made
deferral the prompt-endorsed safe move — so Sherlock asked for the user's
departure area instead of recommending anything. These tests pin the cure.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.inference.engine import DEFAULT_LLM3_PROMPT
from sherlock.memory.entry import MemorySource, MemoryType


def _llm3_with_chain(messages):
    last = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    if "MULTILINGUAL web-search sweep" in last or "META-COGNITION" in last:
        return "[]"
    # mimic the live failure: LLM-3 unrolls a "should I reconsider?" chain even
    # though the user just asked a direct "where should we go?" question.
    return json.dumps(
        {
            "hypotheses": [
                {"intent": "wants an indoor recommendation", "probability": 0.7, "evidence": ["e"]},
                {"intent": "surface question", "probability": 0.2, "evidence": []},
                {"intent": "other", "probability": 0.1, "evidence": []},
            ],
            "implied_chain": [
                "rain is forecast for Saturday",
                "the planned spot is outdoors",
                "so the original plan is no longer ideal",
                "should I change plans entirely?",
            ],
            "really_asking": "Should I proceed with the outdoor plan, or switch indoors?",
            "anticipated_next": [],
            "tools_recommended": [],
            "freshness_required": [],
            "confidence_overall": 0.7,
        }
    )


def _slot_with_chain(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=_llm3_with_chain,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent.chat("부모님 모시고 나들이 가려고")  # turn 1: infer runs post-response
    agent.chat("그럼 토요일에 어디로 가는 게 좋을까?")  # turn 2: chain carried forward
    # v1.4: the inference / active-intent block now rides the FINAL user message.
    return agent.inspect_last_turn().messages_passed_to_llm1[-1].content


# ---- Fix B: the consumption rule must lead with ANSWER, not defer ----------


def test_rule_leads_with_answer_first(tmp_path):
    sys_msg = _slot_with_chain(tmp_path)
    # the chain still renders (we did NOT remove implicit-ask handling) ...
    assert "REALLY ASKING" in sys_msg
    assert "Should I proceed with the outdoor plan" in sys_msg
    # ... but the FIRST imperative the model reads is to answer.
    assert "ANSWER FIRST" in sys_msg
    idx_answer = sys_msg.index("ANSWER FIRST")
    idx_end = sys_msg.index("answer the END")
    assert idx_answer < idx_end, "ANSWER FIRST must precede the chain-end clause"


def test_rule_forbids_asking_for_unmentioned_details(tmp_path):
    sys_msg = _slot_with_chain(tmp_path)
    flat = sys_msg.replace("\n", " ")
    assert "Do NOT ask for their location" in flat
    assert "NEVER replace the answer with a clarifying question" in flat
    # the frozen consumption-rule substring other tests depend on survives.
    assert "answer the END" in sys_msg


# ---- Fix C: the tier label / header must not license deferral --------------


def test_inference_label_is_defanged(tmp_path):
    sys_msg = _slot_with_chain(tmp_path)
    assert "SPECULATIVE" not in sys_msg
    assert "verify before relying" not in sys_msg
    # v1.4: the volatile block now rides the final user message under this fence
    # (replaced the in-system "TIER 3: ACTIVE ANALYSIS" header).
    assert "SYSTEM ANALYSIS FOR THIS TURN" in sys_msg
    # provenance is preserved — it's still "not a fact you quote", just usable.
    assert "INFERENCE HYPOTHESES" in sys_msg
    assert "do NOT quote it back as fact" in sys_msg


# ---- Fix D: pinned facts guide, they don't demand perfection ---------------


def test_pinned_header_permits_assumptions(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    conv = agent._ensure_conversation().id
    agent.memory.add(
        conversation_id=conv,
        content="user is a beginner driver",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        pinned=True,
    )
    block = agent._format_pinned_block(conv)
    assert "PINNED FACTS" in block  # frozen substring other tests rely on
    assert "treat as authoritative" not in block  # the perfectionism trigger is gone
    assert "make a reasonable assumption and state it" in block
    assert "rather than asking the user" in block


# ---- Fix A: LLM-3 gets a null-hypothesis brake against over-reading --------


def test_llm3_prompt_has_null_hypothesis_guard():
    assert "NULL HYPOTHESIS" in DEFAULT_LLM3_PROMPT
    assert "Constraint-listing is NOT hedging" in DEFAULT_LLM3_PROMPT
    # reassurance reads now require an explicit hedge marker
    assert "hedge marker" in DEFAULT_LLM3_PROMPT
    # the chain-unroll skill is retained but gated, not deleted
    assert "Implied-chain unrolling" in DEFAULT_LLM3_PROMPT
    assert "GATED by the null hypothesis" in DEFAULT_LLM3_PROMPT
