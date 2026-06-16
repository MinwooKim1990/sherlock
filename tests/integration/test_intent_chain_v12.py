"""v1.2 — implied-chain unrolling (LLM-3 → LLM-1 consumption), TODAY injection
into research prompts, strategy language matching, citation pairing flags."""

from __future__ import annotations

import json
from datetime import datetime

from sherlock import Sherlock
from sherlock.agent import DEFAULT_SHERLOCK_EXTENSION, _research_date_line


def _llm3_with_chain(messages):
    last = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    if "MULTILINGUAL web-search sweep" in last or "META-COGNITION" in last:
        return "[]"
    return json.dumps(
        {
            "hypotheses": [
                {"intent": "weekday means easier booking?", "probability": 0.7, "evidence": ["e"]},
                {"intent": "surface question", "probability": 0.2, "evidence": []},
                {"intent": "other", "probability": 0.1, "evidence": []},
            ],
            "implied_chain": [
                "that day is a weekday",
                "so people are working",
                "so trains less crowded?",
                "so reservations easy?",
            ],
            "really_asking": "can I delay buying the JR pass without losing seats?",
            "anticipated_next": [
                {
                    "question": "when do reservations open?",
                    "answer_hint": "1 month before, 10am JST",
                }
            ],
            "tools_recommended": [],
            "freshness_required": [],
            "confidence_overall": 0.7,
        }
    )


def test_chain_rides_to_next_turn_slot(tmp_path):
    """LLM-3's unrolled chain + really_asking + prepared next answer must
    appear in the NEXT turn's system prompt with the consumption rule."""
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
    agent.chat("그 날은 평일이잖아?")  # turn 1: infer runs in post-response
    agent.chat("그래서?")  # turn 2: carry-forward injected
    # v1.4: inference / active-intent now rides the FINAL user message.
    sys_msg = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "REALLY ASKING" in sys_msg
    assert "can I delay buying the JR pass" in sys_msg
    assert "that day is a weekday -> so people are working" in sys_msg
    assert "answer the END" in sys_msg  # consumption rule
    assert "when do reservations open?" in sys_msg
    assert "1 month before, 10am JST" in sys_msg  # prefetched answer


def test_chain_extras_cleared_on_new_session(tmp_path):
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
    agent.chat("hello")
    assert agent._pending_inference_extras.get("really_asking")
    agent.new_session()
    assert agent._pending_inference_extras == {}


def test_legacy_llm3_without_chain_unchanged(tmp_path):
    """Old-style LLM-3 output (no chain fields) → block renders exactly the
    legacy hypothesis lines, no chain scaffolding."""

    def llm3(messages):
        return json.dumps(
            {
                "hypotheses": [
                    {"intent": "a", "probability": 0.5, "evidence": []},
                    {"intent": "b", "probability": 0.3, "evidence": []},
                    {"intent": "c", "probability": 0.2, "evidence": []},
                ],
                "tools_recommended": [],
                "freshness_required": [],
                "confidence_overall": 0.5,
            }
        )

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=llm3,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent.chat("hi")
    agent.chat("again")
    # v1.4: inference / active-intent now rides the FINAL user message.
    sys_msg = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "INFERENCE HYPOTHESES" in sys_msg
    assert "REALLY ASKING" not in sys_msg


# ------------------------------------------------------------- TODAY lines


def test_research_date_line_has_current_year():
    line = _research_date_line()
    assert str(datetime.now().year) in line
    assert "올해" in line  # the relative-date instruction mentions Korean too


def test_today_reaches_strategy_round_and_synthesis_prompts(tmp_path):
    prompts: list[str] = []
    year = str(datetime.now().year)

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
        if "RESEARCH STRATEGY" in c:
            return "{}"
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "answers": "a",
                    "key_finding": "k",
                    "summary": "s",
                    "facts": [{"fact": "f", "sources": ["https://a.com/1"]}],
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL https://a.com/1"
        return "plain."

    from tests.integration.test_research_strategy_v10 import MiniEngine

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=MiniEngine(),
        inference_search_engine="disabled",
    )
    agent._run_deep_research(agent._ensure_conversation().id, "올해 12월 행사", 1, "drT")
    strategy_p = [p for p in prompts if "RESEARCH STRATEGY" in p]
    round_p = [p for p in prompts if "Answer these meta-questions" in p]
    synth_p = [p for p in prompts if "RESEARCH DOCUMENTS:" in p]
    for group, name in ((strategy_p, "strategy"), (round_p, "round"), (synth_p, "synthesis")):
        assert group, f"{name} prompt missing"
        assert all(f"TODAY is {year}" in p for p in group), f"{name} prompt lacks TODAY"


def test_strategy_prompt_carries_language_matching(tmp_path):
    prompts: list[str] = []

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
        return "{}"

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent._plan_research_strategy("topic", user_text="올해 12월 삿포로 행사 알려줘")
    sp = next(p for p in prompts if "RESEARCH STRATEGY" in p)
    assert "SAME language" in sp and "삿포로" in sp


# --------------------------------------------------------- citation pairing


def test_mispaired_citation_gets_flagged():
    fact_map = {
        "https://wi.jp/a": ["White Illumination ends December 25"],
        "https://mall.com/b": ["Aeon Mall food courts stay open through the holidays"],
    }
    text = (
        "White Illumination ends December 25 (https://wi.jp/a). "
        "Hotel restaurants are the safest dinner option (https://mall.com/b)."
    )
    out = Sherlock._flag_mispaired_citations(text, fact_map)
    assert "https://wi.jp/a (pairing unverified)" not in out
    assert "https://mall.com/b (pairing unverified)" in out


def test_pairing_flag_skipped_when_any_use_is_grounded():
    fact_map = {"https://a.com/x": ["the festival runs ten days"]}
    text = (
        "The festival runs ten days (https://a.com/x). " "See also https://a.com/x for directions."
    )
    out = Sherlock._flag_mispaired_citations(text, fact_map)
    assert "(pairing unverified)" not in out


# ------------------------------------------------------------ honesty line


def test_protocol_forbids_inventing_internal_logs():
    assert "never" in DEFAULT_SHERLOCK_EXTENSION.lower()
    assert "invent internal logs" in DEFAULT_SHERLOCK_EXTENSION


# ----------------------------------------------- localized ack/ask (v1.2.1)


def test_strategy_localizes_approval_and_ack(tmp_path):
    strategy_reply = json.dumps(
        {
            "objective": "삿포로 연말 행사 정리",
            "sub_topics": ["일루미네이션", "연말 휴관"],
            "scope": {"include": [], "exclude": []},
            "clarifying_questions": [],
            "approval_question": "삿포로 연말 행사를 딥리서치로 조사할까요? 웹 검색을 여러 번 사용합니다.",
            "user_ack": "딥리서치를 시작합니다 — 라운드별 발견을 바로바로 보여드릴게요.",
        }
    )

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return strategy_reply
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "answers": "a",
                    "key_finding": "k",
                    "summary": "s",
                    "facts": [],
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL"
        if "RESEARCHME" in c:
            return 'ok\n<<sherlock-tool: deep_research "삿포로 연말 행사">>'
        return "plain."

    from tests.integration.test_research_strategy_v10 import MiniEngine

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=MiniEngine(),
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: None)  # sink → background run → ack path
    ask = agent.chat("please RESEARCHME")
    # localized head + frozen approval words, no emoji
    assert "조사할까요" in ask and "run it" in ask
    assert "🔬" not in ask and "📋" not in ask
    agent.chat("yes")
    agent.wait_for_background(timeout=15)
    msgs = [m.content for m in agent.messages() if m.role == "assistant"]
    assert any("딥리서치를 시작합니다" in m for m in msgs), "localized ack must be persisted"
    assert not any("🔬" in m for m in msgs)


# -------------------------------------------- novelty-based convergence


def test_paraphrased_facts_converge_but_are_kept(tmp_path):
    """New URLs keep arriving, but every round restates the SAME conclusion in
    slightly different words → the knowledge stall fires (converged_no_new_facts)
    while every restatement is still STORED (nothing dropped)."""
    events: list[tuple[str, dict]] = []
    n = {"r": 0}

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "Answer these meta-questions" in c:
            n["r"] += 1
            # high lexical overlap (J >= 0.55) — same conclusion, reworded
            variants = [
                "the december schedule is not officially announced yet",
                "the december schedule is not announced officially yet for travelers",
                "officially the december schedule is not yet announced anywhere",
                "the december schedule is still not officially announced",
                "not officially announced yet: the december schedule",
            ]
            return json.dumps(
                {
                    "answers": "a",
                    "key_finding": "k",
                    "summary": "s",
                    "facts": [
                        {
                            "fact": variants[(n["r"] - 1) % len(variants)],
                            "sources": [f"https://s{n['r']}.com/a"],
                        }
                    ],
                    "gaps": ["more"],
                    "sufficient": False,
                    "next_queries": [f"query {n['r']}"],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL"
        return "plain."

    from sherlock.tools.web_search import SearchEngine

    class FreshEngine(SearchEngine):
        def search(self, query, *, max_results=5):
            return [
                {
                    "title": f"{query} {i}",
                    "url": f"https://{abs(hash(query)) % 97}.com/{i}",
                    "content": f"c {i}",
                }
                for i in range(3)
            ]

        def fetch(self, url, *, raw=False, timeout=10.0):
            return {"url": url, "status": 200, "text": "page"}

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=FreshEngine(),
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._run_deep_research(agent._ensure_conversation().id, "the topic", 1, "drNV")
    docs_ev = next(d for (t, d) in events if t == "deep_research.documents")
    rounds = [d for (t, d) in events if t == "deep_research.round"]
    assert docs_ev["stop_reason"] == "converged_no_new_facts", docs_ev["stop_reason"]
    assert len(rounds) <= 4, f"restated conclusions must not sustain the loop: {len(rounds)}"
    # nothing is lost: very-close restatements (J>=0.8) MERGE with source
    # union (corroboration accumulates); the rest stay as separate facts.
    assert rounds[-1]["facts_total"] >= 1
