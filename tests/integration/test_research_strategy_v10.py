"""v1.0 P5 — research strategy (C0) + fragment reassembly (C1-C5).

The strategy is a GUIDELINE: it seeds open gaps, sharpens the plan, and rides
the approval ask (with up to 2 clarifying questions) — but a model that can't
produce one gets exactly the legacy behavior.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.agent import (
    _diversify_fragments,
    _fact_tokens,
    _looks_contradictory,
    _select_relevant_excerpt,
    _token_jaccard,
)
from sherlock.tools.web_search import SearchEngine


class MiniEngine(SearchEngine):
    def __init__(self):
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        self.calls.append(query)
        return [
            {
                "title": f"{query} #{i}",
                "url": f"https://ex.com/{abs(hash(query)) % 97}/{i}",
                "content": f"about {query} {i}",
            }
            for i in range(3)
        ]

    def fetch(self, url: str, *, raw: bool = False, timeout: float = 10.0) -> dict:
        return {"url": url, "status": 200, "text": f"page {url}"}


STRATEGY_JSON = json.dumps(
    {
        "objective": "find the real angle",
        "sub_topics": ["pricing history", "user complaints", "alternatives"],
        "scope": {"include": ["recent"], "exclude": ["enterprise plans"]},
        "clarifying_questions": ["Which region matters most?"],
    }
)


def _make_main(prompts, *, strategy_reply=STRATEGY_JSON, trigger="RESEARCHME"):
    def main(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        c = last.get("content", "")
        prompts.append(c)
        # internal-prompt markers BEFORE the trigger (lang_hint embeds user text)
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
            return "FINAL https://ex.com/1"
        if "not parseable" in c:
            return "still not json"
        if trigger in c:
            return 'Sure.\n<<sherlock-tool: deep_research "the topic">>'
        return "plain reply."

    return main


def _agent(tmp_path, prompts, **kw):
    return Sherlock.with_callable(
        main_chat=_make_main(prompts, **{k: v for k, v in kw.items() if k == "strategy_reply"}),
        system_prompt="…",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=MiniEngine(),
        inference_search_engine="disabled",
        **{k: v for k, v in kw.items() if k != "strategy_reply"},
    )


# ------------------------------------------------------------------ C0


def test_proposal_carries_strategy_and_questions(tmp_path):
    prompts: list[str] = []
    events: list[tuple[str, dict]] = []
    agent = _agent(tmp_path, prompts)
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    reply = agent.chat("please RESEARCHME")
    # the ask keeps the frozen wording AND shows the strategy
    assert "run it" in reply.lower() or "yes" in reply.lower()
    assert "pricing history" in reply
    assert "Which region matters most?" in reply
    pending = agent.pending_deep_research
    assert pending and pending["strategy"]["objective"] == "find the real angle"
    assert any(t == "deep_research.strategy" for (t, d) in events)


def test_clarification_answer_folds_and_reasks_once(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts)
    agent.chat("please RESEARCHME")
    # an ANSWER (not refusal, not affirmative) → folded + re-asked, NOT cancelled
    reply = agent.chat("주로 유럽 지역이 중요해")
    assert "yes" in reply.lower()
    assert agent.pending_deep_research is not None
    assert "[Clarification]" in agent.pending_deep_research["user_text"]
    # an explicit yes now runs it
    agent.chat("yes")
    agent.wait_for_background(timeout=15)
    assert agent.pending_deep_research is None
    joined = "\n".join(prompts)
    assert "유럽" in joined, "folded clarification must reach the research prompts"


def test_second_nonaffirmative_cancels(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts)
    agent.chat("please RESEARCHME")
    agent.chat("유럽 위주로")  # fold + re-ask
    assert agent.pending_deep_research is not None
    agent.chat("갑자기 딴 얘긴데 오늘 날씨 어때")  # still not a yes → cancel
    assert agent.pending_deep_research is None


def test_refusal_still_cancels_immediately(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts)
    agent.chat("please RESEARCHME")
    agent.chat("아니 하지마")
    assert agent.pending_deep_research is None


def test_strategy_seeds_open_gaps_and_round_prompts(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts)
    agent.chat("please RESEARCHME")
    agent.chat("yes")
    agent.wait_for_background(timeout=15)
    round_prompts = [p for p in prompts if "Answer these meta-questions" in p]
    assert round_prompts, "research never ran"
    # sub-topics ride the digest (open gaps) + the guideline line
    assert any("user complaints" in p for p in round_prompts)
    assert any("guideline, not a cage" in p for p in round_prompts)
    assert any("Out of scope: enterprise plans" in p for p in round_prompts)


def test_garbage_strategy_falls_back_to_legacy_behavior(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts, strategy_reply="I have no idea")
    reply = agent.chat("please RESEARCHME")
    assert "run it" in reply.lower() or "yes" in reply.lower()
    assert "📋" not in reply  # no fabricated strategy
    agent.chat("yes")
    agent.wait_for_background(timeout=15)
    round_prompts = [p for p in prompts if "Answer these meta-questions" in p]
    assert round_prompts and all("guideline, not a cage" not in p for p in round_prompts)


def test_strategy_killswitch(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts)
    agent.config.search.deep_research_strategy = False
    agent.chat("please RESEARCHME")
    assert not any("RESEARCH STRATEGY" in p for p in prompts), "strategy call must be skipped"


def test_affirmative_with_extra_context_folds_it(tmp_path):
    prompts: list[str] = []
    agent = _agent(tmp_path, prompts, strategy_reply="garbage")  # no questions
    agent.chat("please RESEARCHME")
    agent.chat("yes — focus on the budget angle please")
    agent.wait_for_background(timeout=15)
    joined = "\n".join(prompts)
    assert "budget angle" in joined


# ------------------------------------------------------- C1/C2/C4/C5 units


def test_select_relevant_excerpt_finds_buried_fragment():
    text = ("filler paragraph. " * 40 + "\n\n") * 6
    text += "\n\nA buried comment says the venue moved to Shibuya in 2024.\n\n"
    text += ("more filler. " * 40 + "\n\n") * 6
    out = _select_relevant_excerpt(text, ["venue shibuya"], budget=600)
    assert "Shibuya" in out
    assert len(out) <= 650


def test_select_relevant_excerpt_falls_back_to_head():
    text = "x" * 5000
    assert _select_relevant_excerpt(text, ["nomatch"], budget=100) == "x" * 100


def test_near_duplicate_facts_merge_sources():
    a = _fact_tokens("Tokyo hostels are cheap in winter")
    b = _fact_tokens("hostels in tokyo are cheap in winter")
    assert _token_jaccard(a, b) >= 0.8


def test_contradiction_detection():
    assert _looks_contradictory("the venue is open on mondays", "the venue is not open on mondays")
    assert _looks_contradictory("raised $40M in funding", "raised $25M in funding")
    assert not _looks_contradictory("hostels are cheap", "ramen shops are everywhere")


def test_diversify_fragments_leads_with_type_diversity():
    hits = [
        {"url": "https://news.bbc.co.uk/a", "_q": 0, "_rank": 0},
        {"url": "https://news.bbc.co.uk/b", "_q": 0, "_rank": 1},
        {"url": "https://reddit.com/r/x", "_q": 1, "_rank": 0},
        {"url": "https://example.gov/page", "_q": 1, "_rank": 1},
    ]
    out = _diversify_fragments(hits)
    from sherlock.agent import _source_type

    first_three_types = {_source_type(h["url"]) for h in out[:3]}
    assert len(first_three_types) == 3, f"expected 3 distinct types first, got {first_three_types}"
    assert len(out) == 4  # nothing dropped


def test_disputed_facts_reach_digest_and_synthesis(tmp_path):
    prompts: list[str] = []

    def main(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        c = last.get("content", "")
        prompts.append(c)
        if "Answer these meta-questions" in c:
            n = sum(1 for p in prompts if "Answer these meta-questions" in p)
            fact = (
                {"fact": "the festival runs for 10 days", "sources": ["https://a.com/1"]}
                if n == 1
                else {"fact": "the festival runs for 3 days", "sources": ["https://b.com/2"]}
            )
            return json.dumps(
                {
                    "answers": "a",
                    "key_finding": "k",
                    "summary": "s",
                    "facts": [fact],
                    "gaps": ["more"],
                    "sufficient": n >= 2,
                    "next_queries": [] if n >= 2 else ["follow"],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL both sides https://a.com/1 https://b.com/2"
        return "plain."

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="…",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=MiniEngine(),
        inference_search_engine="disabled",
    )
    agent._run_deep_research(agent._ensure_conversation().id, "festival duration", 1, "drD")
    synth = next(p for p in prompts if "RESEARCH DOCUMENTS:" in p)
    assert "[disputed — sources conflict]" in synth
    assert "present BOTH sides" in synth
