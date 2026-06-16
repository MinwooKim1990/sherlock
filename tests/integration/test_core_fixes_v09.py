"""v0.9 — regression tests for the verified-review fixes (deep research core).

Covers: refusal-aware approval, malformed-fact tolerance, round-1 overflow
backlog (no result ever dropped), honest search-engine-failure stop, state
digest caps, exact research-doc tag matching, queued-message persistence,
background failure surfacing, multilingual plan slicing, and lenient JSON
parsing.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.agent import _is_affirmative
from sherlock.inference.engine import _safe_parse_json
from sherlock.tools.web_search import SearchEngine


class WideEngine(SearchEngine):
    """Returns `n` distinct URLs per query (same set per query string)."""

    def __init__(self, n: int = 12):
        self.n = n
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        self.calls.append(query)
        return [
            {
                "title": f"{query} #{i}",
                "url": f"https://ex.com/{abs(hash(query)) % 97}/{i}",
                "content": f"about {query} {i}",
            }
            for i in range(self.n)
        ]

    def fetch(self, url: str, *, raw: bool = False, timeout: float = 10.0) -> dict:
        return {"url": url, "status": 200, "text": f"page {url}"}


class FailingEngine(SearchEngine):
    def __init__(self):
        self.calls = 0

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        self.calls += 1
        raise RuntimeError("rate limited")

    def fetch(self, url: str, *, raw: bool = False, timeout: float = 10.0) -> dict:
        return {"error": "down"}


def _scripted_main(*, facts_payload, next_queries, sufficient: bool = False):
    """LLM-1 returning a chosen `facts` JSON shape in round answers."""

    def main(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        c = last.get("content", "")
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "answers": "a",
                    "key_finding": "k",
                    "summary": "s",
                    "facts": facts_payload,
                    "gaps": "single string gap",  # malformed on purpose
                    "sufficient": sufficient,
                    "next_queries": next_queries,
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL synthesis https://ex.com/1"
        return "plain."

    return main


# ---------------------------------------------------------------- approval


def test_affirmative_rejects_refusals():
    assert _is_affirmative("yes")
    assert _is_affirmative("그래 해줘")
    assert _is_affirmative("go ahead")
    # explicit refusals that CONTAIN affirmative substrings must not approve
    assert not _is_affirmative("no, don't run the deep research")
    assert not _is_affirmative("아니 하지마")
    assert not _is_affirmative("하지 말아줘")
    assert not _is_affirmative("취소해줘")
    assert not _is_affirmative("nope")
    # merely mentioning the feature is not consent
    assert not _is_affirmative("what is deep research?")


# ------------------------------------------------- malformed model output


def test_string_facts_and_gaps_do_not_crash_run(tmp_path):
    """Small models return facts as bare strings — the run must finish and
    keep those facts (not crash mid-loop and lose everything)."""
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=_scripted_main(
            facts_payload=["Tokyo has cheap hostels", "Kyoto is quieter"],
            next_queries=[],
            sufficient=True,
        ),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=WideEngine(3),
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    answer = agent._run_deep_research(agent._ensure_conversation().id, "the topic", 1, "drA")
    assert "FINAL" in answer
    rounds = [d for (t, d) in events if t == "deep_research.round"]
    assert rounds and rounds[-1]["facts_total"] == 2


# ------------------------------------------------------- backlog (no loss)


def test_round1_overflow_goes_to_backlog_and_is_flushed(tmp_path):
    """A wide round-1 sweep finds more fragments than one round shows (8);
    the rest must be shown in later rounds, never silently dropped."""
    events: list[tuple[str, dict]] = []
    prompts: list[str] = []

    def main(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        c = last.get("content", "")
        prompts.append(c)
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "answers": "a",
                    "key_finding": "k",
                    "summary": "s",
                    "facts": [],
                    "gaps": [],
                    "sufficient": False,
                    "next_queries": [],  # no new searches → loop must flush backlog
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL https://ex.com/1"
        return "plain."

    eng = WideEngine(12)
    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._run_deep_research(agent._ensure_conversation().id, "wide topic", 1, "drB")

    rounds = [d for (t, d) in events if t == "deep_research.round"]
    assert rounds[0]["backlog"] > 0, "round 1 should overflow into the backlog"
    assert rounds[-1]["backlog"] == 0, "backlog must be fully flushed before stopping"
    # every discovered URL was eventually shown to LLM-1
    all_urls = {
        f"https://ex.com/{abs(hash(q)) % 97}/{i}" for q in set(eng.calls) for i in range(12)
    }
    shown_urls = {u for u in all_urls if any(u in p for p in prompts)}
    assert shown_urls == all_urls, f"dropped fragments: {all_urls - shown_urls}"


# --------------------------------------------------- engine failure honesty


def test_search_engine_failure_stops_honestly(tmp_path):
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=_scripted_main(facts_payload=[], next_queries=["retry query"]),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=FailingEngine(),
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    answer = agent._run_deep_research(agent._ensure_conversation().id, "the topic", 1, "drC")

    docs_ev = next(d for (t, d) in events if t == "deep_research.documents")
    assert docs_ev["stop_reason"] == "search_engine_error"
    assert "search engine" in answer.lower()
    assert "FINAL" not in answer, "must not fabricate a synthesis from zero material"


# ------------------------------------------------------------ state digest


def test_state_digest_caps_and_gaps_survive():
    facts = [{"fact": f"fact {i}", "sources": [f"https://a{i}.com"]} for i in range(30)]
    digest = Sherlock._state_digest({"confirmed_facts": facts, "open_gaps": ["gap A", "gap B"]})
    assert "(+10 more confirmed facts not shown)" in digest
    assert "OPEN GAPS" in digest and "gap A" in digest
    assert len(digest) <= 2300


# ------------------------------------------------------- doc tag matching


def test_research_doc_tags_match_exactly(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "plain.",
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=WideEngine(1),
        inference_search_engine="disabled",
    )
    conv_id = agent._ensure_conversation().id
    qa = {"answers": "a", "summary": "s", "key_finding": "k"}
    agent._write_research_doc(conv_id, "dr1", "t", 1, [], [], qa, [], "llm1-fixed", 1)
    agent._write_research_doc(conv_id, "dr10", "t", 1, [], [], qa, [], "llm1-fixed", 1)
    docs = agent._list_research_docs(conv_id, "dr1")
    assert len(docs) == 1, "dr1 must not collect dr10's documents"


# ---------------------------------------------- queued-message persistence


def test_queued_message_persists_ack_and_completes_turn(tmp_path):
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=lambda m: "plain.",
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=WideEngine(1),
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._deep_researching = True
    reply = agent.chat("also check pricing")
    agent._deep_researching = False
    assert "queued" in reply.lower()
    history = agent.messages()
    assert any(m.role == "assistant" and "queued" in m.content.lower() for m in history)
    assert any(t == "turn.completed" for (t, d) in events)


# ------------------------------------------------ background failure event


def test_background_failure_is_surfaced(tmp_path, monkeypatch):
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=lambda m: "plain.",
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=WideEngine(1),
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(agent, "_run_deep_research", boom)
    conv_id = agent._ensure_conversation().id
    agent._execute_deep_research(conv_id, "t", 1, background=True)
    agent.wait_for_background(timeout=10)

    assert agent.is_deep_researching is False
    assert any(t == "deep_research.failed" for (t, d) in events)
    done = next(d for (t, d) in events if t == "deep_research.done")
    assert "failed" in done["answer"].lower()
    assert any(
        m.role == "assistant" and "failed" in m.content.lower() for m in agent.messages()
    ), "failure must reach the transcript, not just stderr"


# ------------------------------------------------- multilingual plan slicing


def test_plan_search_slice_preserves_language_breadth(tmp_path):
    plan_payload = [{"lang": "ja", "keywords": f"日本 キーワード {i}"} for i in range(6)] + [
        {"lang": "ko", "keywords": "일본 여행"},
        {"lang": "en", "keywords": "japan travel"},
    ]

    def llm3(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        if "MULTILINGUAL web-search sweep" in last.get("content", ""):
            return json.dumps(plan_payload)
        return "{}"

    agent = Sherlock.with_callable(
        main_chat=lambda m: "plain.",
        inference_chat=llm3,
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=WideEngine(1),
        inference_search_engine="disabled",
    )
    plan = agent._inferer.plan_search(topic="일본 여행", max_queries=4)
    langs = {p["lang"] for p in plan}
    assert len(plan) <= 4
    assert {"ja", "ko", "en"} <= langs, f"slicing crowded out languages: {langs}"


def test_plan_search_fallback_is_honest(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "plain.",
        inference_chat=lambda m: "complete garbage, not json",
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=WideEngine(1),
        inference_search_engine="disabled",
    )
    plan = agent._inferer.plan_search(topic="일본 여행 명소", user_lang="ko")
    # no fabricated duplicate mislabelled "en" — one honest entry
    assert len(plan) == 1
    assert plan[0]["lang"] == "ko"
    assert "일본" in plan[0]["keywords"]


# ----------------------------------------------------------- JSON parsing


def test_safe_parse_json_trailing_commas_and_prose_brackets():
    assert _safe_parse_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}
    # bracketed prose before the real JSON array
    text = '[1] some citation says X.\n["question one?", "question two?"]'
    assert _safe_parse_json(text) == ["question one?", "question two?"]
    # object preferred over inner array for round answers
    obj = 'noise {"facts": ["f1"], "sufficient": false,} trailing'
    assert _safe_parse_json(obj) == {"facts": ["f1"], "sufficient": False}
