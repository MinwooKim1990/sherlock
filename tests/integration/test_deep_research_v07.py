"""v0.7 Phase 3 — the approval-gated `deep_research` tool.

Operates the feature end-to-end with scripted LLM-1/LLM-3 callables and a
fake counting search engine:

* conversational approval (ask → yes runs, other cancels) — NEVER auto-runs
* programmatic approver callback (False blocks, True runs)
* the ≤20-round deep loop: per-round session documents, round 1–2 meta-Qs from
  LLM-1 (fixed) + round 3+ from LLM-3 (generated), cap honoured, cited synthesis
* mid-research input queue: enqueue while running, drained + folded at the next
  round boundary
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.memory.entry import MemoryType
from sherlock.tools.web_search import SearchEngine


class CountingEngine(SearchEngine):
    def __init__(self):
        self.calls: list[tuple[str, int]] = []
        self.fetches: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        self.calls.append((query, max_results))
        return [
            {
                "title": f"{query} #{i}",
                "url": f"https://ex.com/{abs(hash((query,i)))%1000}",
                "content": f"about {query} {i}",
            }
            for i in range(max_results)
        ]

    def fetch(self, url: str, *, raw: bool = False, timeout: float = 10.0) -> dict:
        self.fetches.append(url)
        return {"url": url, "status": 200, "text": f"full page text for {url}"}


_FACT_SEQ = {"n": 0}


def _meta_json(sufficient: bool, nxt: list[str]) -> str:
    # v1.0 C3: the loop converges when rounds stop ADDING facts, so a scripted
    # run that should keep going must yield a distinct fact per round (letter
    # suffix — numbers would trip the contradiction heuristic).
    _FACT_SEQ["n"] += 1
    n = _FACT_SEQ["n"]
    tag = ""
    while n:
        n, r = divmod(n - 1, 26)
        tag = chr(97 + r) + tag
    return json.dumps(
        {
            "answers": "key facts with https://ex.com/1",
            "key_finding": "a finding",
            "summary": "made progress",
            "facts": [{"fact": f"round finding {tag}", "sources": [f"https://ex.com/{tag}"]}],
            "sufficient": sufficient,
            "next_queries": nxt,
        }
    )


def _make_main(prompts_log: list[str], *, sufficient_at: int = 1, trigger: str = "RESEARCHME"):
    """LLM-1 callable that branches on prompt markers. Counts meta-Q&A calls so
    it can declare `sufficient` at a chosen round."""
    state = {"meta": 0}

    def main(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = last.get("content", "")
        prompts_log.append(content)
        # Order matters: internal-prompt markers are checked BEFORE the trigger
        # word, because v0.7.1 embeds the user's request (which contains the
        # trigger) into the meta-Q&A + synthesis prompts for language-matching.
        if "Answer these meta-questions" in content:
            state["meta"] += 1
            n = state["meta"]
            if n >= sufficient_at:
                return _meta_json(True, [])
            return _meta_json(False, [f"follow-up {n}"])
        if "RESEARCH DOCUMENTS:" in content:  # the synthesis prompt
            return "FINAL: synthesised answer citing https://ex.com/1 and https://ex.com/2."
        if trigger in content:
            return 'Sure — I can dig in.\n<<sherlock-tool: deep_research "the topic">>'
        return "plain reply."

    return main


def _make_llm3(meta_calls: list[str]):
    def llm3(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = last.get("content", "")
        if "META-COGNITION QUESTIONS" in content:
            meta_calls.append(content)
            return json.dumps(["deep angle A?", "deep angle B?", "deep angle C?"])
        # inference path (not exercised here)
        return '{"hypotheses": [], "freshness_required": [], "confidence_overall": 0.5}'

    return llm3


# --------------------------------------------------------------------------
# Approval gate
# --------------------------------------------------------------------------


def test_proposal_requires_approval_then_runs_on_yes(tmp_path):
    eng = CountingEngine()
    prompts: list[str] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main(prompts, sufficient_at=1),
        inference_chat=_make_llm3([]),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    # Turn 1: LLM-1 proposes deep research → approval ASK, nothing runs.
    reply1 = agent.chat("please RESEARCHME thoroughly")
    assert agent.pending_deep_research is not None
    assert "yes" in reply1.lower() or "approve" in reply1.lower() or "run it" in reply1.lower()
    assert eng.calls == [], "deep research ran WITHOUT approval"

    # Turn 2: affirmative → it runs, returns the synthesis.
    reply2 = agent.chat("yes please")
    assert agent.pending_deep_research is None
    assert eng.calls, "approved deep research never searched"
    assert reply2.startswith("FINAL:")
    # session documents were written
    docs = [
        m
        for m in agent._memory.list(conversation_id=agent.conversation_id)
        if m.type == MemoryType.DEEP_RESEARCH
    ]
    assert docs, "no DEEP_RESEARCH session documents were written"


def test_proposal_cancelled_by_non_affirmative(tmp_path):
    eng = CountingEngine()
    prompts: list[str] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main(prompts, sufficient_at=1),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    agent.chat("please RESEARCHME thoroughly")
    assert agent.pending_deep_research is not None
    # A non-affirmative cancels the pending request and proceeds normally.
    reply = agent.chat("actually, what's the weather")
    assert agent.pending_deep_research is None
    assert eng.calls == [], "deep research ran after a non-affirmative"
    assert reply.strip() == "plain reply."


def test_approver_false_blocks_true_runs(tmp_path):
    # Approver returns False → blocked, never searches.
    eng1 = CountingEngine()
    agent_block = Sherlock.with_callable(
        main_chat=_make_main([], sufficient_at=1),
        system_prompt="…",
        storage_dir=tmp_path / "a",
        main_search_engine=eng1,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: False,
    )
    r = agent_block.chat("please RESEARCHME thoroughly")
    assert eng1.calls == [], "approver=False still ran deep research"
    assert agent_block.pending_deep_research is None
    assert "declined" in r.lower()

    # Approver returns True → runs inline immediately (no UI, no sink).
    eng2 = CountingEngine()
    agent_run = Sherlock.with_callable(
        main_chat=_make_main([], sufficient_at=1),
        system_prompt="…",
        storage_dir=tmp_path / "b",
        main_search_engine=eng2,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,
    )
    r2 = agent_run.chat("please RESEARCHME thoroughly")
    assert eng2.calls, "approver=True did not run deep research"
    assert r2.startswith("FINAL:")


# --------------------------------------------------------------------------
# The deep loop: docs, meta-question source, cap, synthesis
# --------------------------------------------------------------------------


def _doc_bodies(agent) -> list[dict]:
    out = []
    for m in agent._memory.list(conversation_id=agent.conversation_id):
        if m.type != MemoryType.DEEP_RESEARCH:
            continue
        try:
            out.append(json.loads((m.content or "").split("\n", 1)[1]))
        except Exception:
            pass
    return out


def test_deep_loop_rounds_docs_and_meta_source(tmp_path):
    eng = CountingEngine()
    prompts: list[str] = []
    llm3_meta: list[str] = []
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main(prompts, sufficient_at=4),  # sufficient at round 4
        inference_chat=_make_llm3(llm3_meta),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,  # auto-approve
    )
    # A sink makes background=True → research runs in the worker; wait for it.
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat("please RESEARCHME thoroughly")
    agent.wait_for_background(timeout=15)

    rounds = [d for (t, d) in events if t == "deep_research.round"]
    assert len(rounds) == 4, f"expected 4 rounds, got {len(rounds)}"
    # rounds 1-2 are LLM-1 fixed meta; round 3+ are LLM-3 generated.
    assert [r["meta_source"] for r in rounds] == [
        "llm1-fixed",
        "llm1-fixed",
        "llm3-generated",
        "llm3-generated",
    ]
    assert len(llm3_meta) == 2, "LLM-3 should have generated meta-Qs for rounds 3 and 4"
    # a DEEP_RESEARCH doc per round + a final synthesis doc
    bodies = _doc_bodies(agent)
    round_docs = [b for b in bodies if not b.get("final")]
    final_docs = [b for b in bodies if b.get("final")]
    assert len(round_docs) == 4, f"expected 4 round docs, got {len(round_docs)}"
    assert len(final_docs) == 1, "expected exactly one final synthesis doc"
    # done event carried the synthesis
    done = [d for (t, d) in events if t == "deep_research.done"]
    assert done and done[-1]["answer"].startswith("FINAL:")
    # v0.7.1: each round event surfaces the queries, the meta-questions, the
    # answer, and the new-source count so the UI can show what's happening.
    assert all(isinstance(r.get("meta_questions"), list) and r["meta_questions"] for r in rounds)
    assert all("answers" in r and "new_sources" in r and r.get("queries") for r in rounds)


# --------------------------------------------------------------------------
# v0.7.1 — convergence stop + language-matching
# --------------------------------------------------------------------------


class StaticEngine(SearchEngine):
    """Returns the SAME urls every round → no new sources after round 1."""

    def __init__(self):
        self.calls = []

    def search(self, query, *, max_results=5):
        self.calls.append((query, max_results))
        return [
            {"title": f"t{i}", "url": f"https://same/{i}", "content": "c"}
            for i in range(max_results)
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "text": "p"}


def test_deep_loop_converges_when_no_new_sources(tmp_path):
    """Even if the model ALWAYS says 'need more', the loop stops once searches
    stop turning up new sources — so 20 rounds means genuine progress, not
    padding."""
    eng = StaticEngine()
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main([], sufficient_at=999),  # never self-satisfied
        inference_chat=_make_llm3([]),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,
    )
    agent.config.search.deep_research_max_rounds = 20
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat("please RESEARCHME thoroughly")
    agent.wait_for_background(timeout=15)

    rounds = [d for (t, d) in events if t == "deep_research.round"]
    synth = [d for (t, d) in events if t == "deep_research.synthesizing"]
    assert len(rounds) == 3, f"should converge at round 3, ran {len(rounds)}"
    assert synth and synth[-1]["stop_reason"] == "converged_no_new_sources"


def test_synthesis_prompt_carries_user_language(tmp_path):
    """The user's original (Korean) request reaches the synthesis prompt with a
    'same language' instruction so the answer isn't forced to English."""
    eng = CountingEngine()
    prompts: list[str] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main(prompts, sufficient_at=1),
        inference_chat=_make_llm3([]),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,
    )
    agent.chat("RESEARCHME 삼성 공급망을 한국어로 자세히 조사해줘")
    # The synthesis prompt embeds the user's request + a language instruction.
    synth_prompts = [p for p in prompts if "RESEARCH DOCUMENTS:" in p]
    assert synth_prompts, "synthesis prompt was never issued"
    assert any("한국어로" in p and "SAME language" in p for p in synth_prompts)


def test_deep_loop_respects_round_cap(tmp_path):
    eng = CountingEngine()
    prompts: list[str] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main(prompts, sufficient_at=999),  # never satisfied
        inference_chat=_make_llm3([]),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,
    )
    agent.config.search.deep_research_max_rounds = 3  # cap low
    events: list[tuple[str, dict]] = []
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat("please RESEARCHME thoroughly")
    agent.wait_for_background(timeout=15)
    rounds = [d for (t, d) in events if t == "deep_research.round"]
    assert len(rounds) == 3, f"cap=3 not honoured, ran {len(rounds)} rounds"


def test_deep_loop_hard_ceiling_is_twenty(tmp_path):
    eng = CountingEngine()
    agent = Sherlock.with_callable(
        main_chat=_make_main([], sufficient_at=999),
        inference_chat=_make_llm3([]),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,
    )
    agent.config.search.deep_research_max_rounds = 50  # above the hard ceiling
    events: list[tuple[str, dict]] = []
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat("please RESEARCHME thoroughly")
    agent.wait_for_background(timeout=30)
    rounds = [d for (t, d) in events if t == "deep_research.round"]
    assert len(rounds) == 20, f"hard ceiling is 20, ran {len(rounds)}"


# --------------------------------------------------------------------------
# Mid-research input queue
# --------------------------------------------------------------------------


def test_message_during_research_is_queued(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=_make_main([], sufficient_at=1),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=CountingEngine(),
        inference_search_engine="disabled",
    )
    # Simulate "research in flight".
    agent._deep_researching = True
    reply = agent.chat("also look at pricing")
    assert "queued" in reply.lower()
    assert agent._deep_research_inbox.qsize() == 1
    agent._deep_researching = False


def test_queued_input_folded_at_round_boundary(tmp_path):
    eng = CountingEngine()
    prompts: list[str] = []
    events: list[tuple[str, dict]] = []
    agent = Sherlock.with_callable(
        main_chat=_make_main(prompts, sufficient_at=1),  # would stop at round 1…
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    # Pre-seed the inbox so round 1's boundary drains it (…but the drain forces
    # at least one MORE round to fold it in, even though round 1 was sufficient).
    agent._deep_research_inbox.put("focus specifically on PRICING_FOCUS")
    agent._run_deep_research(agent._ensure_conversation().id, "the topic", 1, "drX")

    folded = [d for (t, d) in events if t == "deep_research.input_folded"]
    assert folded, "queued input was never drained/folded"
    # the folded text reached a later meta-Q&A prompt
    assert any("PRICING_FOCUS" in p for p in prompts), "folded input never reached LLM-1's prompt"
