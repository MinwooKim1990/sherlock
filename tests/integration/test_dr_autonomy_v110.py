"""Phase 0 (v1.10) regression locks for the autonomy/approval/image work that
shipped this session WITHOUT tests:
  - `_PRESENTATION_GUIDE` injected into every DR synthesis + the v3 editor prompt
  - semantic approval fallback (`_approval_intent`) for natural go-aheads the
    fixed keyword list misses, and refusal short-circuit (no LLM call)
  - og:image threaded fetch → raw fragment → synthesis prompt
  - v3 editor shrink guard relaxed to 0.3 (compact table kept, gutted reverted)

Deterministic: scripted provider + fake engine, no network, no real model.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine

_STRATEGY = json.dumps(
    {
        "objective": "o",
        "sub_topics": ["Alpha events", "Beta events"],
        "scope": {"include": [], "exclude": []},
        "clarifying_questions": [],
    }
)


class _ImgEngine(SearchEngine):
    """Short snippets (forces a thin-round fetch) + every fetched page has an image."""

    IMG = "https://img.example.com/pic.jpg"

    def search(self, query, *, max_results=5):
        h = abs(hash(query)) % 100000  # fresh urls per query → later rounds have NEW urls to fetch
        return [
            {
                "title": "Alpha events",
                "url": f"https://ex.com/alpha{h}",
                "content": "short alpha events",
            },
            {
                "title": "Beta events",
                "url": f"https://ex.com/beta{h}",
                "content": "short beta events",
            },
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        # text routes to the "Alpha events" sub-topic bucket so the raw-recon synthesis re-reads it
        return {
            "url": url,
            "status": 200,
            "text": f"Alpha events full page about {url}",
            "image": self.IMG,
        }


def _dr_main(prompts, *, sufficient_at=1, editor_out=None):
    """Full scripted LLM-1 covering every DR prompt + the approval-intent prompt.
    Logs every user prompt it sees into `prompts`."""
    st = {"r": 0}

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
        if "waiting for the user's go-ahead" in c:  # _approval_intent
            return json.dumps({"approve": True})
        if "RESEARCH STRATEGY" in c:
            return _STRATEGY
        if "Answer these meta-questions" in c:
            st["r"] += 1
            suf = st["r"] >= sufficient_at
            return json.dumps(
                {
                    "facts": [{"fact": f"finding {st['r']}", "sources": ["https://ex.com/alpha"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": suf,
                    "next_queries": [] if suf else ["more"],
                }
            )
        if "fact-checking a research report" in c:  # v3 editor
            return (
                editor_out
                if editor_out is not None
                else ("## Report\nverified body with https://ex.com/alpha")
            )
        if "ONE SECTION" in c:  # sectioned / raw-reconstruction synthesis
            return "## Section\nbody citing https://ex.com/alpha"
        if "RESEARCH DOCUMENTS:" in c:  # single-call synthesis
            return "## Report\nbody citing https://ex.com/alpha"
        if "RESEARCHME" in c:  # proposal trigger
            return 'Sure.\n<<sherlock-tool: deep_research "the topic">>'
        return "plain reply."

    return main


def _agent(main, tmp_path, engine=None):
    a = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=engine or _ImgEngine(),
        inference_search_engine="disabled",
    )
    a.config.search.deep_research_max_rounds = 4
    return a


# --------------------------------------------------------------------------
# _PRESENTATION_GUIDE present in every DR LLM-1 prompt (autonomy lock)
# --------------------------------------------------------------------------
def test_presentation_guide_in_synthesis_and_editor(tmp_path):
    prompts: list[str] = []
    agent = _agent(_dr_main(prompts, sufficient_at=1), tmp_path)
    agent._run_deep_research(agent._ensure_conversation().id, "the topic", 1, "drG")
    synth = [p for p in prompts if "RESEARCH DOCUMENTS:" in p or "ONE SECTION" in p]
    editor = [p for p in prompts if "fact-checking a research report" in p]
    assert synth and all(
        "PRESENTATION — your call" in p for p in synth
    ), "guide missing in synthesis"
    assert editor and all(
        "PRESENTATION — your call" in p for p in editor
    ), "guide missing in editor"


# --------------------------------------------------------------------------
# Semantic approval (_approval_intent) — natural go-ahead the keyword list misses
# --------------------------------------------------------------------------
def test_semantic_approval_runs_on_natural_goahead(tmp_path):
    prompts: list[str] = []
    eng = _ImgEngine()
    agent = _agent(_dr_main(prompts, sufficient_at=1), tmp_path, engine=eng)
    agent.chat("please RESEARCHME thoroughly")
    assert agent.pending_deep_research is not None
    # "research that one thoroughly" is NOT in the affirmative keyword list and is
    # not a refusal → the LLM judge (_approval_intent) must decide → it runs.
    reply = agent.chat("research that one thoroughly")
    assert any(
        "waiting for the user's go-ahead" in p for p in prompts
    ), "approval-intent never fired"
    assert agent.pending_deep_research is None, "research did not run after semantic approval"
    assert reply.lstrip().startswith("##"), f"expected a synthesized report, got: {reply[:80]!r}"


def test_semantic_approval_refusal_short_circuits(tmp_path):
    prompts: list[str] = []
    agent = _agent(_dr_main(prompts, sufficient_at=1), tmp_path)
    agent.chat("please RESEARCHME thoroughly")
    prompts.clear()
    agent.chat("아니 하지마")  # refusal → must NOT call the LLM judge, must not run
    assert not any(
        "waiting for the user's go-ahead" in p for p in prompts
    ), "judge fired on refusal"


# --------------------------------------------------------------------------
# og:image: fetched-page image reaches a raw fragment + the synthesis prompt
# --------------------------------------------------------------------------
def test_og_image_reaches_synthesis(tmp_path):
    prompts: list[str] = []
    # sufficient_at=2 → round 1 not sufficient → round 2 runs → thin round fetches
    # → og:image captured → surfaced in the raw-reconstruction synthesis prompt.
    agent = _agent(_dr_main(prompts, sufficient_at=2), tmp_path)
    agent._run_deep_research(agent._ensure_conversation().id, "the topic", 1, "drI")
    assert any(_ImgEngine.IMG in p for p in prompts), "fetched og:image never reached a DR prompt"


# --------------------------------------------------------------------------
# v3 editor shrink guard relaxed to 0.3
# --------------------------------------------------------------------------
def test_editor_keeps_compact_table_reverts_gutted(tmp_path):
    report = "Y" * 100
    state = {"confirmed_facts": []}
    # editor returns 40% length → kept (a table is allowed to compress)
    a_keep = _agent(_dr_main([], editor_out="K" * 40), tmp_path)
    assert a_keep._verify_research_report(report, state, "topic", "r") == "K" * 40
    # editor returns 20% length → treated as refusal/truncation → original kept
    a_rev = _agent(_dr_main([], editor_out="R" * 20), tmp_path)
    assert a_rev._verify_research_report(report, state, "topic", "r") == report
