"""v1.5 Stage 4 — recursive inference notebook (deep-research mirror).

Bounded (≤ notebook_max_rounds, converge-stop, yields to deep research),
grounded (every kept step cites a verbatim corpus quote — ungrounded steps
discarded), anchored (only high-value open questions enter), background-only.
OFF by default → slot byte-identical; deep-research code path untouched.
"""

from __future__ import annotations

import json

import pytest

from sherlock import Sherlock
from sherlock.agent import _notebook_step_grounded
from sherlock.inference.engine import InferenceEngine
from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.providers.fake import FakeProvider
from sherlock.storage import Storage


def _agent(tmp_path, name, **kw):
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


def _fake_deepen(outputs):
    it = iter(outputs)

    def fn(**kw):
        try:
            return next(it)
        except StopIteration:
            return {}

    return fn


INFER = {
    "really_asking": "does the user actually need to buy the pass early",
    "confidence_overall": 0.5,
    "hypotheses": [{"intent": "wants flexibility"}],
    "anticipated_next": [],
    "implied_chain": ["a", "b"],
}
# corpus carries the verbatim phrase the grounded steps will quote
SEARCH = [{"title": "", "content": "the user said buy the pass early before the trip starts"}]


# ---------- grounding helper ------------------------------------------------
def test_step_grounded_requires_corpus_quote():
    cf = "the user said buy the pass early"
    assert _notebook_step_grounded({"evidence": "buy the pass"}, cf) is True
    assert _notebook_step_grounded({"evidence": "fabricated phrase"}, cf) is False
    assert _notebook_step_grounded({"evidence": ""}, cf) is False


# ---------- anchor ----------------------------------------------------------
def test_anchor_skips_when_no_open_question(tmp_path):
    a = _agent(tmp_path, "anc1", inference_notebook=True)
    flat = {
        "really_asking": "",
        "anticipated_next": [],
        "confidence_overall": 0.9,
        "hypotheses": [],
    }
    assert a._run_inference_notebook(a._ensure_conversation().id, 1, flat, []) is None


def test_anchor_skips_when_confident_and_no_chain(tmp_path):
    a = _agent(tmp_path, "anc2", inference_notebook=True)
    conf = {
        "really_asking": "x",
        "anticipated_next": [],
        "confidence_overall": 0.9,
        "implied_chain": [],
        "hypotheses": [],
    }
    assert a._run_inference_notebook(a._ensure_conversation().id, 1, conf, []) is None


# ---------- happy path: rounds, grounding, convergence ----------------------
def test_notebook_accumulates_grounded_steps_and_converges(tmp_path):
    a = _agent(tmp_path, "happy", inference_notebook=True)
    a._inferer.deepen_notebook = _fake_deepen(
        [
            {
                "steps": [
                    {"question": "q1", "answer": "a1", "evidence": "buy the pass"},
                    {"question": "qX", "answer": "aX", "evidence": "not in the corpus at all"},
                ],
                "conclusions": ["needs flexibility"],
                "open_questions": ["q2"],
                "converged": False,
            },
            {
                "steps": [{"question": "q2", "answer": "a2", "evidence": "before the trip"}],
                "conclusions": ["wants to keep the schedule open"],
                "open_questions": [],
                "converged": True,
            },
        ]
    )
    nb = a._run_inference_notebook(a._ensure_conversation().id, 1, INFER, SEARCH)
    assert nb is not None
    qs = [s["question"] for s in nb["raw"]]
    assert "q1" in qs and "q2" in qs
    assert "qX" not in qs  # ungrounded step discarded
    assert nb["conclusions"] == ["wants to keep the schedule open"]
    assert nb["rounds"] == 2


def test_notebook_stops_at_max_rounds(tmp_path):
    a = _agent(tmp_path, "cap", inference_notebook=True, notebook_max_rounds=2)
    never_converge = {
        "steps": [{"question": "q", "answer": "a", "evidence": "buy the pass"}],
        "conclusions": ["c"],
        "open_questions": ["more"],
        "converged": False,
    }
    # distinct question each round so it doesn't stop on a dry round
    a._inferer.deepen_notebook = _fake_deepen(
        [
            {
                **never_converge,
                "steps": [{"question": f"q{i}", "answer": "a", "evidence": "buy the pass"}],
            }
            for i in range(5)
        ]
    )
    nb = a._run_inference_notebook(a._ensure_conversation().id, 1, INFER, SEARCH)
    assert nb["rounds"] == 2  # capped


def test_notebook_yields_to_deep_research(tmp_path):
    a = _agent(tmp_path, "yield", inference_notebook=True)
    a._deep_researching = True
    a._inferer.deepen_notebook = _fake_deepen(
        [
            {
                "steps": [{"question": "q", "answer": "a", "evidence": "buy the pass"}],
                "conclusions": ["c"],
                "open_questions": [],
                "converged": True,
            }
        ]
    )
    assert a._run_inference_notebook(a._ensure_conversation().id, 1, INFER, SEARCH) is None


# ---------- hardening: init defaults + malformed deepen output --------------
def test_notebook_attrs_initialized(tmp_path):
    # AUDIT L2: the carry-over attrs exist after construction (not getattr-only).
    a = _agent(tmp_path, "init")
    assert a._pending_notebook is None
    assert a._slot_notebook is None


def test_deepen_notebook_non_list_fields_dont_char_splat(tmp_path):
    # AUDIT L1: a scalar string for conclusions/open_questions must NOT iterate
    # into single characters.
    import json as _json

    class _Embed:
        provider = "fake"
        model = "fake-embedding"
        api_key_env = None

    storage = Storage(tmp_path / "n.db")
    store = MemoryStore(
        engine=storage.engine,
        embedding_provider=build_embedding_provider(_Embed()),
        vector_path=tmp_path / "v",
    )
    reply = _json.dumps(
        {"steps": "notalist", "conclusions": "hi", "open_questions": "abc", "converged": False}
    )
    eng = InferenceEngine(provider=FakeProvider(canned_reply=reply), store=store)
    out = eng.deepen_notebook(
        open_questions=["q"], notebook_state={}, corpus="x", round_index=1, max_rounds=3
    )
    assert out["steps"] == []
    assert out["conclusions"] == []  # NOT ['h','i']
    assert out["open_questions"] == []  # NOT ['a','b','c']


# ---------- corpus: no self-talk (assistant turns excluded) -----------------
def test_corpus_excludes_assistant_turns(tmp_path):
    # Grounding on LLM-1's OWN prior reply would be self-talk → bias amplification.
    # The corpus must carry the user's words but NOT the assistant's claims.
    a = _agent(tmp_path, "corpus", inference_notebook=True)
    cid = a._ensure_conversation().id
    a._storage.add_message(cid, role="user", content="USERTOKENxyz buy the pass early")
    a._storage.add_message(cid, role="assistant", content="ASSTTOKENqrs that is a fact")
    corpus = a._notebook_corpus(cid, [])
    assert "USERTOKENxyz" in corpus
    assert "ASSTTOKENqrs" not in corpus


# ---------- render ----------------------------------------------------------
def test_render_notebook_block():
    nb = {
        "raw": [{"question": "q1", "answer": "a1", "evidence": "buy the pass"}],
        "conclusions": ["needs flexibility"],
        "rounds": 1,
    }
    block = Sherlock._render_notebook_block(nb)
    assert "INFERENCE NOTEBOOK" in block
    assert "RAW STEPS" in block and "q1" in block and "buy the pass" in block
    assert "CONCLUSIONS" in block and "needs flexibility" in block


# ---------- slot wiring -----------------------------------------------------
def test_slot_off_no_notebook_block(tmp_path):
    a = _agent(tmp_path, "slotoff")  # notebook off (default)
    a._pending_notebook = {
        "raw": [{"question": "q", "answer": "a", "evidence": "e"}],
        "conclusions": ["c"],
        "rounds": 1,
    }
    a.chat("hello")
    final = a.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "INFERENCE NOTEBOOK" not in final  # gated off → never rendered


def test_slot_on_pulls_notebook(tmp_path):
    a = _agent(tmp_path, "sloton", inference_notebook=True)
    a._pending_notebook = {
        "raw": [{"question": "buy early?", "answer": "maybe", "evidence": "buy the pass"}],
        "conclusions": ["wants flexibility"],
        "rounds": 1,
    }
    a.chat("so what do you think?")
    final = a.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "INFERENCE NOTEBOOK" in final
    assert "wants flexibility" in final


# ---------- async parity (external audit P1) --------------------------------
_INFER_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "intent": "wants flexibility",
                "probability": 0.6,
                "evidence": ['user said "buy the pass"'],
                "search_keywords": [],
                "reasoning_type": "pragmatic",
            }
        ],
        "implied_chain": ["a", "b"],
        "really_asking": "is it safe to wait on the pass",
        "anticipated_next": [],
        "tools_recommended": [],
        "freshness_required": [],
        "confidence_overall": 0.6,
        "evolution_signals": {},
    }
)


@pytest.mark.asyncio
async def test_achat_v15_parity(tmp_path):
    # AUDIT P1: native async achat() must fire the v1.5 LLM-3 upgrades + the v1.2
    # chain carry-forward + the notebook, exactly like sync chat().
    async def main(messages):
        return "here is the answer.\n<<sherlock-companions: infer>>"

    a = Sherlock.with_callable(
        main_chat=main,
        inference_chat=lambda m: _INFER_JSON,
        system_prompt="You are terse.",
        storage_dir=tmp_path / "asyncparity",
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        perception=True,
        evidence_grounding=True,
        premise_conflict=True,
        inference_notebook=True,
    )
    a._inferer.deepen_notebook = lambda **kw: {
        "steps": [{"question": "q1", "answer": "a1", "evidence": "buy the pass"}],
        "conclusions": ["wants flexibility"],
        "open_questions": [],
        "converged": True,
    }
    await a.achat("should I buy the pass early before the trip")
    # v1.2 implied-chain carry-forward now lands on the async path (was dropped):
    assert a._pending_inference_extras.get("really_asking")
    # v1.5 inference notebook now runs on the async path (was never produced):
    assert a._pending_notebook is not None
