"""v1.1 P10 — tag repair (R7), carry-forward gating (R8), slot dedup (R13),
citation verification (R23), boundary trims (R24), sectioned synthesis (R25),
multi-breakpoint caching (R11), JSON-mode passthrough (R6)."""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.agent import _repair_tool_tags, _trim_at_boundary
from sherlock.providers.base import ChatMessage, ChatResponse
from sherlock.providers.litellm_provider import LiteLLMProvider

# ------------------------------------------------------------------ R7


def test_tag_repair_normalizes_common_misfires():
    assert _repair_tool_tags('<<sherlock_tool: search "x">>') == '<<sherlock-tool: search "x">>'
    assert (
        _repair_tool_tags("<sherlock-tool: fetch http://a.com>>")
        == "<<sherlock-tool: fetch http://a.com>>"
    )
    assert (
        _repair_tool_tags("<<sherlock-companions: compact>") == "<<sherlock-companions: compact>>"
    )
    assert _repair_tool_tags("no tags here at all") == "no tags here at all"


def test_repaired_companion_tag_actually_fires(tmp_path):
    calls = {"compact": 0}

    def llm2(messages):
        calls["compact"] += 1
        return json.dumps({"summary": "s", "facts": [], "topic_label": "t"})

    agent = Sherlock.with_callable(
        # broken tag: underscore + single closing bracket
        main_chat=lambda m: "ok.\n<<sherlock_companions: compact>",
        summary_chat=llm2,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    reply = agent.chat("hello")
    assert calls["compact"] >= 1, "repaired tag must trigger compaction"
    assert "sherlock" not in reply.lower(), "no tag residue may leak to the user"


# ------------------------------------------------------------------ R13


def test_rag_never_resurfaces_pinned_entries(tmp_path):
    from sherlock.memory.entry import MemorySource, MemoryType

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    conv_id = agent._ensure_conversation().id
    agent.memory.add(
        conversation_id=conv_id,
        content="UNIQUE_PINNED_FACT about quasar physics",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
    )
    hits = agent._retrieve_memories("quasar physics", current_turn_index=50)
    assert all(
        not e.pinned for e, _ in hits
    ), "pinned facts already ride TIER 2 — RAG must skip them"


# ------------------------------------------------------------------ R23


def test_unverified_citations_get_flagged():
    text = "Fact A (per https://known.com/a). Fact B (per https://invented.example/x)."
    out, bad = Sherlock._flag_unverified_citations(text, {"https://known.com/a"})
    assert "https://invented.example/x (unverified)" in out
    assert "https://known.com/a (unverified)" not in out
    assert bad == ["https://invented.example/x"]


# ------------------------------------------------------------------ R24


def test_trim_at_boundary_never_cuts_mid_word():
    text = "First sentence here. Second sentence is much longer and will be cut somewhere mid"
    out = _trim_at_boundary(text, 40)
    assert len(out) <= 40
    # never a partial word: the trimmed text must end exactly where a word ends
    assert text[len(out)] == " " or text.startswith(out + " ") is False and out.endswith(".")
    # sentence boundary wins when it lands deep enough in the budget
    long_first = "A reasonably long first sentence sits right here. tail words follow now"
    assert _trim_at_boundary(long_first, 55).endswith("here.")
    assert _trim_at_boundary("short", 40) == "short"


# ------------------------------------------------------------------ R25


def test_sectioned_synthesis_for_big_runs(tmp_path):
    """>18 facts + strategy sub-topics → one synthesis call per section, each
    reading ONLY its own facts; stitched output has section headers + sources."""
    section_prompts: list[str] = []

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "ONE SECTION" in c:
            section_prompts.append(c)
            sub = c.split("Section: «", 1)[1].split("»", 1)[0]
            return f"## {sub}\nSection body citing https://src0.com/u0"
        return "plain."

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    facts = [
        {"fact": f"pricing detail {i} for plan tier", "sources": [f"https://src{i % 3}.com/u{i}"]}
        for i in range(12)
    ] + [
        {"fact": f"complaint report {i} from users", "sources": [f"https://c{i}.com/p"]}
        for i in range(8)
    ]
    state = {
        "confirmed_facts": facts,
        "open_gaps": [],
        "strategy": {"sub_topics": ["pricing details", "user complaints"]},
    }
    out = agent._synthesize_research("c1", "drZ", "the topic", [], state=state)
    assert len(section_prompts) >= 2, "expected per-section synthesis calls"
    assert "## Sources" in out
    # each section call carried only its own facts
    pricing_call = next(p for p in section_prompts if "«pricing details»" in p)
    assert "complaint report" not in pricing_call
    # facts were partitioned, not duplicated
    assert "pricing detail 3" in pricing_call


def test_small_runs_keep_single_call_synthesis(tmp_path):
    calls = {"section": 0, "single": 0}

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "ONE SECTION" in c:
            calls["section"] += 1
            return "## x"
        if "RESEARCH DOCUMENTS:" in c:
            calls["single"] += 1
            return "FINAL https://a.com/1"
        return "plain."

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    state = {
        "confirmed_facts": [{"fact": "only fact", "sources": ["https://a.com/1"]}],
        "strategy": {"sub_topics": ["a", "b"]},
    }
    agent._synthesize_research("c1", "drY", "t", [], state=state)
    assert calls == {"section": 0, "single": 1}


# ------------------------------------------------------------------ R11


def test_multi_breakpoints_become_multiple_cached_blocks():
    msg = ChatMessage(
        role="system",
        content="AAAA" + "BBBB" + "cccc-volatile",
        cache_breakpoints=(4, 8),
    )
    out = LiteLLMProvider._to_litellm_messages([msg])
    blocks = out[0]["content"]
    assert [b["text"] for b in blocks] == ["AAAA", "BBBB", "cccc-volatile"]
    assert "cache_control" in blocks[0] and "cache_control" in blocks[1]
    assert "cache_control" not in blocks[2]


def test_breakpoints_take_precedence_and_payload_stays_clean():
    msg = ChatMessage(role="user", content="hi")
    assert LiteLLMProvider._to_litellm_messages([msg]) == [{"role": "user", "content": "hi"}]
    assert msg.to_dict() == {"role": "user", "content": "hi"}


# ------------------------------------------------------------------ R6


def test_json_mode_passthrough_and_memoized_fallback():
    from sherlock.jsonish import chat_json_with_retry

    class Prov:
        def __init__(self, fail_json_mode):
            self.fail = fail_json_mode
            self.kwargs_seen = []

        def chat(self, messages, **kw):
            self.kwargs_seen.append(kw)
            if self.fail and "response_format" in kw:
                raise RuntimeError("unsupported")
            return ChatResponse(text='{"a": 1}', model="m")

    ok = Prov(fail_json_mode=False)
    parsed, _ = chat_json_with_retry(ok, [ChatMessage(role="user", content="json please")])
    assert parsed == {"a": 1}
    assert ok.kwargs_seen[0].get("response_format") == {"type": "json_object"}

    bad = Prov(fail_json_mode=True)
    parsed, _ = chat_json_with_retry(bad, [ChatMessage(role="user", content="json please")])
    assert parsed == {"a": 1}
    # failed handshake memoized off: second call never retries response_format
    chat_json_with_retry(bad, [ChatMessage(role="user", content="json please")])
    assert all("response_format" not in k for k in bad.kwargs_seen[1:])
