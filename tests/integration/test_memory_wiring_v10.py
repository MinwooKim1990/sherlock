"""v1.0 P6 — agent-side wiring: retrieval-keyword query expansion + superseded
rows kept out of the prompt blocks."""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.memory.entry import MemorySource, MemoryType


def _llm2(messages):
    last = messages[-1].get("content", "")
    if "TRANSCRIPT" in last or "not parseable" in last:
        return json.dumps(
            {
                "summary": "user planning a trip",
                "facts": [],
                "topic_label": "trip",
                "retrieval_keywords": ["메밀", "알레르기"],
            }
        )
    return "{}"


def test_retrieval_keywords_expand_the_rag_query(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: compact>>",
        summary_chat=_llm2,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    # a fact only reachable via the keyword terms, not the next user input
    agent._ensure_conversation()
    agent.memory.add(
        conversation_id=agent.conversation_id,
        content="유진 메밀 알레르기 epipen 필요",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=0,
    )
    agent.chat("we are planning dinner")  # compaction stores the keywords
    hits = agent._retrieve_memories("what should we order", current_turn_index=99)
    assert any("메밀" in e.content for e, _ in hits), "keyword expansion never reached RAG"


def test_superseded_rows_never_reach_the_pinned_block(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    conv_id = agent._ensure_conversation().id
    old = agent.memory.add(
        conversation_id=conv_id,
        content="User lives in Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
    )
    new = agent.memory.add(
        conversation_id=conv_id,
        content="User lives in Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        dedup=False,
    )
    agent.memory.supersede(old.id, new.id)
    block = agent._format_pinned_block(conv_id)
    assert "Busan" in block
    assert "Tokyo" not in block, "superseded fact leaked into the pinned block"
