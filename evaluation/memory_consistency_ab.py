"""Stage 3 live check — LLM-2 memory-consistency, OFF vs "code".

Pin a fact, then send a message that NEGATES it (a self-correction). The "code"
run should surface a MEMORY-CONSISTENCY cue in the LLM-1 slot and the reply
should RECONCILE (acknowledge the change / not silently contradict). The OFF run
gets no cue. Same gemini-flash-lite both sides.

Run:  .venv/bin/python evaluation/memory_consistency_ab.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock  # noqa: E402
from sherlock.memory.entry import MemorySource, MemoryType  # noqa: E402
from sherlock.providers.base import ChatResponse, TokenUsage  # noqa: E402

MODEL = "gemini-2.5-flash-lite"
OUT = Path("evaluation/memory_consistency_ab.json")
_CLIENT = None
_LOCK = threading.Lock()

# A clean negation MISMATCH (one side negates, the other doesn't) — the case the
# code-first check is built for. (Semantic contradictions where both sides carry
# a negation word are an honest false-negative of the heuristic, by design.)
PINNED = "나는 매운 음식을 진짜 잘 먹어."
MESSAGE = "사실 나 매운 거 잘 못 먹어. 안 매운 메뉴로 추천 좀 해줘."


def _raw_chat(prompt):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("gemini", model=MODEL, timeout=90.0)
        return _CLIENT.chat(prompt)


def _flatten(messages):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _cb():
    def fn(messages):
        resp = _raw_chat(_flatten(messages))
        return ChatResponse(
            text=getattr(resp, "text", "") or "",
            model=MODEL,
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
        )

    return fn


def _run(mode, label):
    import tempfile

    agent = Sherlock.with_callable(
        main_chat=_cb(),
        system_prompt="You are a warm, careful assistant. Reply in Korean, concise.",
        storage_dir=tempfile.mkdtemp(prefix=f"mc_{label}_"),
        embedding="fake",
        background=False,
        context_window=128_000,
        main_search_engine=None,
        inference_search_engine=None,
        memory_consistency_check=mode,
    )
    cid = agent._ensure_conversation().id
    agent.memory.add(
        conversation_id=cid,
        content=PINNED,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        pinned=True,
    )
    reply = agent.chat(MESSAGE)
    final = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    return {
        "label": label,
        "cue_in_slot": "MEMORY-CONSISTENCY CHECK" in final,
        "reply": reply,
    }


def main():
    print(f"=== memory-consistency A/B ===\n  pinned : {PINNED}\n  message: {MESSAGE}\n", flush=True)
    off = _run("off", "off")
    on = _run("code", "code")
    OUT.write_text(json.dumps({"pinned": PINNED, "message": MESSAGE, "off": off, "on": on},
                              ensure_ascii=False, indent=2))
    for r in (off, on):
        print(f"### {r['label'].upper()}  | cue_in_slot={r['cue_in_slot']}")
        print(f"  reply: {r['reply']}\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
