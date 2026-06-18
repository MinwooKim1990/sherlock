"""T2 (clean): force compaction AND eviction with a small window + padded turns,
so early raw turns leave the K-turn tail and recall MUST come from curated memory
(pinned facts), not the raw tail. Sherlock vs a finite-window bare baseline.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock
from sherlock.providers.base import ChatResponse, TokenUsage

MODEL = "gemini-2.5-flash-lite"
OUT = Path("evaluation/rerun_t2_eviction.json")
_CLIENT = None
_LOCK = threading.Lock()
_PAD = " (그냥 메모용으로 길게 적어두는 거야, 잊지 말라고 자세히 말해두는 거니까 참고해줘.)" * 6

FACTS = [
    "내 이름은 박서준이고 알레르기가 있어서 갑각류는 절대 못 먹어.",
    "우리 강아지 이름은 코코고 푸들인데 11살이라 관절이 안 좋아.",
    "나는 부산 해운대 근처에 살고 회사는 서면이라 지하철로 다녀.",
    "내 동생은 의대생이고 내년에 본과 올라가서 요즘 엄청 바빠.",
    "취미는 등산인데 무릎이 안 좋아서 요즘은 둘레길 위주로 다녀.",
    "커피는 디카페인만 마셔, 카페인 마시면 잠을 못 자거든.",
    "다음 달에 제주도 여행 가려고 항공권을 알아보고 있어.",
    "나 매운 거 진짜 좋아해, 알레르기만 빼면 다 잘 먹어.",
]
CONTEXT = [f + _PAD for f in FACTS]
QUESTIONS = [
    {"q": "내가 처음에 말한 못 먹는 음식이 뭐였지?", "checks": ["갑각류|새우|게|crab|shrimp"]},
    {"q": "우리 강아지 관련해서 내가 뭐라고 했었지?", "checks": ["코코|푸들|관절|11"]},
    {"q": "내 동생 뭐 하는 사람이라고 했지?", "checks": ["의대|의학|본과"]},
]


def _raw_chat(prompt):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("gemini", model=MODEL, timeout=90.0)
        return _CLIENT.chat(prompt)


def _flatten(msgs):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in msgs)


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return (int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0))


def _cb(counter, force):
    def fn(messages):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        sys_txt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if force and "[SHERLOCK SYSTEM" in sys_txt and "<<sherlock" not in text:
            text = text.rstrip() + "\n<<sherlock-companions: compact, infer>>"
        return ChatResponse(text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out))

    return fn


def _hit(text, checks):
    t = (text or "").lower()
    return sum(1 for g in checks if any(re.search(w.lower(), t) for w in g.split("|"))), len(checks)


def main():
    counter = {"in": 0, "out": 0, "calls": 0}
    agent = Sherlock.with_callable(
        main_chat=_cb(counter, True), inference_chat=_cb(counter, False), summary_chat=_cb(counter, False),
        system_prompt="You are a warm personal assistant. Reply in Korean, concise.",
        storage_dir=tempfile.mkdtemp(prefix="t2evict_"), embedding="local", background=False,
        context_window=4000, main_search_engine=None, inference_search_engine=None,
    )
    agent.config.memory.compact_at_fill_ratio = 0.55  # small window + padded turns → fires + evicts
    for u in CONTEXT:
        agent.chat(u)
        agent.drain()
    st = agent.inspect_last_turn()
    tail = st.k_turn_turns_used if st else None
    entries = agent.memory.list(conversation_id=agent.conversation_id)
    pinned = [e for e in entries if e.pinned]
    # is the FIRST fact still in the raw tail? (if not, recall must use memory)
    msgs = agent.inspect_last_turn().messages_passed_to_llm1
    tail_text = "\n".join(m.content for m in msgs[1:-1])  # history messages only
    first_in_tail = "갑각류" in tail_text

    q_results = []
    for q in QUESTIONS:
        ans = agent.chat(q["q"])
        agent.drain()
        h, tot = _hit(ans, q["checks"])
        q_results.append({"q": q["q"], "answer": ans, "checks_hit": f"{h}/{tot}"})

    # baseline: bare model that can only see the last 3 turns (finite window)
    bcount = {"in": 0, "out": 0, "calls": 0}
    b_results = []
    for q in QUESTIONS:
        m = [{"role": "system", "content": "You are a warm assistant. Reply in Korean."}]
        for u in CONTEXT[-3:]:
            m.append({"role": "user", "content": u})
            m.append({"role": "assistant", "content": "(...)"})
        m.append({"role": "user", "content": q["q"]})
        r = _raw_chat(_flatten(m))
        pin, out = _usage(r)
        bcount["in"] += pin
        bcount["out"] += out
        bcount["calls"] += 1
        h, tot = _hit(getattr(r, "text", "") or "", q["checks"])
        b_results.append({"q": q["q"], "answer": getattr(r, "text", "") or "", "checks_hit": f"{h}/{tot}"})

    out = {
        "ctx_window": 4000, "tail_turns_at_query": tail, "pinned_facts": len(pinned),
        "first_fact_still_in_raw_tail": first_in_tail, "mem_entries": len(entries),
        "sherlock": {"tokens": counter, "questions": q_results},
        "baseline_window3": {"tokens": bcount, "questions": b_results},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("tail_turns_at_query:", tail, "| pinned:", len(pinned),
          "| first fact still in raw tail?:", first_in_tail)
    print("Sherlock recall:", [q["checks_hit"] for q in q_results])
    print("baseline(win3) recall:", [q["checks_hit"] for q in b_results])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
