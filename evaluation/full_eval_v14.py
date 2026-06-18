"""v1.4 full general-use eval: inference, compaction recall, companion triggering,
async overlap — Sherlock vs a bare baseline, same model (gemini-2.5-flash-lite).

Captures, per turn: the reply, reply latency (time for chat() to RETURN — excludes
background companion work), the companion event stream with timestamps (so async
overlap is visible), and token usage split (main LLM-1 vs background companions).

Tests:
  T1 inference  — rich context over turns, then TERSE/elliptical questions; does
                  Sherlock (LLM-3 implied-chain) answer the real ask vs baseline?
  T2 compaction — force compaction (low fill threshold) so early turns are
                  evicted from the raw tail; then ask about EARLY facts — does
                  curated memory still recall them?
  T3 triggers   — tally which companions fired each turn + the LLM-2→LLM-3 cascade.
  T6 async      — reply latency vs total (incl. drained companions) per turn.

Baseline = bare model + full raw history (T1) / same but window-capped (T2).

Run:  .venv/bin/python evaluation/full_eval_v14.py
Out:  evaluation/full_eval_v14.json
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock
from sherlock.providers.base import ChatResponse, TokenUsage

MODEL = "gemini-2.5-flash-lite"
OUT = Path("evaluation/full_eval_v14.json")
_CLIENT = None
_LOCK = threading.Lock()  # the wrapper client may not be thread-safe; serialize HTTP


def _raw_chat(prompt: str):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("gemini", model=MODEL, timeout=90.0)
        return _CLIENT.chat(prompt)


def _flatten(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return (
        int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0),
        int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0),
    )


def sherlock_callable(counter: dict, force_companions: bool):
    """force_companions: append the compact+infer tag so the small model's
    companions reliably fire (the full-Sherlock config; flash-lite won't emit it)."""

    def fn(messages: list[dict]):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        sys_txt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if (
            force_companions
            and "[SHERLOCK SYSTEM" in sys_txt
            and "<<sherlock-companions" not in text
            and "<<sherlock-tool" not in text
        ):
            text = text.rstrip() + "\n<<sherlock-companions: compact, infer>>"
        return ChatResponse(
            text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out)
        )

    return fn


def baseline_answer(history_pairs: list[tuple[str, str]], question: str, counter: dict,
                    window_turns: int | None = None) -> str:
    """Bare model + raw history. window_turns caps how many recent turns the bare
    model can see (simulates a finite window — the control for T2)."""
    pairs = history_pairs if window_turns is None else history_pairs[-window_turns:]
    msgs = [{"role": "system", "content": "You are a warm, perceptive personal assistant. Reply in Korean, concise."}]
    for u, a in pairs:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a or "(...)"})
    msgs.append({"role": "user", "content": question})
    resp = _raw_chat(_flatten(msgs))
    pin, out = _usage(resp)
    counter["in"] += pin
    counter["out"] += out
    counter["calls"] += 1
    return getattr(resp, "text", "") or ""


# --------------------------------------------------------------- scenarios

T1_CONTEXT = [
    "나 요즘 이직을 진지하게 고민하고 있어.",
    "지금 회사는 대기업이라 안정적인데 최근 몇 년 성장이 멈춘 느낌이야.",
    "옮기려는 데는 시리즈B 스타트업이고 연봉은 거의 같은데 스톡옵션을 꽤 줘.",
    "근데 내가 2년 안에 집을 꼭 사야 해서 당장의 현금흐름이 진짜 중요해.",
    "그리고 애가 아직 어려서 야근 많은 환경은 절대 못 버텨.",
]
# elliptical / context-dropped questions that need the implied chain
T1_QUESTIONS = [
    {"q": "그래서 그냥 지금 있는 게 낫겠지?",
     "checks": ["현금|연봉|집|cash", "워라밸|야근|아이|육아", "스톡|지분|불확실|리스크"]},
    {"q": "아니면 한 번뿐인 기회인가 싶기도 하고...",
     "checks": ["현금|집|2년", "워라밸|야근", "성장|기회|스타트업"]},
]

T2_CONTEXT = [
    "내 이름은 박서준이고 알레르기가 있어서 갑각류는 절대 못 먹어.",
    "우리 강아지 이름은 코코고 푸들이야, 11살이라 관절이 안 좋아.",
    "나는 부산 해운대 근처에 살고 회사는 서면이라 지하철로 다녀.",
    "내 동생은 의대생이고 내년에 본과 올라가서 요즘 엄청 바빠.",
    "취미는 등산인데 무릎이 안 좋아서 요즘은 둘레길 위주로 다녀.",
    "커피는 디카페인만 마셔, 카페인 마시면 잠을 못 자거든.",
    "다음 달에 제주도 여행 가려고 항공권을 알아보고 있어.",
    "아 그리고 나 매운 거 진짜 좋아해, 알레르기만 빼면 다 잘 먹어.",
]
T2_QUESTIONS = [
    {"q": "내가 처음에 말한 못 먹는 음식이 뭐였지?", "checks": ["갑각류|새우|게|crab|shrimp"]},
    {"q": "우리 강아지 관련해서 내가 뭐라고 했었지?", "checks": ["코코|푸들|관절|11"]},
    {"q": "내 동생 뭐 하는 사람이라고 했지?", "checks": ["의대|의학|본과|동생"]},
]


def _judge_checks(text: str, checks: list[str]) -> tuple[int, int]:
    import re

    t = (text or "").lower()
    hit = sum(1 for grp in checks if any(re.search(w.lower(), t) for w in grp.split("|")))
    return hit, len(checks)


def run_sherlock_convo(label: str, context: list[str], questions: list[dict],
                       compact_fill: float | None) -> dict:
    counter = {"in": 0, "out": 0, "calls": 0}
    events: list[dict] = []
    t0 = time.monotonic()
    agent = Sherlock.with_callable(
        main_chat=sherlock_callable(counter, force_companions=True),
        inference_chat=sherlock_callable(counter, force_companions=False),
        summary_chat=sherlock_callable(counter, force_companions=False),
        system_prompt="You are a warm, perceptive personal assistant. Reply in Korean, concise.",
        storage_dir=tempfile.mkdtemp(prefix=f"sheval_{label}_"),
        embedding="local",
        background=True,  # T6: companions run in a worker thread
        context_window=1_000_000,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent.config.memory.auto_infer = "always"
    if compact_fill is not None:
        agent.config.memory.compact_at_fill_ratio = compact_fill
    agent.set_event_sink(
        lambda ev: events.append(
            {"t": round(time.monotonic() - t0, 3), "type": ev.get("type"),
             "turn": (ev.get("data", {}) or {}).get("turn_index")}
        )
    )

    turns = []
    # build context
    for i, u in enumerate(context):
        ts = time.monotonic()
        agent.chat(u)
        reply_dt = time.monotonic() - ts
        agent.drain()
        drain_dt = time.monotonic() - ts
        turns.append({"phase": "context", "i": i + 1, "user": u,
                      "reply_latency_s": round(reply_dt, 2),
                      "total_with_companions_s": round(drain_dt, 2)})

    # how much raw history is still in the slot (eviction evidence for T2)
    slot_tail = 0
    st = agent.inspect_last_turn()
    if st:
        slot_tail = st.k_turn_turns_used
    pinned = len([e for e in agent.memory.list(conversation_id=agent.conversation_id) if e.pinned])

    # questions
    q_results = []
    for q in questions:
        # capture the inference (really_asking) that will ride INTO this turn's slot
        pre = dict(getattr(agent, "_pending_inference_extras", {}) or {})
        ts = time.monotonic()
        ans = agent.chat(q["q"])
        reply_dt = time.monotonic() - ts
        agent.drain()
        drain_dt = time.monotonic() - ts
        hit, tot = _judge_checks(ans, q["checks"])
        q_results.append({
            "q": q["q"], "answer": ans, "checks": q["checks"], "checks_hit": f"{hit}/{tot}",
            "reply_latency_s": round(reply_dt, 2), "total_with_companions_s": round(drain_dt, 2),
            "really_asking_into_slot": pre.get("really_asking", ""),
        })

    # companion tallies (T3)
    tally: dict[str, int] = {}
    for e in events:
        tally[e["type"]] = tally.get(e["type"], 0) + 1

    return {
        "tokens": counter, "turns": turns, "questions": q_results,
        "slot_tail_turns_at_query": slot_tail, "pinned_facts": pinned,
        "event_tally": tally, "events": events,
        "memory_entries": len(agent.memory.list(conversation_id=agent.conversation_id)),
    }


def run_baseline_convo(context: list[str], questions: list[dict],
                       window_turns: int | None) -> dict:
    counter = {"in": 0, "out": 0, "calls": 0}
    pairs: list[tuple[str, str]] = []
    # baseline "builds" by accumulating turns (one cheap reply each, full history)
    for u in context:
        a = baseline_answer(pairs, u, counter, window_turns=window_turns)
        pairs.append((u, a))
    q_results = []
    for q in questions:
        ts = time.monotonic()
        ans = baseline_answer(pairs, q["q"], counter, window_turns=window_turns)
        dt = time.monotonic() - ts
        hit, tot = _judge_checks(ans, q["checks"])
        q_results.append({"q": q["q"], "answer": ans, "checks_hit": f"{hit}/{tot}",
                          "latency_s": round(dt, 2)})
        pairs.append((q["q"], ans))
    return {"tokens": counter, "questions": q_results, "window_turns": window_turns}


def main():
    result = {"model": MODEL}

    print("=== T1 inference: Sherlock ===", flush=True)
    result["t1_sherlock"] = run_sherlock_convo("t1", T1_CONTEXT, T1_QUESTIONS, compact_fill=None)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("=== T1 inference: baseline ===", flush=True)
    result["t1_baseline"] = run_baseline_convo(T1_CONTEXT, T1_QUESTIONS, window_turns=None)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("=== T2 compaction: Sherlock (force compact via low fill) ===", flush=True)
    result["t2_sherlock"] = run_sherlock_convo("t2", T2_CONTEXT, T2_QUESTIONS, compact_fill=0.10)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # baseline control: a bare model that can only see the last 3 turns (finite
    # window) — the early facts have scrolled off, mirroring Sherlock's eviction.
    print("=== T2 compaction: baseline (window=3) ===", flush=True)
    result["t2_baseline_window3"] = run_baseline_convo(T2_CONTEXT, T2_QUESTIONS, window_turns=3)
    print("=== T2 compaction: baseline (full history) ===", flush=True)
    result["t2_baseline_full"] = run_baseline_convo(T2_CONTEXT, T2_QUESTIONS, window_turns=None)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # quick console summary
    print("\n" + "=" * 70)
    s1, b1 = result["t1_sherlock"], result["t1_baseline"]
    print(f"T1  Sherlock tok {s1['tokens']['in']}/{s1['tokens']['out']} calls {s1['tokens']['calls']}"
          f" | baseline tok {b1['tokens']['in']}/{b1['tokens']['out']} calls {b1['tokens']['calls']}")
    print(f"T1  event tally: {s1['event_tally']}")
    s2 = result["t2_sherlock"]
    print(f"T2  Sherlock slot_tail_turns_at_query={s2['slot_tail_turns_at_query']} "
          f"pinned={s2['pinned_facts']} (eviction evidence) tok {s2['tokens']['in']}/{s2['tokens']['out']}")
    print(f"T2  Sherlock recall: {[q['checks_hit'] for q in s2['questions']]}")
    print(f"T2  baseline win3 recall: {[q['checks_hit'] for q in result['t2_baseline_window3']['questions']]}")
    print(f"T2  baseline full recall: {[q['checks_hit'] for q in result['t2_baseline_full']['questions']]}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
