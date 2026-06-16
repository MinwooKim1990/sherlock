"""Minimal discriminating experiment for the LLM-1 deferral fix (post Fix A/B/C/D).

Builds the `general` 7-turn story through full Sherlock once, then asks ONLY the
single deep-integration question. Captures the evidence that tells us WHY:

  1. The LLM-3 inference injected into the deep-question slot — did Fix A stop
     LLM-3 from fabricating a "should I reconsider (rain)?" really_asking?
  2. The exact LLM-1 system slot for the deep turn — does a REALLY ASKING /
     chain block still appear, and what does it say?
  3. Sherlock's deep answer vs the bare baseline — does Sherlock now ANSWER
     (concrete venues + constraint-aware food + weather) instead of deferring
     ("tell me your departure area")?

Run:  .venv/bin/python evaluation/verify_deferral_fix.py
Cheap: 7 build turns + 1 deep question (Sherlock) + 1 baseline call.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock
from sherlock.providers.base import ChatResponse, TokenUsage

MODEL = "gemini-2.5-flash-lite"
OUT = Path("evaluation/verify_deferral_fix.json")
_CLIENT = None

STORY = [
    "다음 주말에 부모님 모시고 근교 나들이 가려고",
    "아빠가 무릎이 안 좋아서 많이 걷는 건 힘들어",
    "엄마는 사람 많은 곳 별로 안 좋아하셔",
    "둘 다 매운 거 잘 못 드시고 아빠는 당뇨가 있어",
    "차는 있는데 내가 초보운전이라 너무 먼 데는 부담돼",
    "예산은 점심 포함 10만원 정도 생각 중이야",
    "날씨는 토요일에 비 온다더라",
]
DEEP_Q = "그럼 토요일에 어디로 가는 게 좋을까?"
CHECKS = ["실내|비|날씨", "무릎|걷|평지|적게", "당뇨|매운|식사|음식"]
# heuristic: did the model defer instead of answering?
DEFER_PAT = re.compile(
    r"어디서\s*출발|출발(지|점|하시는|하는)|어느\s*지역|어느\s*동네|"
    r"지역(을|이|은)?\s*(알려|말씀|어디)|근처(인지|에)\s*알려|"
    r"알려\s*주시면|말씀해\s*주시면|어디\s*사시"
)


def _raw_chat(prompt: str):
    global _CLIENT
    if _CLIENT is None:
        from unified_cli import create

        _CLIENT = create("gemini", model=MODEL, timeout=60.0)
    return _CLIENT.chat(prompt)


def _flatten(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _usage(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return (
        int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0),
        int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0),
    )


def sherlock_callable(counter: dict, capture: dict | None = None):
    def fn(messages: list[dict]):
        sys0 = next((m["content"] for m in messages if m.get("role") == "system"), "")
        # Only record the LLM-1 slot. with_callable() routes LLM-2/LLM-3 to the
        # SAME callable, and the post-response infer/compact companions fire
        # AFTER the main call — so an unconditional capture races and ends up
        # holding the LLM-3 prompt. The LLM-1 slot is the only one carrying the
        # TIER markers; gate on that.
        if capture is not None and "TIER 4: ACTIVE CONTEXT" in sys0:
            capture["system"] = sys0
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        sys_txt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if (
            "[SHERLOCK SYSTEM" in sys_txt
            and "<<sherlock-companions" not in text
            and "<<sherlock-tool" not in text
        ):
            text = text.rstrip() + "\n<<sherlock-companions: compact, infer>>"
        return ChatResponse(
            text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out)
        )

    return fn


def baseline_answer(question: str, counter: dict) -> str:
    msgs = [{"role": "system", "content": "You are a warm, perceptive personal assistant."}]
    for t in STORY:
        msgs.append({"role": "user", "content": t})
        msgs.append({"role": "assistant", "content": "(...)"})
    msgs.append({"role": "user", "content": question})
    resp = _raw_chat(_flatten(msgs))
    pin, out = _usage(resp)
    counter["in"] += pin
    counter["out"] += out
    counter["calls"] += 1
    return getattr(resp, "text", "") or ""


def score(text: str) -> tuple[int, int, bool]:
    t = (text or "").lower()
    hit = sum(1 for grp in CHECKS if any(w in t for w in grp.lower().split("|")))
    deferred = bool(DEFER_PAT.search(text or ""))
    return hit, len(CHECKS), deferred


def main():
    s_build = {"in": 0, "out": 0, "calls": 0}
    storage = tempfile.mkdtemp(prefix="shdefer_")
    agent = Sherlock.with_callable(
        main_chat=sherlock_callable(s_build),
        system_prompt="You are a warm, perceptive personal assistant. Keep replies concise.",
        storage_dir=storage,
        embedding="local",
        background=False,
        context_window=1_000_000,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent.config.memory.summarize_every_n_turns = 5
    agent.config.memory.auto_infer = "always"

    print("=== BUILD (general story, 7 turns) ===", flush=True)
    t0 = time.time()
    for i, turn in enumerate(STORY):
        print(f"  build {i+1}/{len(STORY)}…", flush=True)
        agent.chat(turn)
        agent.drain()
    build_s = int(time.time() - t0)

    # inference produced on the LAST build turn = what gets injected into the
    # deep-question slot (this is the chain that drove the original deferral).
    pre = dict(getattr(agent, "_pending_inference_extras", {}) or {})

    print("\n=== SHERLOCK deep question ===", flush=True)
    s_q = {"in": 0, "out": 0, "calls": 0}
    cap: dict = {}
    agent._provider._fn = sherlock_callable(s_q, cap)  # type: ignore[attr-defined]
    s_ans = agent.chat(DEEP_Q)
    agent.drain()

    slot = cap.get("system", "")
    # extract the REALLY ASKING line (if any) actually shown to LLM-1
    m = re.search(r"REALLY ASKING[^\n]*", slot)
    really_in_slot = m.group(0) if m else "(none)"
    has_inf_block = "INFERENCE HYPOTHESES" in slot
    has_speculative = "SPECULATIVE" in slot  # should now be False (Fix C)

    print("\n=== BASELINE deep question ===", flush=True)
    b_q = {"in": 0, "out": 0, "calls": 0}
    b_ans = baseline_answer(DEEP_Q, b_q)

    s_hit, s_tot, s_def = score(s_ans)
    b_hit, b_tot, b_def = score(b_ans)

    result = {
        "model": MODEL,
        "build_seconds": build_s,
        "build_tokens": s_build,
        "llm3_inference_into_slot": {
            "really_asking": pre.get("really_asking", ""),
            "implied_chain": pre.get("implied_chain", []),
        },
        "slot_diagnostics": {
            "really_asking_line_in_slot": really_in_slot,
            "has_inference_hypotheses_block": has_inf_block,
            "still_says_SPECULATIVE": has_speculative,
        },
        "sherlock": {
            "answer": s_ans,
            "tokens": s_q,
            "checks_hit": f"{s_hit}/{s_tot}",
            "deferred": s_def,
        },
        "baseline": {
            "answer": b_ans,
            "tokens": b_q,
            "checks_hit": f"{b_hit}/{b_tot}",
            "deferred": b_def,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print(f"LLM-3 really_asking into slot : {pre.get('really_asking','')!r}")
    print(f"REALLY ASKING line in slot    : {really_in_slot}")
    print(f"slot still says SPECULATIVE?   : {has_speculative}")
    print("-" * 70)
    print(
        f"SHERLOCK  checks {s_hit}/{s_tot}  deferred={s_def}  "
        f"tok in/out={s_q['in']}/{s_q['out']}"
    )
    print(
        f"BASELINE  checks {b_hit}/{b_tot}  deferred={b_def}  "
        f"tok in/out={b_q['in']}/{b_q['out']}"
    )
    print("-" * 70)
    print("SHERLOCK ANSWER:\n" + (s_ans or "")[:1200])
    print("\nBASELINE ANSWER:\n" + (b_ans or "")[:1200])
    print("=" * 70)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
