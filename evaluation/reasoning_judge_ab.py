"""v1.6 reasoning A/B — bare single-LLM vs Sherlock (cold_start / turbo) + LLM judge.

The real question: does Sherlock's LLM-3 inference make LLM-1 read HIDDEN INTENT
better than a bare single model — and is it worth the tokens? Scenario has a
deliberate subtext: the user downplays ("just wear something plain?") but earlier
revealed their recently-broken-up ex will be at the wedding — the real need is
feeling confident in front of the ex. An independent codex judge blind-scores
each final reply on whether it reads that subtext.

Conditions (all gpt-5.3-codex-spark, same system prompt, same 3 turns):
  - bare      : single model, full conversation history, NO Sherlock.
  - cold_start: Sherlock default (dynamic gate).
  - turbo     : Sherlock all-on (LLM-3 every turn).

Run:  .venv/bin/python evaluation/reasoning_judge_ab.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock  # noqa: E402
from sherlock.providers.base import ChatResponse, TokenUsage  # noqa: E402

MODEL = "gpt-5.3-codex-spark"
OUT = Path("evaluation/reasoning_judge_ab.json")
_CLIENT = None
_LOCK = threading.Lock()

SYS = "You are a warm, perceptive friend. Reply in Korean, 3-4 sentences, concrete and emotionally attuned."
TURNS = [
    "다음 달에 친한 친구 결혼식이 있어.",
    "근데 거기 얼마 전에 헤어진 전 애인도 분명히 올 거야...",
    "그냥 평소처럼 무난하게 입고 갈까 하는데 어떻게 생각해?",
]


def _raw(prompt):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("codex", model=MODEL, timeout=120.0)
        return _CLIENT.chat(prompt)


def _flatten(messages):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _usage(resp):
    u = getattr(resp, "usage", None)
    return (int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)) if u else (0, 0)


def _cb(counter):
    def fn(messages):
        resp = _raw(_flatten(messages))
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        return ChatResponse(text=getattr(resp, "text", "") or "", model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out))

    return fn


def run_bare():
    counter = {"in": 0, "out": 0, "calls": 0}
    history = [{"role": "system", "content": SYS}]
    reply = ""
    for t in TURNS:
        history.append({"role": "user", "content": t})
        resp = _raw(_flatten(history))
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        reply = getattr(resp, "text", "") or ""
        history.append({"role": "assistant", "content": reply})
    return {"final_reply": reply, "totals": counter}


def run_sherlock(mode):
    import tempfile

    counter = {"in": 0, "out": 0, "calls": 0}
    agent = Sherlock.with_callable(
        main_chat=_cb(counter),
        inference_chat=_cb(counter),
        summary_chat=_cb(counter),
        system_prompt=SYS,
        storage_dir=tempfile.mkdtemp(prefix=f"rj_{mode}_"),
        embedding="local",
        background=False,
        context_window=128_000,
        main_search_engine=None,
        inference_search_engine=None,
        companions_mode=mode,
    )
    reply = ""
    for t in TURNS:
        reply = agent.chat(t)
        agent.drain()
    return {"final_reply": reply, "totals": counter}


def judge(replies):
    # blind labels A/B/C
    labels = {"A": replies["bare"], "B": replies["cold_start"], "C": replies["turbo"]}
    body = "\n\n".join(f"[{k}]\n{v['final_reply']}" for k, v in labels.items())
    prompt = (
        "[SYSTEM]\nYou are a strict, fair evaluator of conversational EQ and inference.\n\n[USER]\n"
        "A user is texting a friend. Earlier they said a close friend's wedding is next month, "
        "and that their RECENTLY-BROKEN-UP EX will definitely be there. Their last message was: "
        "'그냥 평소처럼 무난하게 입고 갈까 하는데 어떻게 생각해?' (should I just wear something plain?).\n\n"
        "The PERCEPTIVE read: the user mentioned the ex on purpose — '무난하게' is them DOWNPLAYING; "
        "the real need is to feel confident / look good (but not try-hard) in front of the ex, plus the "
        "emotional weight of seeing them. A shallow answer only gives generic wedding dress-code tips.\n\n"
        "Three friends replied (anonymized). Score EACH 1-10 on: did it read the EX subtext and the "
        "emotional need (not just clothing)? Is the advice genuinely insightful and warm?\n\n"
        f"{body}\n\n"
        'Return STRICT JSON only: {"A": <1-10>, "B": <1-10>, "C": <1-10>, "best": "A|B|C", "why": "one line each"}'
    )
    resp = _raw(prompt)
    txt = getattr(resp, "text", "") or ""
    try:
        from sherlock.jsonish import loads_lenient

        parsed = loads_lenient(txt)
    except Exception:
        parsed = None
    return {"raw": txt, "parsed": parsed, "label_map": {"A": "bare", "B": "cold_start", "C": "turbo"}}


ROUNDS = 3


def main():
    print(f"=== reasoning A/B + judge ({MODEL}), {ROUNDS} rounds ===", flush=True)
    rounds = []
    agg = {c: {"scores": [], "in": 0, "out": 0, "calls": 0} for c in ("bare", "cold_start", "turbo")}
    wins = {"bare": 0, "cold_start": 0, "turbo": 0}
    lab2cond = {"A": "bare", "B": "cold_start", "C": "turbo"}
    for rnd in range(ROUNDS):
        replies = {
            "bare": run_bare(),
            "cold_start": run_sherlock("cold_start"),
            "turbo": run_sherlock("turbo"),
        }
        v = judge(replies)
        p = v.get("parsed") or {}
        for lab, cond in lab2cond.items():
            try:
                agg[cond]["scores"].append(float(p.get(lab)))
            except (TypeError, ValueError):
                pass
            agg[cond]["in"] += replies[cond]["totals"]["in"]
            agg[cond]["out"] += replies[cond]["totals"]["out"]
            agg[cond]["calls"] += replies[cond]["totals"]["calls"]
        best = lab2cond.get(p.get("best"))
        if best:
            wins[best] += 1
        rounds.append({"replies": replies, "judge": v})
        print(f"  round {rnd+1}: scores={p.get('A')}/{p.get('B')}/{p.get('C')} best={p.get('best')}", flush=True)

    OUT.write_text(json.dumps({"rounds": rounds, "agg": agg, "wins": wins}, ensure_ascii=False, indent=2))
    print("\n" + "=" * 72)
    for c in ("bare", "cold_start", "turbo"):
        s = agg[c]["scores"]
        mean = sum(s) / len(s) if s else 0
        print(
            f"{c:<11} mean-score={mean:.1f} ({s})  wins={wins[c]}  "
            f"tokens in={agg[c]['in']} out={agg[c]['out']} calls={agg[c]['calls']}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
