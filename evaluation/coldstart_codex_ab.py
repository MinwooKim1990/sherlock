"""v1.6 REAL-model A/B — Quiescence Gate (cold_start) vs turbo, gpt-5.3-codex-spark.

Re-run of the live cold_start experiment on a real frontier model (gemini CLI
auth was revoked mid-session; codex-spark works). Same single-topic Korean chat;
captures REAL codex usage + call counts + the actual LLM-1 replies so quality
parity (cold_start ≈ turbo) is visible at a lower companion-call cost. Each
companion call is a full codex round-trip, so fewer calls = real token+latency
savings on the quiet turns.

Run:  .venv/bin/python evaluation/coldstart_codex_ab.py
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
OUT = Path("evaluation/coldstart_codex_ab.json")
_CLIENT = None
_LOCK = threading.Lock()

TURNS = [
    "이번 주말에 그냥 집에서 푹 쉬려고 해요.",
    "넷플릭스 보면서 좀 늘어지고 싶네요.",
    "음... 근데 이거 진짜 해야 할지 말아야 할지 모르겠어.",  # signal
    "아무튼 주말엔 푹 쉴래요.",
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


def _cb(counter, role):
    def fn(messages):
        resp = _raw(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        u = getattr(resp, "usage", None)
        pin = int(getattr(u, "input_tokens", 0) or 0) if u else 0
        out = int(getattr(u, "output_tokens", 0) or 0) if u else 0
        cached = int(getattr(u, "cached_tokens", 0) or 0) if u else 0
        counter["in"] += pin
        counter["out"] += out
        counter["cached"] += cached
        counter["calls"] += 1
        counter[role] = counter.get(role, 0) + 1
        return ChatResponse(text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out))

    return fn


def _run(mode):
    import tempfile

    counter = {"in": 0, "out": 0, "cached": 0, "calls": 0}
    events = []
    agent = Sherlock.with_callable(
        main_chat=_cb(counter, "main"),
        inference_chat=_cb(counter, "infer"),
        summary_chat=_cb(counter, "summary"),
        system_prompt="You are a warm, brief assistant. Reply in Korean, 2-3 sentences.",
        storage_dir=tempfile.mkdtemp(prefix=f"cx_{mode}_"),
        embedding="local",
        background=False,
        context_window=128_000,
        main_search_engine=None,
        inference_search_engine=None,
        companions_mode=mode,
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    replies, per_turn = [], []
    for t in TURNS:
        before = dict(counter)
        reply = agent.chat(t)
        agent.drain()
        replies.append(reply)
        gate = [d for (ty, d) in events if ty == "companion.gate"]
        per_turn.append(
            {
                "turn": t[:16],
                "infer": counter.get("infer", 0) - before.get("infer", 0),
                "fire3": (gate[-1].get("fire3") if gate else None),
                "calls": counter["calls"] - before["calls"],
            }
        )
    return {"mode": mode, "totals": counter, "per_turn": per_turn, "replies": replies}


def main():
    print(f"=== REAL codex A/B ({MODEL}) ===", flush=True)
    turbo = _run("turbo")
    print("  turbo done", flush=True)
    cold = _run("cold_start")
    print("  cold_start done", flush=True)
    OUT.write_text(json.dumps({"turbo": turbo, "cold_start": cold}, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    for r in (turbo, cold):
        t = r["totals"]
        print(
            f"\n### {r['mode'].upper()}  real-usage: in={t['in']} out={t['out']} "
            f"cached={t['cached']} | calls={t['calls']} (infer={t.get('infer',0)})"
        )
        for pt in r["per_turn"]:
            print(f"   {pt['turn']:<18} calls={pt['calls']} infer={pt['infer']} fire3={pt['fire3']}")
        for i, rep in enumerate(r["replies"]):
            print(f"   reply{i+1}: {rep[:90]}")
    ti, ci = turbo["totals"]["in"], cold["totals"]["in"]
    print("\n" + "=" * 70)
    print(
        f"cold_start real input tokens = {ci} vs turbo {ti} → {(1-ci/max(1,ti))*100:.0f}% fewer; "
        f"calls {cold['totals']['calls']} vs {turbo['totals']['calls']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
