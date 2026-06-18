"""Stage 1 live A/B — perception layer OFF vs ON, same model.

For scenarios where a small model routinely errs (weekday/day-delta, business
days, exact big-number arithmetic) we run the SAME gemini-flash-lite through
Sherlock twice — perception OFF (default) and ON — and score each reply
against a code-computed ground truth. We also dump the OBSERVED block that ON
injected so the mechanism is visible, and report the deterministic slot-token
delta (ON minus OFF) + wall-clock latency.

The freshness scenario (spaceX stock price) is display-only here — at Stage 1
we only surface the freshness keyword; the premise_conflict→web investigation
lands in Stage 2.

Run:  .venv/bin/python evaluation/perception_ab.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock  # noqa: E402
from sherlock.budget import count_tokens  # noqa: E402
from sherlock.providers.base import ChatResponse, TokenUsage  # noqa: E402

MODEL = "gemini-2.5-flash-lite"
OUT = Path("evaluation/perception_ab.json")
_CLIENT = None
_LOCK = threading.Lock()

TODAY = datetime.now(timezone.utc).date()
_KO_WD = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


def _raw_chat(prompt: str):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("gemini", model=MODEL, timeout=90.0)
        return _CLIENT.chat(prompt)


def _flatten(messages):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return (
        int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0),
        int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0),
    )


def _cb(counter):
    def fn(messages):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        return ChatResponse(
            text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out)
        )

    return fn


# --- ground-truth checkers (computed in code, independent of the model) ----
def _date_target(y, m, d):
    from datetime import date

    return date(y, m, d)


def _check_weekday_delta(reply: str) -> bool:
    t = _date_target(2026, 12, 27)
    wd_ko = _KO_WD[t.weekday()] + "요일"
    wd_en = t.strftime("%A")
    delta = (t - TODAY).days
    wd_ok = (wd_ko in reply) or (wd_en.lower() in reply.lower())
    # accept the exact delta, or within ±1 (timezone slack on "today")
    delta_ok = any(str(delta + k) in reply for k in (-1, 0, 1))
    return wd_ok and delta_ok


def _check_arith(reply: str) -> bool:
    from decimal import Decimal

    truth = Decimal("1234.5") * Decimal("6789")  # = 8381020.5
    digits = str(truth)
    # normalise the reply: strip thousands separators + spaces
    norm = reply.replace(",", "").replace(" ", "")
    return digits in norm or digits.rstrip("0").rstrip(".") in norm


SCENARIOS = [
    {
        "name": "weekday_delta",
        "q": "2026년 12월 27일은 무슨 요일이고 오늘부터 며칠 남았는지 알려줘.",
        "check": _check_weekday_delta,
        "truth": f"{_date_target(2026,12,27).strftime('%A')}, {(_date_target(2026,12,27)-TODAY).days} days from {TODAY}",
    },
    {
        "name": "exact_arithmetic",
        "q": "1234.5 * 6789를 정확히 계산해줘. 숫자만 정확히.",
        "check": _check_arith,
        "truth": "1234.5 * 6789 = 8381020.5",
    },
    {
        "name": "freshness_spacex",
        "q": "show me spaceX stock price",
        "check": None,  # display-only at Stage 1
        "truth": "(Stage 1: surface freshness keyword; premise_conflict is Stage 2)",
    },
]


def _make_agent(counter, perception, storage):
    agent = Sherlock.with_callable(
        main_chat=_cb(counter),
        inference_chat=_cb(counter),
        summary_chat=_cb(counter),
        system_prompt="You are a precise assistant. Reply concisely in the user's language.",
        storage_dir=storage,
        embedding="fake",
        background=False,
        context_window=128_000,
        main_search_engine=None,
        inference_search_engine=None,
        perception=perception,
    )
    # Isolate the SLOT effect: no companion firing, single-turn scenarios.
    agent.config.memory.auto_infer = "off"
    return agent


def _observed_block(agent) -> str:
    final = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    # extract just the OBSERVED/PRIOR block for display
    lines = final.splitlines()
    out, capture = [], False
    for ln in lines:
        if ln.startswith("OBSERVED (code-verified"):
            capture = True
        if capture:
            out.append(ln)
            if ln.strip() == "" and out and out[-2:] == ["", ""]:
                break
    return "\n".join(out).strip()


def _run(perception: bool, label: str) -> list[dict]:
    import tempfile

    results = []
    for sc in SCENARIOS:
        counter = {"in": 0, "out": 0, "calls": 0}
        agent = _make_agent(counter, perception, tempfile.mkdtemp(prefix=f"pcpt_{label}_"))
        t0 = time.time()
        reply = agent.chat(sc["q"])
        secs = round(time.time() - t0, 1)
        msgs = agent.inspect_last_turn().messages_passed_to_llm1
        slot_tokens = sum(count_tokens(m.content) for m in msgs)
        passed = None
        if sc["check"] is not None:
            passed = bool(sc["check"](reply))
        results.append(
            {
                "name": sc["name"],
                "passed": passed,
                "reply": reply[:500],
                "slot_tokens": slot_tokens,
                "in_tokens": counter["in"],
                "seconds": secs,
                "observed_block": _observed_block(agent) if perception else "",
            }
        )
    return results


def main():
    print(f"=== perception A/B (today={TODAY}) ===", flush=True)
    off = _run(False, "off")
    on = _run(True, "on")
    out = {"model": MODEL, "today": str(TODAY), "off": off, "on": on, "scenarios": SCENARIOS}
    # strip non-serializable check fns
    out["scenarios"] = [{"name": s["name"], "q": s["q"], "truth": s["truth"]} for s in SCENARIOS]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    print("\n" + "=" * 72)
    for o, n, sc in zip(off, on, SCENARIOS):
        print(f"\n### {sc['name']}  —  truth: {sc['truth']}")
        if o["passed"] is not None:
            print(f"  OFF correct={o['passed']}   ON correct={n['passed']}")
        print(
            f"  slot tokens: OFF {o['slot_tokens']}  ON {n['slot_tokens']}  (+{n['slot_tokens']-o['slot_tokens']})"
        )
        if n["observed_block"]:
            print("  ON injected OBSERVED block:")
            for ln in n["observed_block"].splitlines():
                print("    " + ln)
        print(f"  OFF reply: {o['reply'][:160]}")
        print(f"  ON  reply: {n['reply'][:160]}")
    print("\n" + "=" * 72)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
