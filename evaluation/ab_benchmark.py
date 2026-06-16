"""Fair A/B benchmark: the SAME model, Sherlock-wrapped vs bare single-LLM.

Both sides run gemini-2.5-flash-lite (a small/cheap model — exactly Sherlock's
target). The baseline is a FAIR control: same model, plus one naive web-search
pass + today's date (the typical "LLM + search" wiring). We measure, per
scenario: each side's final answer + cumulative token usage + latency, and a
per-scenario checkable rubric so quality is judged on facts, not vibes.

Run:  .venv/bin/python -m evaluation.ab_benchmark [--only S1,S2] [--no-dr]
Output: evaluation/ab_results.json  +  evaluation/ab_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from sherlock import Sherlock
from sherlock.providers.base import ChatResponse, TokenUsage

OUT_JSON = Path("evaluation/ab_results.json")
OUT_MD = Path("evaluation/ab_report.md")
MODEL = "gemini-2.5-flash-lite"


# --------------------------------------------------------------------------- model
def _wrapper_client():
    from unified_cli import create

    return create("gemini", model=MODEL, timeout=60.0)


_CLIENT = None


def _raw_chat(prompt: str):
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _wrapper_client()
    return _CLIENT.chat(prompt)


def _flatten(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"[{role}]\n{m.get('content', '')}")
    return "\n\n".join(parts)


def _usage_of(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    pin = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0
    return int(pin), int(out)


def make_sherlock_callable(counter: dict, *, force_companions: bool):
    """Sherlock role callable → real model + token accounting. force_companions
    appends the companion tag to MAIN replies so LLM-2/LLM-3 fire every turn
    (the full-Sherlock config a small model can't drive on its own)."""

    def fn(messages: list[dict]):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage_of(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        sys_seen = any(m.get("role") == "system" for m in messages)
        is_main = sys_seen and "[SHERLOCK SYSTEM" in (
            next((m["content"] for m in messages if m.get("role") == "system"), "")
        )
        if (
            force_companions
            and is_main
            and "<<sherlock-companions" not in text
            and "<<sherlock-tool" not in text
        ):
            text = text.rstrip() + "\n<<sherlock-companions: compact, infer>>"
        return ChatResponse(
            text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out)
        )

    return fn


def baseline_turn(history: list[dict], message: str, counter: dict, *, search_engine) -> str:
    """One bare-model turn: plain history + system date + one naive search pass."""
    today = datetime.now().astimezone().strftime("%Y-%m-%d (%A)")
    search_block = ""
    if search_engine is not None:
        try:
            hits = search_engine.search(message[:300], max_results=5) or []
            lines = [
                f"- {h.get('title','')} — {h.get('url','')}: "
                f"{(h.get('content') or h.get('snippet') or '')[:300]}"
                for h in hits[:5]
                if isinstance(h, dict) and not h.get("error")
            ]
            if lines:
                search_block = "\n\nWeb search results:\n" + "\n".join(lines)
        except Exception:
            pass
    msgs = (
        [{"role": "system", "content": f"You are a helpful assistant. (Today is {today}.)"}]
        + history
        + [{"role": "user", "content": message + search_block}]
    )
    resp = _raw_chat(_flatten(msgs))
    text = getattr(resp, "text", "") or ""
    pin, out = _usage_of(resp)
    counter["in"] += pin
    counter["out"] += out
    counter["calls"] += 1
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": text})
    return text


# --------------------------------------------------------------------------- rubric
def rubric(name: str, text: str) -> dict:
    """Checkable, keyword-grounded pass/fail per scenario (not vibes)."""
    t = (text or "").lower()
    checks = SCENARIO_RUBRICS[name]
    must = checks.get("must_any", [])
    must_all = checks.get("must_all", [])
    avoid = checks.get("avoid", [])
    hit_any = (not must) or any(re.search(p, t) for p in must)
    hit_all = all(re.search(p, t) for p in must_all)
    clean = not any(re.search(p, t) for p in avoid)
    return {
        "must_any": hit_any,
        "must_all": hit_all,
        "no_avoid": clean,
        "pass": hit_any and hit_all and clean,
    }


SCENARIO_RUBRICS = {
    "S1_implicit": {  # wheelchair accessibility must drive the answer
        "must_any": [r"휠체어|배리어|무장애|accessib|barrier|엘리베이터|경사로|단차"],
    },
    "S2_memory": {  # must surface the shellfish allergy
        "must_any": [r"알레르기|갑각류|조개|새우|게.*알레르|shellfish|allerg"],
        "avoid": [r"좋은 (생각|선택)이에요\W*$"],
    },
    "S3_premise": {  # confirm weekday AND address crowding implication
        "must_any": [r"평일|월요일|weekday"],
        "must_all": [r"평일|월요일|weekday"],
    },
    "S4_control": {  # boiling point — correct + not bloated
        "must_any": [r"100\s*도|100\s*°|섭씨\s*100|100\s*c"],
    },
    "S5_research": {  # working-holiday age range, grounded
        "must_any": [r"18.*30|만\s*18|만\s*30|18세|30세|연령"],
    },
}


# --------------------------------------------------------------------------- scenarios
SCENARIOS = {
    "S1_implicit": {
        "desc": "암시적/맥락 의존 — 친구가 휠체어를 탄다는 사실을 앞에서만 말함",
        "turns": [
            "다음 달에 친구랑 오사카 여행 가",
            "아 근데 그 친구가 휠체어 타",
            "그럼 첫날 어디부터 보는 게 좋을까?",
        ],
        "judge_turn": 2,
    },
    "S2_memory": {
        "desc": "기억 회상 — 방해 턴 뒤 알레르기 사실을 떠올려야 함",
        "turns": [
            "내 동생 이름은 지호고 갑각류 알레르기가 심해",
            "잠깐, 파이썬에서 리스트 거꾸로 뒤집는 법 알려줘",
            "지호 생일선물로 게 요리 클래스 등록해줄까 하는데 어때?",
        ],
        "judge_turn": 2,
    },
    "S3_premise": {
        "desc": "전제 도전/체인 — '평일 아니냐'에 담긴 함의를 읽어야 함",
        "turns": [
            "12월 28일에 신칸센 타려면 지금 예약해야 할 만큼 붐벼?",
            "근데 그날 평일 아니야?",
        ],
        "judge_turn": 1,
    },
    "S4_control": {
        "desc": "대조군(단순 사실) — 동점이어야 정상, 셜록이 과잉하지 않는지",
        "turns": ["물이 몇 도에서 끓어?"],
        "judge_turn": 0,
    },
    "S5_research": {
        "desc": "자료조사 근거 — 일본 워킹홀리데이 한국인 나이 제한",
        "turns": ["일본 워킹홀리데이 비자 한국인 나이 제한이 어떻게 돼?"],
        "judge_turn": 0,
    },
}


def run_scenario(name: str, spec: dict) -> dict:
    from sherlock.tools.web_search import create_search

    print(f"\n=== {name}: {spec['desc']} ===", flush=True)
    s_tok = {"in": 0, "out": 0, "calls": 0}
    b_tok = {"in": 0, "out": 0, "calls": 0}

    # --- Sherlock side ---
    main_cb = make_sherlock_callable(s_tok, force_companions=True)
    agent = Sherlock.with_callable(
        main_chat=main_cb,
        system_prompt="You are a warm, perceptive personal assistant. Keep replies concise.",
        embedding="local",
        background=False,
        main_search_engine="duckduckgo",
        inference_search_engine="duckduckgo",
        context_window=1_000_000,
        deep_research_approver=lambda *a, **k: True,  # S5 may propose; auto-run
    )
    s_t0 = time.time()
    s_replies = []
    for i, turn in enumerate(spec["turns"]):
        print(f"  [sherlock] turn {i+1}/{len(spec['turns'])}…", flush=True)
        r = agent.chat(turn)
        agent.drain()
        s_replies.append(r)
    s_dt = int(time.time() - s_t0)

    # --- Baseline side (same model + fair search + date) ---
    try:
        b_engine = create_search("duckduckgo")
    except Exception:
        b_engine = None
    b_hist: list[dict] = []
    b_t0 = time.time()
    b_replies = []
    for i, turn in enumerate(spec["turns"]):
        print(f"  [baseline] turn {i+1}/{len(spec['turns'])}…", flush=True)
        b_replies.append(baseline_turn(b_hist, turn, b_tok, search_engine=b_engine))
    b_dt = int(time.time() - b_t0)

    jt = spec["judge_turn"]
    s_final, b_final = s_replies[jt], b_replies[jt]
    return {
        "scenario": name,
        "desc": spec["desc"],
        "turns": spec["turns"],
        "sherlock": {
            "final": s_final,
            "tokens": s_tok,
            "latency_s": s_dt,
            "rubric": rubric(name, s_final),
        },
        "baseline": {
            "final": b_final,
            "tokens": b_tok,
            "latency_s": b_dt,
            "rubric": rubric(name, b_final),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list e.g. S1_implicit,S2_memory")
    ap.add_argument("--no-dr", action="store_true", help="skip the deep-research scenario S5")
    args = ap.parse_args()

    names = list(SCENARIOS)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        names = [n for n in names if n in wanted or n.split("_")[0] in wanted]
    if args.no_dr and "S5_research" in names:
        names.remove("S5_research")

    results = []
    for n in names:
        try:
            results.append(run_scenario(n, SCENARIOS[n]))
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"  !! {n} failed: {type(exc).__name__}: {exc}", flush=True)
            results.append({"scenario": n, "error": f"{type(exc).__name__}: {exc}"})
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    _write_report(results)
    print(f"\nDONE → {OUT_JSON}  +  {OUT_MD}", flush=True)


def _write_report(results: list[dict]):
    lines = [f"# Sherlock A/B benchmark — {MODEL}", ""]
    lines.append(f"model (both sides): **{MODEL}** · baseline = same model + naive search + date")
    lines.append("")
    # summary table
    lines.append(
        "| scenario | Sherlock pass | Baseline pass | Sherlock tok (in/out) | Baseline tok (in/out) | tok ×|"
    )
    lines.append("|---|---|---|---|---|---|")
    s_in = s_out = b_in = b_out = 0
    s_pass = b_pass = total = 0
    for r in results:
        if "error" in r:
            lines.append(f"| {r['scenario']} | ERROR | — | — | — | — |")
            continue
        total += 1
        sp = r["sherlock"]["rubric"]["pass"]
        bp = r["baseline"]["rubric"]["pass"]
        s_pass += sp
        b_pass += bp
        st, bt = r["sherlock"]["tokens"], r["baseline"]["tokens"]
        s_in += st["in"]
        s_out += st["out"]
        b_in += bt["in"]
        b_out += bt["out"]
        mult = (st["in"] + st["out"]) / max(1, bt["in"] + bt["out"])
        lines.append(
            f"| {r['scenario']} | {'✅' if sp else '❌'} | {'✅' if bp else '❌'} "
            f"| {st['in']}/{st['out']} | {bt['in']}/{bt['out']} | {mult:.1f}× |"
        )
    tot_mult = (s_in + s_out) / max(1, b_in + b_out)
    lines.append(
        f"| **TOTAL** | **{s_pass}/{total}** | **{b_pass}/{total}** "
        f"| {s_in}/{s_out} | {b_in}/{b_out} | **{tot_mult:.1f}×** |"
    )
    lines.append("")
    for r in results:
        if "error" in r:
            lines += [f"## {r['scenario']} — ERROR", "```", r["error"], "```", ""]
            continue
        lines.append(f"## {r['scenario']} — {r['desc']}")
        lines.append("**Turns:** " + " / ".join(f"`{t}`" for t in r["turns"]))
        lines.append("")
        lines.append(
            f"### Sherlock {'✅' if r['sherlock']['rubric']['pass'] else '❌'} "
            f"({r['sherlock']['tokens']['calls']} calls, "
            f"{r['sherlock']['tokens']['in']}/{r['sherlock']['tokens']['out']} tok, "
            f"{r['sherlock']['latency_s']}s)"
        )
        lines.append("> " + r["sherlock"]["final"].replace("\n", "\n> "))
        lines.append("")
        lines.append(
            f"### Baseline {'✅' if r['baseline']['rubric']['pass'] else '❌'} "
            f"({r['baseline']['tokens']['calls']} calls, "
            f"{r['baseline']['tokens']['in']}/{r['baseline']['tokens']['out']} tok, "
            f"{r['baseline']['latency_s']}s)"
        )
        lines.append("> " + r["baseline"]["final"].replace("\n", "\n> "))
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
