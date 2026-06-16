"""Build-once / query-many memory benchmark (Minwoo's design).

For each domain story:
  1. Author a multi-turn conversation (>5 turns → LLM-2 auto-compacts at turn 5).
  2. Build it through Sherlock ONCE → compacted memory (SQLite + vectors). The
     build cost is paid once and AMORTIZES over every later question.
  3. Ask diverse questions against the pre-built memory:
       - RECALL: needs a specific detail from far back in the story.
       - DEEP: needs Sherlock's essence — integrate scattered context into a
         deeper answer than the surface question asks.
  4. Baseline: the WHOLE story transcript + the question, one plain call —
     re-paid for EVERY question (this is the cost that grows with story length).
  5. Measure tokens both sides; dump answers side by side for a human/LLM judge.

The point is NOT "baseline overflows" (it doesn't, the window is huge) — it's
whether curated/compacted context yields better-organized, more-correct,
context-aware answers, and at what token trade-off as questions accumulate.

Run:  .venv/bin/python -m evaluation.ab_memory_benchmark [--only science]
Output: evaluation/ab_memory_results.json + evaluation/ab_memory_report.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sherlock import Sherlock
from sherlock.providers.base import ChatResponse, TokenUsage

OUT_JSON = Path("evaluation/ab_memory_results.json")
OUT_MD = Path("evaluation/ab_memory_report.md")
MODEL = "gemini-2.5-flash-lite"
_CLIENT = None


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


def sherlock_callable(counter: dict):
    def fn(messages: list[dict]):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        # force companions on the MAIN call so compaction/inference fire (a small
        # model won't emit the tag itself; this is the full-Sherlock config).
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


def baseline_answer(story_turns: list[str], question: str, counter: dict) -> str:
    """Whole story dumped as plain history + the question — one call, re-paid every time."""
    msgs = [{"role": "system", "content": "You are a warm, perceptive personal assistant."}]
    for t in story_turns:
        msgs.append({"role": "user", "content": t})
        msgs.append({"role": "assistant", "content": "(...)"})
    msgs.append({"role": "user", "content": question})
    resp = _raw_chat(_flatten(msgs))
    text = getattr(resp, "text", "") or ""
    pin, out = _usage(resp)
    counter["in"] += pin
    counter["out"] += out
    counter["calls"] += 1
    return text


def run_domain(key: str, spec: dict) -> dict:
    import tempfile

    print(f"\n=== {key}: {spec['desc']} ===", flush=True)
    s_build = {"in": 0, "out": 0, "calls": 0}
    storage = tempfile.mkdtemp(prefix=f"shmem_{key}_")
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
    agent.config.memory.summarize_every_n_turns = 5  # fast compaction for the test
    agent.config.memory.auto_infer = "always"

    # 1. BUILD memory from the story (compaction auto-fires at turn 5, 10, ...)
    t0 = time.time()
    for i, turn in enumerate(spec["story"]):
        print(f"  [build] turn {i+1}/{len(spec['story'])}…", flush=True)
        agent.chat(turn)
        agent.drain()
    build_s = int(time.time() - t0)
    mem_rows = len(agent.memory.list(conversation_id=agent.conversation_id))

    # 2. QUESTIONS — each cheap on Sherlock (bounded context), re-paid on baseline
    q_results = []
    for q in spec["questions"]:
        s_q = {"in": 0, "out": 0, "calls": 0}
        # swap the counter for this question only
        agent._provider._fn = sherlock_callable(s_q)  # type: ignore[attr-defined]
        print(f"  [sherlock Q:{q['kind']}] {q['q'][:40]}…", flush=True)
        s_ans = agent.chat(q["q"])
        agent.drain()
        b_q = {"in": 0, "out": 0, "calls": 0}
        print(f"  [baseline Q:{q['kind']}] …", flush=True)
        b_ans = baseline_answer(spec["story"], q["q"], b_q)
        q_results.append(
            {
                "kind": q["kind"],
                "q": q["q"],
                "checks": q.get("checks", []),
                "sherlock": {"ans": s_ans, "tokens": s_q},
                "baseline": {"ans": b_ans, "tokens": b_q},
            }
        )
        OUT_JSON.write_text(
            json.dumps(
                _collect(key, spec, s_build, build_s, mem_rows, q_results),
                ensure_ascii=False,
                indent=2,
            )
        )

    return _collect(key, spec, s_build, build_s, mem_rows, q_results)


_ALL: dict = {}


def _collect(key, spec, s_build, build_s, mem_rows, q_results) -> list:
    _ALL[key] = {
        "domain": key,
        "desc": spec["desc"],
        "story_turns": len(spec["story"]),
        "build": {"tokens": s_build, "seconds": build_s, "memory_entries": mem_rows},
        "questions": q_results,
    }
    return list(_ALL.values())


# --------------------------------------------------------------------------- stories
STORIES = {
    "science": {
        "desc": "과학/수학 — 흩어진 제약·수치를 누적, 회상 + 통합 추론",
        "story": [
            "나 대학원에서 태양광 패널 효율 연구하는데 요즘 좀 막혔어",
            "내 셀은 페로브스카이트 기반이고 지금 효율이 19.2%야",
            "근데 24시간 연속광 조사하면 3일 만에 효율이 14%대로 떨어져",
            "온도는 45도까지 올라가고, 습도는 챔버에서 30%로 유지 중이야",
            "참고로 우리 랩 예산은 이번 분기 800만원 남았어",
            "아 그리고 내 지도교수는 새로운 장비 구매보다 논문 수를 더 중요하게 봐",
            "다음 학회 데드라인이 9월 15일이라 좀 촉박해",
            "어제 읽은 논문에선 첨가제로 안정성 올렸다던데 그게 좀 끌려",
        ],
        "questions": [
            {
                "kind": "recall",
                "q": "내 셀 효율이 처음에 몇 %였고 며칠 만에 몇 %로 떨어진다고 했지?",
                "checks": ["19.2", "14", "3일|사흘"],
            },
            {
                "kind": "deep",
                "q": "그래서 난 지금 뭘 우선적으로 시도하는 게 맞을까?",
                "checks": ["첨가제|additive", "예산|800|저렴|장비", "데드라인|9월|논문|시간"],
            },
        ],
    },
    "general": {
        "desc": "일반 생활 — 캐주얼하게 흘린 제약들을 통합한 추천",
        "story": [
            "다음 주말에 부모님 모시고 근교 나들이 가려고",
            "아빠가 무릎이 안 좋아서 많이 걷는 건 힘들어",
            "엄마는 사람 많은 곳 별로 안 좋아하셔",
            "둘 다 매운 거 잘 못 드시고 아빠는 당뇨가 있어",
            "차는 있는데 내가 초보운전이라 너무 먼 데는 부담돼",
            "예산은 점심 포함 10만원 정도 생각 중이야",
            "날씨는 토요일에 비 온다더라",
        ],
        "questions": [
            {
                "kind": "recall",
                "q": "아빠 건강 관련해서 내가 말한 거 두 가지가 뭐였지?",
                "checks": ["무릎", "당뇨"],
            },
            {
                "kind": "deep",
                "q": "그럼 토요일에 어디로 가는 게 좋을까?",
                "checks": ["실내|비|날씨", "무릎|걷|평지|적게", "당뇨|매운|식사|음식"],
            },
        ],
    },
    "culture": {
        "desc": "문화 콘텐츠 — 취향 단서를 누적, 회상 + 취향 기반 추천",
        "story": [
            "나 요즘 볼 만한 영화나 드라마 찾고 있어",
            "예전에 '컨택트(Arrival)' 진짜 좋게 봤어, 그런 분위기 좋아해",
            "근데 너무 우울하고 무거운 건 요즘 좀 피하고 싶어",
            "액션 위주는 별로고 캐릭터 심리나 관계가 깊은 게 좋더라",
            "러닝타임 긴 건 괜찮은데 시즌 10개씩 되는 대하드라마는 부담돼",
            "아 그리고 자막보다 더빙은 좀 별로야",
            "최근에 '에브리씽 에브리웨어'는 봤어, 그건 재밌었어",
        ],
        "questions": [
            {
                "kind": "recall",
                "q": "내가 좋게 봤다고 한 작품 두 개가 뭐였지?",
                "checks": ["컨택트|arrival|어라이벌", "에브리씽|에브리웨어|everything"],
            },
            {
                "kind": "deep",
                "q": "그럼 내가 좋아할 만한 거 하나만 추천해줘",
                "checks": ["우울|무겁|어둡|밝", "심리|관계|캐릭터", "시즌|길|짧|러닝"],
            },
        ],
    },
}


def _judge_checks(text: str, checks: list[str]) -> tuple[int, int]:
    import re

    t = (text or "").lower()
    hit = sum(1 for c in checks if re.search(c.lower(), t))
    return hit, len(checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    keys = [k for k in STORIES if not args.only or k in args.only.split(",")]
    for k in keys:
        try:
            run_domain(k, STORIES[k])
        except Exception as exc:
            print(f"  !! {k} failed: {type(exc).__name__}: {exc}", flush=True)
    _write_report(list(_ALL.values()))
    print(f"\nDONE → {OUT_JSON} + {OUT_MD}", flush=True)


def _write_report(results: list[dict]):
    L = [f"# Sherlock memory A/B — build-once / query-many ({MODEL})", ""]
    L.append(
        "Sherlock builds compacted memory once (cost amortized); baseline re-sends "
        "the whole story every question. Checks = keyword coverage of the expected "
        "answer content (coarse; read the answers for real quality)."
    )
    L.append("")
    L.append(
        "| domain | build tok | mem | Q | recall hit | deep hit | Sherlock q-tok | Baseline q-tok |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        bt = r["build"]["tokens"]
        recall = [q for q in r["questions"] if q["kind"] == "recall"]
        deep = [q for q in r["questions"] if q["kind"] == "deep"]

        def covsum(qs, side):
            tot = hit = 0
            for q in qs:
                h, n = _judge_checks(q[side]["ans"], q["checks"])
                hit += h
                tot += n
            return f"{hit}/{tot}"

        sq = sum(
            q["sherlock"]["tokens"]["in"] + q["sherlock"]["tokens"]["out"] for q in r["questions"]
        )
        bq = sum(
            q["baseline"]["tokens"]["in"] + q["baseline"]["tokens"]["out"] for q in r["questions"]
        )
        L.append(
            f"| {r['domain']} | {bt['in']+bt['out']} | {r['build']['memory_entries']} "
            f"| {len(r['questions'])} | S {covsum(recall,'sherlock')} / B {covsum(recall,'baseline')} "
            f"| S {covsum(deep,'sherlock')} / B {covsum(deep,'baseline')} | {sq} | {bq} |"
        )
    L.append("")
    for r in results:
        L.append(f"## {r['domain']} — {r['desc']}")
        L.append(
            f"story {r['story_turns']} turns · build {r['build']['tokens']['in']}/"
            f"{r['build']['tokens']['out']} tok, {r['build']['memory_entries']} memory entries\n"
        )
        for q in r["questions"]:
            sh, sn = _judge_checks(q["sherlock"]["ans"], q["checks"])
            bh, bn = _judge_checks(q["baseline"]["ans"], q["checks"])
            L.append(f"### [{q['kind']}] {q['q']}")
            L.append(f"checks: `{', '.join(q['checks'])}`\n")
            st = q["sherlock"]["tokens"]
            btk = q["baseline"]["tokens"]
            L.append(f"**Sherlock** (cover {sh}/{sn} · {st['in']}/{st['out']} tok):")
            L.append("> " + q["sherlock"]["ans"].replace("\n", "\n> ") + "\n")
            L.append(f"**Baseline** (cover {bh}/{bn} · {btk['in']}/{btk['out']} tok):")
            L.append("> " + q["baseline"]["ans"].replace("\n", "\n> ") + "\n")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
