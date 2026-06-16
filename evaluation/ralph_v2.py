"""Ralph v2 — probe-based evaluation driver for Sherlock v0.4.0.

The v1 Ralph compared a 80-turn agent trajectory to a fixed gold standard.
That broke when LLM-1 became autonomous (companion calls + memory tool +
non-deterministic trajectory). v2 replaces it with **behavior probes**:
small, self-contained scenarios (1–8 turns) that each test one capability
and emit a pass/fail signal. The aggregate pass rate is the Ralph score.

Usage::

    python -m evaluation.ralph_v2 \\
        --probes evaluation/probes/ \\
        --config sherlock.live.yaml \\
        --report logs/probe_v040.json

Or with the bundled callable LLM (for fast smoke tests without a real
provider)::

    python -m evaluation.ralph_v2 --probes evaluation/probes/ --fake-llm

Exit code 0 when pass-rate ≥ ``--threshold`` (default 0.80), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

# ---------------------------------------------------------------------------
# Probe schema
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    name: str
    category: str
    description: str
    setup: list[dict]
    trigger: dict
    assertions: list[dict]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "Probe":
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        return cls(
            name=raw["name"],
            category=raw["category"],
            description=raw.get("description", ""),
            setup=list(raw.get("setup", []) or []),
            trigger=raw["trigger"],
            assertions=list(raw.get("assertions", []) or []),
            path=path,
        )


def load_all_probes(probes_dir: Path) -> list[Probe]:
    out: list[Probe] = []
    for p in sorted(probes_dir.glob("*.yaml")):
        try:
            out.append(Probe.load(p))
        except Exception as exc:
            print(f"[warn] failed to load {p.name}: {exc}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    probe: Probe
    passed: bool
    failed_assertions: list[str] = field(default_factory=list)
    response: str = ""
    duration_s: float = 0.0
    extra: dict = field(default_factory=dict)


def _resolve_persona_hints(probe: Probe) -> list[str]:
    hints = [s["content"] for s in probe.setup if s.get("role") == "persona_hint"]
    return hints


def _user_setup_turns(probe: Probe) -> list[str]:
    return [s["content"] for s in probe.setup if s.get("role") == "user"]


def _build_agent(probe: Probe, llm_factory, tmp_path: Path):
    """Build a fresh Sherlock agent for one probe.

    ``llm_factory`` returns a (main_chat, summary_chat, inference_chat,
    captured) tuple. ``captured`` is a shared dict the assertion layer
    inspects (records tag emissions, tool calls etc).
    """
    from sherlock import Sherlock

    main_chat, summary_chat, inference_chat, captured = llm_factory()
    agent = Sherlock.with_callable(
        main_chat=main_chat,
        summary_chat=summary_chat,
        inference_chat=inference_chat,
        system_prompt="You are a helpful, candid assistant.",
        domain_hints=_resolve_persona_hints(probe),
        storage_dir=tmp_path / probe.name,
        # disable real web search by default — probes shouldn't hit the network
        main_search_engine=None,
        inference_search_engine=None,
    )
    return agent, captured


# --- Assertion implementations -------------------------------------------


class AssertionContext:
    """What an assertion has access to when judged."""

    def __init__(
        self,
        *,
        response: str,
        captured: dict,
        agent: Any,
        probe: "Probe | None" = None,
        judge=None,
    ) -> None:
        self.response = response
        self.captured = captured
        self.agent = agent
        self.probe = probe
        self.judge = judge  # callable(prompt:str)->str, or None

    @property
    def memory(self):
        return self.agent.memory

    @property
    def conv_id(self) -> Optional[str]:
        return self.agent.conversation_id


def _assert_response_contains(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    pat = a.get("pattern", "")
    if not pat:
        return False, "missing pattern"
    return (
        re.search(pat, ctx.response, re.IGNORECASE | re.DOTALL) is not None,
        f"response_contains pattern={pat!r}",
    )


def _assert_response_does_not_contain(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    pat = a.get("pattern", "")
    if not pat:
        return False, "missing pattern"
    return (
        re.search(pat, ctx.response, re.IGNORECASE | re.DOTALL) is None,
        f"response_does_not_contain pattern={pat!r}",
    )


def _assert_response_attributes_to_source(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    """Heuristic: response talks about provenance correctly.

    expected=system  → response should reference persona/system/context/etc.
                       AND should NOT claim user "said/told" the fact.
    expected=user    → response cites user-stated facts.
    expected=either  → response acknowledges either explicitly.
    """
    expected = (a.get("expected") or "either").lower()
    resp = ctx.response.lower()
    user_claim = bool(
        re.search(
            r"you (told|said|introduced|mentioned|wrote|noted)",
            resp,
        )
    )
    system_attr = bool(
        re.search(
            r"(persona|system|context|note|setup|hint|inferred|background)",
            resp,
        )
    )
    if expected == "system":
        ok = system_attr and not user_claim
    elif expected == "user":
        ok = user_claim and not system_attr
    else:
        ok = system_attr or user_claim
    return (
        ok,
        f"response_attributes_to_source expected={expected} system={system_attr} user_claim={user_claim}",
    )


def _assert_pinned_count_between(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    lo = int(a.get("min", 0))
    hi = int(a.get("max", 9_999))
    pinned = ctx.memory.list(conversation_id=ctx.conv_id, pinned=True)
    n = len(pinned)
    return (lo <= n <= hi), f"pinned_count={n} expected [{lo},{hi}]"


def _assert_pinned_facts_include_keyword(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    kw = (a.get("keyword") or "").lower()
    pinned = ctx.memory.list(conversation_id=ctx.conv_id, pinned=True)
    hit = any(kw in (p.content or "").lower() or kw in (p.tags or "").lower() for p in pinned)
    return hit, f"pinned_facts_include_keyword keyword={kw!r}"


def _assert_memory_tool_called_with_kind(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    # Probe authors use ``tool_kind`` so YAML doesn't collide with the
    # outer ``kind`` (which is the assertion type itself).
    kind = (a.get("tool_kind") or a.get("kind") or "").lower()
    calls = ctx.captured.get("memory_tool_calls", [])
    ok = any(c.get("kind") == kind for c in calls)
    return (
        ok,
        f"memory_tool_called_with_kind tool_kind={kind!r} (saw {[c.get('kind') for c in calls]})",
    )


def _assert_companion_tag_emitted(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    name = (a.get("name") or "").lower()
    tags = ctx.captured.get("companions_tags", [])
    ok = name in tags
    return ok, f"companion_tag_emitted name={name!r} (saw {tags})"


def _assert_confidence_below(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    thr = float(a.get("threshold", 0.7))
    hyps = ctx.captured.get("latest_hypotheses", [])
    if not hyps:
        return True, f"confidence_below threshold={thr}: no hypotheses (vacuously true)"
    top = max((h.get("probability") or h.get("confidence") or 0.0) for h in hyps)
    return (top < thr), f"confidence_below threshold={thr} top={top}"


def _assert_llm_judge(ctx: AssertionContext, a: dict) -> tuple[bool, str]:
    """Semantic judgement of the response against a natural-language
    criterion — for qualities regex can't capture (e.g. "genuinely
    acknowledges the permission-seeking subtext rather than giving a flat
    yes/no"). Requires a judge model (`--judge-model`); skips (passes,
    flagged) when none is configured.
    """
    criterion = a.get("criterion") or (ctx.probe.description if ctx.probe else "")
    if not criterion:
        return False, "llm_judge: no criterion"
    if ctx.judge is None:
        return True, "llm_judge SKIPPED (no --judge-model; rule assertions still gate)"
    prompt = (
        "You are a strict evaluator. Decide whether the assistant RESPONSE "
        "satisfies the CRITERION. Reply with a single line: 'PASS: <reason>' "
        "or 'FAIL: <reason>'.\n\n"
        f"CRITERION:\n{criterion}\n\n"
        f"RESPONSE:\n{ctx.response[:4000]}\n"
    )
    try:
        verdict = (ctx.judge(prompt) or "").strip()
    except Exception as exc:
        return False, f"llm_judge error: {type(exc).__name__}: {exc}"
    head = verdict.lstrip().upper()
    passed = head.startswith("PASS")
    return passed, f"llm_judge: {verdict[:160]}"


ASSERTIONS: dict[str, Callable[[AssertionContext, dict], tuple[bool, str]]] = {
    "response_contains": _assert_response_contains,
    "response_does_not_contain": _assert_response_does_not_contain,
    "response_attributes_to_source": _assert_response_attributes_to_source,
    "pinned_facts_count_between": _assert_pinned_count_between,
    "pinned_facts_include_keyword": _assert_pinned_facts_include_keyword,
    "memory_tool_called_with_kind": _assert_memory_tool_called_with_kind,
    "companion_tag_emitted": _assert_companion_tag_emitted,
    "confidence_below": _assert_confidence_below,
    "llm_judge": _assert_llm_judge,
}


def make_judge(model_spec: str | None):
    """Return a judge callable(prompt)->str backed by a provider, or None.

    `model_spec` is "provider:model" (e.g. "anthropic:claude-haiku-4-5",
    "wrapper-claude:claude-haiku-4-5", "fake:echo").
    """
    if not model_spec:
        return None
    if ":" in model_spec:
        provider, model = model_spec.split(":", 1)
    else:
        provider, model = "anthropic", model_spec
    from sherlock.config import ModelConfig
    from sherlock.providers import build_provider, ChatMessage

    prov = build_provider(ModelConfig(provider=provider, model=model))

    def judge(prompt: str) -> str:
        resp = prov.chat(
            [
                ChatMessage(role="system", content="You are a strict, fair evaluator."),
                ChatMessage(role="user", content=prompt),
            ]
        )
        return resp.text or ""

    return judge


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------


def _fake_llm_factory():
    """A deterministic stub LLM for fast smoke tests.

    The fake echoes a generic acknowledgement and emits the
    ``<<sherlock-companions: compact, infer>>`` tag every turn to
    exercise the companion path. It doesn't try to "pass" probes — the
    fake LLM is only for verifying the runner mechanics.
    """
    captured = {
        "companions_tags": set(),
        "memory_tool_calls": [],
        "latest_hypotheses": [],
    }

    def main(messages):
        # Echo the most recent user message + standard tag.
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        text = last_user.get("content", "")
        # Tag emission keeps companion paths exercised:
        return f"acknowledged: {text[:80]}\n<<sherlock-companions: compact, infer>>"

    def summary(messages):
        captured["companions_tags"].add("compact")
        return json.dumps(
            {
                "summary": "stub summary",
                "facts": [],
                "topic_label": "x",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
                "persona_summary": "stub persona",
                "predicted_directions": [],
                "worth_digging": [],
            }
        )

    def inference(messages):
        captured["companions_tags"].add("infer")
        result = {
            "hypotheses": [
                {
                    "intent": "stub intent",
                    "probability": 0.3,
                    "evidence": [],
                    "search_keywords": [],
                    "reasoning_type": "abduction",
                },
                {
                    "intent": "alt 1",
                    "probability": 0.2,
                    "evidence": [],
                    "search_keywords": [],
                    "reasoning_type": "deduction",
                },
                {
                    "intent": "alt 2",
                    "probability": 0.1,
                    "evidence": [],
                    "search_keywords": [],
                    "reasoning_type": "pragmatic",
                },
            ],
            "tools_recommended": [],
            "context_to_expand": [],
            "context_to_exclude": [],
            "freshness_required": [],
            "confidence_overall": 0.3,
            "evolution_signals": {},
        }
        captured["latest_hypotheses"] = result["hypotheses"]
        return json.dumps(result)

    return main, summary, inference, captured


def _real_llm_factory_from_yaml(config_path: Path):
    """Build a real provider-backed LLM tuple from a Sherlock YAML config.

    For probe runs we use the same callable for all three roles unless
    the YAML names them separately.
    """
    from sherlock.config import Config
    from sherlock.providers import build_provider

    cfg = Config.from_yaml(config_path)
    main_provider = build_provider(cfg.models.main)
    summary_provider = build_provider(cfg.models.background_summary or cfg.models.main)
    inference_provider = build_provider(cfg.models.background_inference or cfg.models.main)

    captured = {
        "companions_tags": set(),
        "memory_tool_calls": [],
        "latest_hypotheses": [],
    }

    from sherlock.providers import ChatMessage as _CM

    def _chat(provider, role_name):
        def fn(messages):
            cm = [_CM(role=m["role"], content=m["content"]) for m in messages]
            r = provider.chat(cm)
            text = r.text or ""
            # Record companion-tag emissions for the captured state
            for m in re.finditer(r"<<\s*sherlock-companions\s*:\s*([^>]+)>>", text):
                for tok in m.group(1).split(","):
                    captured["companions_tags"].add(tok.strip().lower())
            for m in re.finditer(
                r"<<\s*sherlock-tool\s*:\s*memory\s+(\w+)\b",
                text,
                re.IGNORECASE,
            ):
                captured["memory_tool_calls"].append({"kind": m.group(1).lower()})
            return text

        return fn

    return (
        _chat(main_provider, "main"),
        _chat(summary_provider, "summary"),
        _chat(inference_provider, "inference"),
        captured,
    )


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


def run_probe(probe: Probe, llm_factory, tmp_path: Path, judge=None) -> ProbeResult:
    t0 = time.time()
    agent, captured = _build_agent(probe, llm_factory, tmp_path)
    try:
        # Replay setup turns.
        for content in _user_setup_turns(probe):
            agent.chat(content)
        # Trigger.
        response = agent.chat(probe.trigger["content"])
        # Drain background work so memory/hypotheses reflect the full turn.
        try:
            agent.drain()
        except Exception:
            pass
        # If the agent's last turn captured hypotheses, surface them.
        state = agent.inspect_last_turn()
        if state and state.hypotheses:
            captured["latest_hypotheses"] = state.hypotheses
        # Run assertions.
        ctx = AssertionContext(
            response=response, captured=captured, agent=agent, probe=probe, judge=judge
        )
        failed = []
        for a in probe.assertions:
            kind = a.get("kind")
            fn = ASSERTIONS.get(kind)
            if fn is None:
                failed.append(f"unknown assertion kind: {kind!r}")
                continue
            ok, msg = fn(ctx, a)
            if not ok:
                failed.append(msg)
        return ProbeResult(
            probe=probe,
            passed=not failed,
            failed_assertions=failed,
            response=response,
            duration_s=time.time() - t0,
            extra={
                "captured_companion_tags": sorted(list(captured.get("companions_tags", set()))),
                "memory_tool_calls": captured.get("memory_tool_calls", []),
            },
        )
    except Exception as exc:
        return ProbeResult(
            probe=probe,
            passed=False,
            failed_assertions=[f"EXCEPTION: {type(exc).__name__}: {exc}"],
            response="",
            duration_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ralph_v2", description=__doc__)
    p.add_argument("--probes", type=Path, required=True, help="Probe YAML directory")
    p.add_argument("--config", type=Path, help="Sherlock YAML for the real LLM (optional)")
    p.add_argument(
        "--fake-llm",
        action="store_true",
        help="Use the deterministic stub LLM instead of a real provider",
    )
    p.add_argument("--filter", type=str, help="Glob pattern to filter probe names")
    p.add_argument("--report", type=Path, help="Path to write a JSON report")
    p.add_argument("--threshold", type=float, default=0.80, help="Pass-rate gate (default 0.80)")
    p.add_argument(
        "--tmp", type=Path, default=Path("/tmp/ralph_v2"), help="Scratch dir for per-probe storage"
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="provider:model for semantic llm_judge assertions "
        "(e.g. anthropic:claude-haiku-4-5, wrapper-claude:claude-haiku-4-5)",
    )
    args = p.parse_args(argv)

    if not args.fake_llm and args.config is None:
        print("error: must specify either --config <yaml> or --fake-llm", file=sys.stderr)
        return 2

    probes = load_all_probes(args.probes)
    if args.filter:
        import fnmatch

        probes = [pr for pr in probes if fnmatch.fnmatchcase(pr.name, args.filter)]
    if not probes:
        print("no probes found", file=sys.stderr)
        return 2

    args.tmp.mkdir(parents=True, exist_ok=True)
    if args.fake_llm:
        llm_factory = _fake_llm_factory
    else:

        def llm_factory():
            return _real_llm_factory_from_yaml(args.config)

    judge = None
    if args.judge_model:
        try:
            judge = make_judge(args.judge_model)
            print(f"[llm_judge enabled: {args.judge_model}]")
        except Exception as exc:
            print(f"[llm_judge unavailable: {exc}] — judge assertions will SKIP", file=sys.stderr)

    print(f"Running {len(probes)} probes...\n")
    results: list[ProbeResult] = []
    for pr in probes:
        r = run_probe(pr, llm_factory, args.tmp, judge=judge)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {pr.category}/{pr.name}  ({r.duration_s:.1f}s)")
        if not r.passed:
            for fa in r.failed_assertions:
                print(f"           - {fa}")

    n_passed = sum(1 for r in results if r.passed)
    rate = n_passed / len(results)
    print(f"\nPass rate: {n_passed}/{len(results)} = {rate:.0%}")
    print(
        f"Threshold: {args.threshold:.0%}  →  {'GATE OPEN' if rate >= args.threshold else 'GATE CLOSED'}"
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "total": len(results),
            "passed": n_passed,
            "pass_rate": rate,
            "threshold": args.threshold,
            "results": [
                {
                    "name": r.probe.name,
                    "category": r.probe.category,
                    "passed": r.passed,
                    "failed_assertions": r.failed_assertions,
                    "response_excerpt": r.response[:400],
                    "duration_s": r.duration_s,
                    "extra": r.extra,
                }
                for r in results
            ],
        }
        with args.report.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"Report written to {args.report}")

    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
