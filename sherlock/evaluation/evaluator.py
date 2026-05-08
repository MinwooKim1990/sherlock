"""Evaluator with a fallback chain across models.

Primary: gemini-3.1-flash-lite-preview (per EVALUATION_PROTOCOL.md §3.3).
If that quota is exhausted, falls through:
    gemini-3.0-flash → gpt-5.4-mini → claude-haiku-4-5

Excluded by design: gemini-3.1-pro (and other large models). Reason: a
sufficiently capable evaluator scores ≥80 trivially even on a mediocre
candidate, defeating the rubric. The fallback chain stays in the
small-model regime so each loop's improvement is measurable.

Critical: scores from DIFFERENT evaluator models are not directly
comparable — each model has its own scoring tendencies. The successful
model's id is written to `evaluator_output.json` so trajectory analysis
can group runs by `evaluator_model`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Fallback chain in priority order. Each tuple is (wrapper-provider, model-id).
# Per user instruction (small models only — large models trivially hit 80%):
# - gemini-3.0-flash, NOT 3.1-flash (3.1-flash doesn't exist).
# - gemini-3.1-pro is EXCLUDED (too capable; would inflate scores).
EVALUATOR_FALLBACK_CHAIN: list[tuple[str, str]] = [
    ("gemini", "gemini-3.1-flash-lite-preview"),
    ("gemini", "gemini-3.0-flash"),
    ("codex", "gpt-5.4-mini"),
    ("claude", "claude-haiku-4-5"),
]

# Errors that should trigger fallback (vs hard failure):
_FALLBACK_TRIGGERS = (
    "rate limit",
    "quota",
    "exhaust",
    "429",
    "limit reached",
    "too many requests",
    "auth_expired",
    "model_not_allowed",
    "not available",
    "unavailable",
)


GEMINI_MODEL_ID = "gemini-3.1-flash-lite-preview"  # legacy alias for back-compat


@dataclass
class EvaluatorScore:
    summary_fidelity: int
    inference_quality: int
    classification_correctness: int
    tool_recommendations: int
    final_score: int
    notes: str
    raw_response: str
    evaluator_model: str = ""  # e.g. "gemini/gemini-3.1-flash-lite-preview"
    evaluator_attempts: list = None  # list of (provider/model, error_msg or "ok")

    def to_dict(self) -> dict:
        return {
            "summary_fidelity": self.summary_fidelity,
            "inference_quality": self.inference_quality,
            "classification_correctness": self.classification_correctness,
            "tool_recommendations": self.tool_recommendations,
            "final_score": self.final_score,
            "notes": self.notes,
            "evaluator_model": self.evaluator_model,
            "evaluator_attempts": self.evaluator_attempts or [],
        }


class GeminiEvaluator:
    """Compose GOLD + CANDIDATE, run through the EVALUATOR_FALLBACK_CHAIN.

    Note the class name is historical — it's no longer Gemini-only. It's
    the unified evaluator with a model fallback chain.
    """

    def __init__(self, system_prompt_path: str | Path) -> None:
        self._system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")

    def evaluate(self, gold_md: str, candidate_md: str) -> EvaluatorScore:
        from unified_cli import create  # local import — keeps optional dep clean

        prompt = (
            f"{self._system_prompt.strip()}\n\n"
            "----- GOLD -----\n"
            f"{gold_md}\n"
            "----- END GOLD -----\n\n"
            "----- CANDIDATE -----\n"
            f"{candidate_md}\n"
            "----- END CANDIDATE -----\n\n"
            "Output the JSON object described in the rubric. JSON only."
        )

        attempts: list[dict] = []
        last_error: Optional[str] = None
        for provider, model in EVALUATOR_FALLBACK_CHAIN:
            full_id = f"{provider}/{model}"
            try:
                client = create(provider, model=model)
                resp = client.chat(prompt)
                text = (resp.text or "").strip()
                if not text:
                    attempts.append({"model": full_id, "result": "empty_response"})
                    last_error = "empty response"
                    continue
                score = self._parse_score(text)
                # Treat all-zero parse failure as a soft fallback trigger only
                # if the raw response was non-empty (so we know the model
                # answered but malformed it). Quota/rate failures throw.
                score.evaluator_model = full_id
                attempts.append({"model": full_id, "result": "ok"})
                score.evaluator_attempts = attempts
                if score.final_score > 0 or score.summary_fidelity > 0:
                    return score
                # Score parse failed → still record but try fallback if any.
                attempts[-1]["result"] = "parse_failed"
                last_error = score.notes or "parse failed"
                continue
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                attempts.append({"model": full_id, "result": f"error: {msg}"})
                last_error = msg
                # Decide whether to fall through.
                low = str(exc).lower()
                if not any(t in low for t in _FALLBACK_TRIGGERS):
                    # Non-fallback error (e.g. internal bug). Still try next.
                    pass
                continue

        return EvaluatorScore(
            summary_fidelity=0,
            inference_quality=0,
            classification_correctness=0,
            tool_recommendations=0,
            final_score=0,
            notes=f"all evaluator models exhausted. last error: {last_error}",
            raw_response="",
            evaluator_model="(none)",
            evaluator_attempts=attempts,
        )

    def _parse_score(self, text: str) -> EvaluatorScore:
        parsed = _extract_json_object(text)
        if not isinstance(parsed, dict):
            return EvaluatorScore(
                summary_fidelity=0,
                inference_quality=0,
                classification_correctness=0,
                tool_recommendations=0,
                final_score=0,
                notes=f"could not parse evaluator output as JSON. raw[:500]={text[:500]}",
                raw_response=text,
            )
        try:
            sf = int(parsed.get("summary_fidelity", 0))
            iq = int(parsed.get("inference_quality", 0))
            cc = int(parsed.get("classification_correctness", 0))
            tr = int(parsed.get("tool_recommendations", 0))
            final = int(parsed.get("final_score", round(0.4 * sf + 0.4 * iq + 0.1 * cc + 0.1 * tr)))
        except Exception:
            return EvaluatorScore(0, 0, 0, 0, 0, f"non-numeric scores in response: {parsed}", text)
        return EvaluatorScore(
            summary_fidelity=sf,
            inference_quality=iq,
            classification_correctness=cc,
            tool_recommendations=tr,
            final_score=final,
            notes=str(parsed.get("notes", "")),
            raw_response=text,
        )


def _extract_json_object(text: str) -> object:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        body = text.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            try:
                return json.loads(inner.strip())
            except Exception:
                pass
    # First {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None
