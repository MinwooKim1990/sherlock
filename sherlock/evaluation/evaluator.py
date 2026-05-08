"""Gemini-Flash-Lite evaluator via cli-wrapper-unified (DEVIATION-001 path).

Uses the wrapper's Python import (`from unified_cli import create`) — preferred
over subprocess for speed. Subprocess form is the documented fallback.

Per EVALUATION_PROTOCOL.md §3.4 the rubric is a JSON-only weighted score:
final = 0.4*A + 0.4*B + 0.1*C + 0.1*D.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


GEMINI_MODEL_ID = "gemini-3.1-flash-lite-preview"


@dataclass
class EvaluatorScore:
    summary_fidelity: int
    inference_quality: int
    classification_correctness: int
    tool_recommendations: int
    final_score: int
    notes: str
    raw_response: str

    def to_dict(self) -> dict:
        return {
            "summary_fidelity": self.summary_fidelity,
            "inference_quality": self.inference_quality,
            "classification_correctness": self.classification_correctness,
            "tool_recommendations": self.tool_recommendations,
            "final_score": self.final_score,
            "notes": self.notes,
        }


class GeminiEvaluator:
    """Compose GOLD + CANDIDATE, send to Gemini Flash Lite via unified_cli, parse score."""

    def __init__(self, system_prompt_path: str | Path) -> None:
        self._system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")

    def evaluate(self, gold_md: str, candidate_md: str) -> EvaluatorScore:
        from unified_cli import create  # local import — keeps optional dep clean

        client = create("gemini", model=GEMINI_MODEL_ID)
        # The wrapper does not expose --system separately; prepend the rubric
        # inline. Per DEVIATION-001 in INTENT_DEVIATIONS.md.
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
        try:
            resp = client.chat(prompt)
        except Exception as exc:
            return EvaluatorScore(
                summary_fidelity=0,
                inference_quality=0,
                classification_correctness=0,
                tool_recommendations=0,
                final_score=0,
                notes=f"evaluator call failed: {type(exc).__name__}: {exc}",
                raw_response="",
            )
        text = resp.text or ""
        return self._parse_score(text)

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
