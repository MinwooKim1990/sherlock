"""Evaluation harness — replay dummy conversation through Sherlock and score
against the gold standard via Gemini Flash Lite (cli-wrapper-unified)."""
from sherlock.evaluation.evaluator import GeminiEvaluator
from sherlock.evaluation.output_format import format_sherlock_output
from sherlock.evaluation.replay import replay_dummy_conversation

__all__ = ["GeminiEvaluator", "format_sherlock_output", "replay_dummy_conversation"]
