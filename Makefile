.PHONY: check test lint probes install install-embeddings

# One-shot quality gate: lint + tests + fast probe smoke (fake LLM).
check: lint test probes

install:
	pip install -e ".[dev]"

install-embeddings:
	pip install -e ".[dev,embeddings]"

lint:
	ruff check .
	black --check .

test:
	pytest tests/ -q

# Fast probe mechanics check (no provider tokens; judge assertions skip).
probes:
	python -m evaluation.ralph_v2 --probes evaluation/probes/ --fake-llm --threshold 0.0

# Full evaluation against a real provider + semantic judge (needs keys).
# Example:
#   make eval CONFIG=sherlock.live.yaml JUDGE=anthropic:claude-haiku-4-5
eval:
	python -m evaluation.ralph_v2 --probes evaluation/probes/ \
		--config $(CONFIG) --judge-model $(JUDGE) \
		--report logs/probe_v050.json
