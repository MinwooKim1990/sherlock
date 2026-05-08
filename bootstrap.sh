#!/usr/bin/env bash
set -euo pipefail

# Sherlock build sandbox bootstrap.
# Per OPERATIONS.md § 1.3, with one amendment:
# python3.12 is preferred (3.13/3.11/3 fall back) — chromadb and
# sentence-transformers wheels are not yet guaranteed on 3.14.

PY=""
for cand in python3.12 python3.13 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "Python 3.11+ required, none found" >&2
    exit 1
fi

echo "Using interpreter: $PY ($($PY --version))"

# Create venv if absent
if [ ! -d ".venv" ]; then
    "$PY" -m venv .venv
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
    pip install -e .
fi

# Sanity
python -c "import sys; print('Python:', sys.version)"

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
