#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif [[ -x "$HOME/.pyenv/versions/3.11.12/bin/python" ]]; then
  PYTHON_BIN="$HOME/.pyenv/versions/3.11.12/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" -B -m unittest discover -s tests

"$PYTHON_BIN" -B -m bayes_llm.train \
  --task exchangeable \
  --model adaptive \
  --device cpu \
  --steps 2 \
  --batch-size 4 \
  --context 4 \
  --x-dim 3 \
  --hidden-dim 16 \
  --n-heads 4 \
  --n-layers 1 \
  --log-every 1 \
  --eval-every 2 \
  --n-eval-batches 1 \
  --n-permutations 2 \
  --out-dir /tmp/bayes_llm_smoke
