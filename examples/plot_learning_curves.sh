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

TASK="${TASK:-exchangeable}"
DEVICE="${DEVICE:-cuda}"
CONTEXTS="${CONTEXTS:-1,2,4,8,16,32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_BATCHES="${N_BATCHES:-8}"
X_DIM="${X_DIM:-8}"
OUT_DIR="${OUT_DIR:-artifacts/plots/${TASK}}"

run_args=()
if [[ -n "${RUN_DIRS:-}" ]]; then
  for run_dir in $RUN_DIRS; do
    run_args+=(--run-dir "$run_dir")
  done
fi

"$PYTHON_BIN" -m bayes_llm.plot_curves \
  --task "$TASK" \
  --device "$DEVICE" \
  --contexts "$CONTEXTS" \
  --batch-size "$BATCH_SIZE" \
  --n-batches "$N_BATCHES" \
  --x-dim "$X_DIM" \
  --out-dir "$OUT_DIR" \
  "${run_args[@]}"

