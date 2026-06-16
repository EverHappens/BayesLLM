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

FUNCTION="${FUNCTION:-cosine}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-artifacts/function_fit/${FUNCTION}}"
DEMO_X="${DEMO_X:--3.1416,-1.5708,0,1.5708,3.1416}"
X_MIN="${X_MIN:--3.1416}"
X_MAX="${X_MAX:-3.1416}"
N_GRID="${N_GRID:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"

run_args=()
if [[ -n "${RUN_DIRS:-}" ]]; then
  for run_dir in $RUN_DIRS; do
    run_args+=(--run-dir "$run_dir")
  done
fi

"$PYTHON_BIN" -m bayes_llm.function_fit \
  --function "$FUNCTION" \
  --device "$DEVICE" \
  --demo-x "$DEMO_X" \
  --x-min "$X_MIN" \
  --x-max "$X_MAX" \
  --n-grid "$N_GRID" \
  --batch-size "$BATCH_SIZE" \
  --out-dir "$OUT_DIR" \
  "${run_args[@]}"

