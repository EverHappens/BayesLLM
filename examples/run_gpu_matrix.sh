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

DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-bf16}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CONTEXT="${CONTEXT:-32}"
X_DIM="${X_DIM:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-2}"
OUT_ROOT="${OUT_ROOT:-artifacts/gpu_matrix}"
TASKS="${TASKS:-exchangeable random_walk changepoint}"
MODELS="${MODELS:-regular set adaptive}"
COMPILE="${COMPILE:-0}"

compile_flag=()
if [[ "$COMPILE" == "1" ]]; then
  compile_flag=(--compile)
fi

for task in $TASKS; do
  for model in $MODELS; do
    out_dir="${OUT_ROOT}/${task}_${model}"
    echo "Running task=${task} model=${model} out_dir=${out_dir}"
    "$PYTHON_BIN" -m bayes_llm.train \
      --task "$task" \
      --model "$model" \
      --device "$DEVICE" \
      --amp "$AMP" \
      --steps "$STEPS" \
      --batch-size "$BATCH_SIZE" \
      --context "$CONTEXT" \
      --x-dim "$X_DIM" \
      --hidden-dim "$HIDDEN_DIM" \
      --n-heads "$N_HEADS" \
      --n-layers "$N_LAYERS" \
      --log-every 100 \
      --eval-every 500 \
      --n-eval-batches 8 \
      --n-permutations 8 \
      --out-dir "$out_dir" \
      "${compile_flag[@]}"
  done
done
