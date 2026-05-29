# Runbook

This project has one main training/evaluation entrypoint and two convenience scripts.

## Setup

Work from the project root:

```bash
cd /Users/ever/Desktop/work/codex/bayes_llm
```

Install a CUDA-enabled PyTorch build for your machine using the PyTorch install selector. Then install this package:

```bash
pip install -e .
```

If you do not install the package, prefix commands with:

```bash
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
```

## Scripts

### `examples/smoke_cpu.sh`

Runs the unit tests and a two-step CPU training job. Use this before launching GPU jobs:

```bash
bash examples/smoke_cpu.sh
```

It writes the tiny training run to `/tmp/bayes_llm_smoke`.

If your shell needs a specific Python executable:

```bash
PYTHON=/path/to/python bash examples/smoke_cpu.sh
```

### `examples/run_gpu_matrix.sh`

Runs the core task/model grid:

- tasks: `exchangeable`, `random_walk`, `changepoint`
- models: `regular`, `set`, `adaptive`

Default command:

```bash
bash examples/run_gpu_matrix.sh
```

If needed, select the Python executable explicitly:

```bash
PYTHON=/path/to/python bash examples/run_gpu_matrix.sh
```

Useful overrides:

```bash
STEPS=20000 BATCH_SIZE=1024 CONTEXT=64 AMP=bf16 COMPILE=1 \
  bash examples/run_gpu_matrix.sh
```

Run only one task/model pair:

```bash
TASKS="random_walk" MODELS="adaptive" OUT_ROOT="artifacts/random_walk_only" \
  bash examples/run_gpu_matrix.sh
```

Outputs go under `artifacts/gpu_matrix/<task>_<model>/` by default.

## Direct CLI

The underlying entrypoint is:

```bash
python -m bayes_llm.train
```

Example:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model adaptive \
  --device cuda \
  --amp bf16 \
  --compile \
  --steps 5000 \
  --batch-size 512 \
  --context 32 \
  --x-dim 8 \
  --hidden-dim 128 \
  --n-heads 4 \
  --n-layers 2 \
  --out-dir artifacts/random_walk_adaptive
```

Important flags:

- `--task`: `exchangeable`, `random_walk`, `changepoint`, or `heteroscedastic`.
- `--model`: `regular`, `set`, `set_llm`, or `adaptive`.
- `--device`: `cuda`, `cpu`, `mps`, or `auto`.
- `--amp`: `bf16`, `fp16`, or `off`; use `bf16` first on modern GPUs.
- `--compile`: enables `torch.compile`.
- `--context`: number of in-context examples.
- `--x-dim`: regression input dimension.
- `--tau`, `--sigma`, `--q`: prior, observation noise, and random-walk process variance.
- `--n-permutations`: number of context permutations for permutation-gap evaluation.

Each run writes:

- `config.json`: resolved command-line settings.
- `metrics.jsonl`: train/eval metrics, one JSON object per line.
- `checkpoint.pt`: final model and optimizer state.

## Code map

### `src/bayes_llm/tasks.py`

Defines synthetic data generators. Every generator returns a `RegressionBatch`:

```python
context_x: [batch, n_context, x_dim]
context_y: [batch, n_context]
query_x: [batch, x_dim]
query_y: [batch]
metadata: dict
```

The generators are:

- `ExchangeableLinearRegression`: samples one latent weight vector per prompt; paired permutations of examples should not matter.
- `RandomWalkLinearRegression`: latent weights drift over time; order is evidence.
- `ChangepointRegression`: context has old and new regimes; suffix evidence matters.
- `HeteroscedasticTemporalRegression`: noise changes over time.

### `src/bayes_llm/oracles.py`

Implements Bayesian reference predictors:

- closed-form Bayesian linear regression for exchangeable prompts
- Kalman-filter prediction for random-walk prompts
- changepoint mixture oracle by enumerating changepoint positions

These are used for oracle mean/variance distance and KL-style diagnostics.

### `src/bayes_llm/models.py`

Defines the model baselines:

- `PositionAwareTransformerRegressor`: regular ordered transformer with learned between-example positions.
- `SetLLMStyleInvariantRegressor`: Set-LLM-style invariant branch. It uses shared within-example positions and invariant pooling, so paired context permutations produce the same prediction.
- `AdaptiveTwoBranchRegressor`: combines set and ordered branches with a learned gate `gate_alpha`.

All models expose:

```python
out = model(context_x, context_y, query_x)
out["mean"]
out["variance"]
out["log_variance"]
```

The adaptive model also returns:

```python
out["set_mean"]
out["ordered_mean"]
out["gate_alpha"]
```

### `src/bayes_llm/metrics.py`

Computes:

- MSE
- Gaussian NLL
- permutation gap
- oracle mean/variance distance
- Gaussian KL from oracle to model
- martingale-style predictive drift on exchangeable tasks

### `src/bayes_llm/train.py`

Creates the task, model, optimizer, AMP context, training loop, periodic evaluation, metric logging, and checkpoint writing.

The training loop is intentionally synthetic-data native: each batch is generated directly on the selected device when possible, avoiding a CPU dataloader bottleneck for GPU experiments.

## Suggested first runs

First verify exchangeable invariance:

```bash
python -m bayes_llm.train \
  --task exchangeable \
  --model set \
  --device cuda \
  --amp bf16 \
  --steps 3000 \
  --batch-size 512 \
  --context 32 \
  --out-dir artifacts/exchangeable_set
```

Then compare against the ordered baseline on drift:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model regular \
  --device cuda \
  --amp bf16 \
  --steps 3000 \
  --batch-size 512 \
  --context 32 \
  --out-dir artifacts/random_walk_regular
```

Then run the adaptive model:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model adaptive \
  --device cuda \
  --amp bf16 \
  --steps 3000 \
  --batch-size 512 \
  --context 32 \
  --out-dir artifacts/random_walk_adaptive
```

Interpretation targets:

- On `exchangeable`, `set` should have near-zero permutation gap.
- On `random_walk` and `changepoint`, `regular` should have useful order sensitivity.
- On mixed comparisons, `adaptive` should learn higher `gate_alpha` for exchangeable data and lower `gate_alpha` for ordered data.
