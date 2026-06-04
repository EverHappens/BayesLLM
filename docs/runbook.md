# Runbook

This project has one main training/evaluation entrypoint and two convenience scripts.

## Setup

Work from the project root:

```bash
cd /Users/ever/Desktop/work/codex/bayes_llm
```

Install a CUDA-enabled PyTorch build for your machine using the PyTorch install selector. Then install this package with Hugging Face and plotting support:

```bash
pip install -e ".[hf,plots]"
```

If you do not install the package, prefix commands with:

```bash
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
```

## Scripts

### `examples/smoke_cpu.sh`

Runs the unit tests and a two-step CPU training job. This smoke test uses the tiny scratch model path so it does not download Qwen on machines without `transformers`.

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
- models: `qwen`, `qwen_set`, `qwen_adaptive`

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
STEPS=20000 BATCH_SIZE=64 CONTEXT=32 AMP=bf16 COMPILE=1 \
  bash examples/run_gpu_matrix.sh
```

Run only one task/model pair:

```bash
TASKS="random_walk" MODELS="qwen_adaptive" OUT_ROOT="artifacts/random_walk_only" \
  bash examples/run_gpu_matrix.sh
```

Select a different small Hugging Face backbone:

```bash
HF_MODEL_ID="Qwen/Qwen3-0.6B" TASKS="exchangeable" MODELS="qwen" \
  bash examples/run_gpu_matrix.sh
```

By default, `examples/run_gpu_matrix.sh` sets `FREEZE_BACKBONE=1`, so it trains only numeric adapters, regression heads, and the adaptive gate. Set `FREEZE_BACKBONE=0` to fine-tune the whole pretrained model.

### `examples/plot_learning_curves.sh`

Evaluates analytical references and any trained checkpoints across a grid of demonstration counts, then writes CSV and PNG files:

```bash
RUN_DIRS="artifacts/gpu_matrix/exchangeable_qwen_set artifacts/gpu_matrix/exchangeable_qwen_adaptive" \
TASK=exchangeable \
  bash examples/plot_learning_curves.sh
```

It produces `eval_curves.csv` plus plots such as:

- `mse_vs_demonstrations.png`
- `nll_vs_demonstrations.png`
- `oracle_kl_vs_demonstrations.png`
- `permutation_mean_var_vs_demonstrations.png`
- `gate_alpha_mean_vs_demonstrations.png`
- `martingale_abs_increment_vs_demonstrations.png`

You can also run analytical references only:

```bash
TASK=gp_rbf DEVICE=cpu bash examples/plot_learning_curves.sh
```

Training outputs from `examples/run_gpu_matrix.sh` go under `artifacts/gpu_matrix/<task>_<model>/` by default. Plot outputs from `examples/plot_learning_curves.sh` go under `artifacts/plots/<task>/` by default.

## Direct CLI

The underlying entrypoint is:

```bash
python -m bayes_llm.train
```

Example:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model qwen_adaptive \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device cuda \
  --amp bf16 \
  --compile \
  --steps 5000 \
  --batch-size 32 \
  --context 16 \
  --x-dim 8 \
  --freeze-backbone \
  --out-dir artifacts/random_walk_adaptive
```

Important flags:

- `--task`: `exchangeable`, `random_walk`, `changepoint`, `heteroscedastic`, `gp_rbf`, `gp_matern12`, or `gp_matern32`.
- `--model`: use `qwen`, `qwen_set`, or `qwen_adaptive` for pretrained experiments; `qwen_deepset`, `regular`, `set`, and `adaptive` are ablations.
- `--hf-model-id`: Hugging Face model id for pretrained backbones; default is `Qwen/Qwen2.5-0.5B`.
- `--freeze-backbone`: freeze the pretrained Qwen/HF transformer and train only adapters/heads/gate.
- `--device`: `cuda`, `cpu`, `mps`, or `auto`.
- `--amp`: `bf16`, `fp16`, or `off`; use `bf16` first on modern GPUs.
- `--compile`: enables `torch.compile`.
- `--context`: number of in-context examples.
- `--x-dim`: regression input dimension.
- `--tau`, `--sigma`, `--q`: prior, observation noise, and random-walk process variance.
- `--gp-amplitude`, `--gp-lengthscale`: GP function-prior hyperparameters.
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

- `PretrainedOrderedRegressor`: ordered pretrained Qwen/HF decoder backbone, selected by `--model qwen`.
- `PretrainedSetRegressor`: Set-LLM-style pretrained Qwen/HF backbone using SetPE positions and a paper-style SetMask, selected by `--model qwen_set`.
- `PretrainedAdaptiveRegressor`: shared pretrained Qwen/HF backbone with set and ordered paths plus `gate_alpha`, selected by `--model qwen_adaptive`.
- `PretrainedDeepSetRegressor`, `PositionAwareTransformerRegressor`, `SetLLMStyleInvariantRegressor`, and `AdaptiveTwoBranchRegressor`: ablations only.

The pretrained models do not serialize tensors into text. They feed learned numeric prompt embeddings through `AutoModel.from_pretrained(...)` via `inputs_embeds`, reusing the pretrained transformer weights while training small numeric adapters and regression heads.

For `qwen_set`, the numeric prompt is `[x1, y1, ..., xN, yN, query]`. Role embeddings distinguish predictor, target, and query tokens. SetPE reuses positions across demonstrations, and SetMask permits attention within each demonstration while allowing the query token to attend to all demonstrations. Set-LLM is used here as an architectural invariance mechanism, not as an ICL framework by itself; the ICL setup comes from our synthetic demonstrations plus held-out query. This is closer to the Set-LLM mechanism than the old DeepSets-style mean-pooled branch because the query representation is produced by the pretrained attention stack under the set mask.

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

### `src/bayes_llm/plot_curves.py`

Evaluates checkpoints across context counts and plots:

- analytical oracle MSE/NLL from closed-form Bayesian linear, Kalman, changepoint, or GP regression
- 1-nearest-neighbor MSE reference, following the kind of upper-reference curve used in function-learning work
- trained model MSE/NLL
- permutation gap
- oracle KL / oracle mean error
- gate behavior for adaptive models

### `src/bayes_llm/train.py`

Creates the task, model, optimizer, AMP context, training loop, periodic evaluation, metric logging, and checkpoint writing.

The training loop is intentionally synthetic-data native: each batch is generated directly on the selected device when possible, avoiding a CPU dataloader bottleneck for GPU experiments.

## Suggested first runs

First verify exchangeable invariance:

```bash
python -m bayes_llm.train \
  --task exchangeable \
  --model qwen_set \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device cuda \
  --amp bf16 \
  --steps 3000 \
  --batch-size 32 \
  --context 16 \
  --freeze-backbone \
  --out-dir artifacts/exchangeable_set
```

Then compare against the ordered baseline on drift:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model qwen \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device cuda \
  --amp bf16 \
  --steps 3000 \
  --batch-size 32 \
  --context 16 \
  --freeze-backbone \
  --out-dir artifacts/random_walk_regular
```

Then run the adaptive model:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model qwen_adaptive \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device cuda \
  --amp bf16 \
  --steps 3000 \
  --batch-size 32 \
  --context 16 \
  --freeze-backbone \
  --out-dir artifacts/random_walk_adaptive
```

Interpretation targets:

- On `exchangeable`, `qwen_set` should have near-zero permutation gap.
- On `random_walk` and `changepoint`, `qwen` should have useful order sensitivity.
- On mixed comparisons, `qwen_adaptive` should learn higher `gate_alpha` for exchangeable data and lower `gate_alpha` for ordered data.
