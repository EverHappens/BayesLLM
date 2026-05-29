# Bayesian ICL Experiments

This is a minimal PyTorch scaffold for testing adaptive in-context learning under exchangeable and ordered regression priors.

The project is GPU-ready, but the included tests are CPU smoke checks. Install a CUDA-enabled PyTorch build using the command recommended by the PyTorch install selector for your GPU/driver, then install this package in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run a small GPU experiment:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model adaptive \
  --device cuda \
  --amp bf16 \
  --steps 5000 \
  --batch-size 512 \
  --context 32 \
  --x-dim 8 \
  --out-dir artifacts/random_walk_adaptive
```

Useful model names are `regular`, `set`, and `adaptive`. Useful task names are `exchangeable`, `random_walk`, `changepoint`, and `heteroscedastic`.

Metrics are written to `metrics.jsonl`; the final checkpoint and config are written to the run directory.

For the full command reference, runnable scripts, and code explanations, read:

```bash
docs/runbook.md
```

