# Bayesian ICL Experiments

This is a minimal PyTorch scaffold for testing adaptive in-context learning under exchangeable and ordered regression priors.

The project is GPU-ready, but the included tests are CPU smoke checks. Install a CUDA-enabled PyTorch build using the command recommended by the PyTorch install selector for your GPU/driver, then install this package in editable mode with Hugging Face support:

```bash
pip install -e ".[hf,plots]"
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run a small GPU experiment:

```bash
python -m bayes_llm.train \
  --task random_walk \
  --model qwen_adaptive \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device cuda \
  --amp bf16 \
  --steps 5000 \
  --batch-size 32 \
  --context 16 \
  --x-dim 8 \
  --out-dir artifacts/random_walk_adaptive
```

Primary pretrained model names are `qwen`, `qwen_set`, and `qwen_adaptive`. Scratch ablation names are still available as `regular`, `set`, and `adaptive`. Useful task names are `exchangeable`, `random_walk`, `changepoint`, and `heteroscedastic`.

Metrics are written to `metrics.jsonl`; the final checkpoint and config are written to the run directory.

Plot analytical references and trained model learning curves:

```bash
RUN_DIRS="artifacts/random_walk_adaptive" TASK=random_walk \
  bash examples/plot_learning_curves.sh
```

For the full command reference, runnable scripts, and code explanations, read:

```bash
docs/runbook.md
```
