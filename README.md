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

Preview the human-readable ICL prompt corresponding to a sampled synthetic batch:

```bash
python -m bayes_llm.prompting --task exchangeable --context 4 --x-dim 3
```

The `qwen_text` path tokenizes this prompt format. The `qwen`, `qwen_set`, and `qwen_adaptive` architectural ablations instead use learned numeric prompt embeddings via `inputs_embeds`.

Run a small GPU experiment:

```bash
python -m bayes_llm.train \
  --task gp_rbf \
  --model qwen_text \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device cuda \
  --amp bf16 \
  --freeze-backbone \
  --steps 5000 \
  --batch-size 32 \
  --context 16 \
  --x-dim 8 \
  --out-dir artifacts/gp_rbf_qwen_text
```

Primary model names are `qwen_text` for full-prompt tokenized function-learning experiments, `qwen_set` for the Set-LLM-inspired masked embedding architecture, and `qwen_adaptive` for the hybrid model. Scratch ablation names are still available as `regular`, `set`, and `adaptive`. Useful task names include `exchangeable`, `random_walk`, `changepoint`, `heteroscedastic`, `gp_rbf`, `gp_matern12`, and `gp_matern32`.

Metrics are written to `metrics.jsonl`; the final checkpoint and config are written to the run directory.

Plot analytical references and trained model learning curves:

```bash
RUN_DIRS="artifacts/random_walk_adaptive" TASK=random_walk \
  bash examples/plot_learning_curves.sh
```

Visualize how a trained checkpoint approximates a concrete function such as cosine:

```bash
RUN_DIRS="artifacts/gp_rbf_qwen_text" FUNCTION=cosine \
  bash examples/plot_function_fit.sh
```

For the full command reference, runnable scripts, and code explanations, read:

```bash
docs/runbook.md
```
