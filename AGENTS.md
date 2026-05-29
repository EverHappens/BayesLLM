# Project instructions for Codex

This repository is for a research project on adaptive in-context learning under exchangeable and ordered data-generating processes.

Before implementing anything, read:

- `docs/research_brief.md`

Core research goal:
Build and evaluate an adaptive ICL architecture with two inductive-bias pathways:

1. A permutation-invariant set branch for exchangeable prompts.
2. A position-aware ordered branch for temporal / non-exchangeable prompts.
3. A learned gate that combines the two predictions.

Do not frame the project as merely "position-invariant ICL." The core thesis is:

Permutation invariance is statistically correct only under exchangeability. For ordered data-generating processes such as drift, changepoints, or time-varying noise, forced invariance can be anti-Bayesian.

Implementation priorities:

1. Start with synthetic Bayesian tasks where the oracle posterior is computable.
2. Implement exchangeable noisy linear regression.
3. Implement ordered random-walk linear regression.
4. Implement changepoint regression.
5. Implement three model baselines:
   - set-only model
   - ordered-only model
   - adaptive two-branch model
6. Evaluate MSE, NLL, permutation gap, and KL or distance to oracle when available.

Keep code modular and experiment configs reproducible.