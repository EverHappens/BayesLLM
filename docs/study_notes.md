# Study notes: Bayesian properties, invariance, and order

This scaffold tests whether meta-ICL regressors preserve Bayesian properties when the data-generating prior is exchangeable versus ordered.

The exchangeable baseline follows the Bayesian ICL view used in work such as "In-Context Learning through the Bayesian Prism", "What and How does In-Context Learning Learn?", and "What learning algorithm is in-context learning?". For Gaussian linear regression, the posterior predictive is available in closed form and must be invariant to paired permutations of context examples.

The ordered tasks intentionally violate exchangeability. In random-walk regression, the latent weight vector evolves over time, so recent observations carry different information than old observations. In changepoint regression, a suffix of the context can belong to a different regime from the prefix. For these priors, a forced invariant predictor can erase evidence that a Bayesian state-space or changepoint posterior would use.

The primary experiment models use a small pretrained Hugging Face/Qwen decoder backbone, defaulting to `Qwen/Qwen2.5-0.5B`. Synthetic tensors are mapped into learned numeric prompt embeddings and passed to the pretrained backbone through `inputs_embeds`; the transformer weights are loaded from `AutoModel.from_pretrained(...)`, not initialized from scratch.

The pretrained ordered baseline uses Qwen over the serialized example sequence. The pretrained set baseline applies the same Qwen backbone per example with shared local positions and invariant pooling. The pretrained adaptive model shares the Qwen backbone across ordered and set paths, then learns a gate between their Gaussian predictions. Scratch transformer models remain available only as controlled ablations.

The martingale perspective paper, "Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective" by Falck, Wang, and Holmes, motivates an additional diagnostic: for exchangeable data, Bayesian predictive beliefs should satisfy martingale-style consistency as more observations are revealed. The implemented `martingale_prediction_stats` reports sample-path predictive drift across growing prefixes. It is a practical diagnostic rather than a complete reproduction of the paper's LLM experiments.

Primary metrics:

- MSE and Gaussian NLL against sampled query labels.
- Permutation gap of model predictions under paired context permutations.
- Oracle mean/variance distance where an oracle is implemented.
- Adaptive gate statistics.
- Martingale-style predictive drift on exchangeable tasks.

Expected qualitative outcomes:

- Exchangeable regression: Bayesian oracle and set branch should have near-zero permutation gap; adaptive gate should learn to favor the set branch.
- Random-walk regression: order-aware model should outperform set-only; adaptive gate should learn to favor the ordered branch.
- Changepoint regression: order-aware and adaptive models should handle recent-regime evidence better than set-only.
