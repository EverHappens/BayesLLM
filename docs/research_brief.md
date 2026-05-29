# Research brief: Adaptive Set / Sequence In-Context Learning

## Motivation

Recent papers such as:

- Rethinking Invariance in In-Context Learning
- Set-LLM: A Permutation-Invariant LLM
- One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation
- Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective

study ways to reduce position or order sensitivity in LLMs. Their common premise is that many prompts contain unordered elements, but LLMs serialize these elements into sequences, creating arbitrary order bias.

Our project argues that this is only half of the story.

Permutation invariance is not universally desirable. It is correct when the prompt examples are exchangeable, but wrong when the prompt examples come from an ordered data-generating process.

The main research question:

Can an ICL model learn when to treat a context as an exchangeable set and when to treat it as an ordered sequence?

## Core thesis

Permutation invariance is a posterior symmetry induced by exchangeability, not a universal architectural virtue.

For exchangeable data:

p(y_* | D, x_*) = p(y_* | pi(D), x_*)

For ordered data, generally:

p(y_{n+1} | z_1, ..., z_n, x_{n+1})
!=
p(y_{n+1} | z_{pi(1)}, ..., z_{pi(n)}, x_{n+1})

where z_i = (x_i, y_i).

Therefore, hard-coded invariant models can be Bayes-optimal under exchangeable priors but incur excess risk under ordered priors.

## Proposed architecture

The model has:

1. Shared local example encoder
2. Set / invariant branch
3. Ordered / sequence branch
4. Adaptive gate
5. Mixture prediction

Input:

D = {(x_i, y_i)}_{i=1}^n, test query x_*

Local encoder:

e_i = phi(x_i, y_i)
e_* = phi_*(x_*)

Set branch:

h_set = SetEncoder(e_1, ..., e_n)

This branch must be permutation-invariant with respect to example order. It should use no between-example positional encoding, symmetric attention or DeepSets-style aggregation, and invariant readout.

Ordered branch:

h_ord = OrderEncoder(e_1, ..., e_n)

This branch is position-aware. It may use relative position bias, causal masking, recency bias, or temporal attention.

Predictions:

p_set(y_* | D, x_*) = Head_set(h_set, e_*)
p_ord(y_* | D, x_*) = Head_ord(h_ord, e_*)

Gate:

alpha(D) = sigmoid(g([h_set, h_ord, e_*]))

Final prediction:

p(y_* | D, x_*) =
alpha(D) * p_set(y_* | D, x_*) +
(1 - alpha(D)) * p_ord(y_* | D, x_*)

For regression, use a mixture of Gaussian predictions or mix means/variances carefully.

## Important architecture distinction

The local token/example encoder may preserve token order inside each example.

Example order across demonstrations should be removed only in the set branch.

The set branch should guarantee invariance structurally, not merely learn it.

A Transformer without positional encodings is usually permutation-equivariant, not invariant. Use invariant pooling/readout to obtain invariant prediction.

## Synthetic task families

### 1. Exchangeable linear regression

w ~ N(0, tau^2 I)
x_i ~ N(0, I)
y_i = w^T x_i + eps_i
eps_i ~ N(0, sigma^2)

Examples are exchangeable. Bayesian posterior predictive is invariant to example order.

Expected behavior:
- set branch should perform well
- ordered branch may perform but may show spurious order sensitivity
- adaptive model should put alpha close to 1

### 2. Random-walk / drifting linear regression

w_1 ~ N(0, tau^2 I)
w_t = w_{t-1} + eta_t
eta_t ~ N(0, q I)
y_t = w_t^T x_t + eps_t

Examples are ordered. Recent examples should matter more. Permuting examples changes the implied trajectory.

Expected behavior:
- set-only model should fail or underperform
- ordered branch should perform well
- adaptive model should put alpha close to 0

### 3. Changepoint regression

For t <= tau:
y_t = w_1^T x_t + eps_t

For t > tau:
y_t = w_2^T x_t + eps_t

Examples before and after changepoint belong to different regimes.

Expected behavior:
- set-only model averages incompatible regimes
- ordered model detects recent regime
- adaptive gate should favor ordered branch

### 4. Heteroscedastic temporal noise

y_t = w^T x_t + eps_t
eps_t ~ N(0, sigma_t^2)

sigma_t may increase or decrease over time.

Expected behavior:
- model should learn whether recent examples are more or less reliable
- useful for testing calibrated order sensitivity, not just recency bias

## Metrics

Use more than accuracy/MSE.

1. MSE or NLL
2. Permutation gap:

Var_pi[p_theta(y_* | pi(D), x_*)]

For exchangeable tasks this should be near zero.

For ordered tasks this should not necessarily be zero. Instead compare to oracle order sensitivity.

3. Oracle distance:

KL(p_oracle(y_* | D, x_*) || p_model(y_* | D, x_*))

or Gaussian NLL / posterior mean error.

4. Gate behavior:

alpha close to 1 on exchangeable tasks, close to 0 on ordered tasks.

5. Calibration:

For probabilistic regression, compare predictive variance to oracle variance.

6. Martingale-style Bayesian checks:

For exchangeable data, Bayesian predictive beliefs should satisfy martingale consistency as more observations are revealed. Track predictive drift across growing prefixes and compare it against the oracle and model baselines.

## Baselines

Implement:

1. Set-only model
2. Ordered-only model
3. Adaptive two-branch model
4. Oracle-selected branch, if branch labels known
5. Bayesian oracle
6. Optional PFN baseline

## Theoretical targets

Theorem 1:
If the data-generating prior is exchangeable, the Bayesian posterior predictive is permutation-invariant.

Theorem 2:
For an ordered state-space prior, there exist contexts D and permutations pi such that the posterior predictive changes under pi.

Theorem 3:
There exist ordered data-generating processes for which every permutation-invariant predictor has nonzero excess risk relative to the Bayes predictor.

## Main paper framing

Do not claim:
"We make ICL position-invariant."

Claim:
"We study when prompt order is nuisance and when it is evidence."

The core contribution is an adaptive model and diagnostic framework for distinguishing serialization artifacts from meaningful order.
