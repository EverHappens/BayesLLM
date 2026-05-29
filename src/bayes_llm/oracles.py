from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OraclePrediction:
    mean: torch.Tensor
    variance: torch.Tensor

    @property
    def log_variance(self) -> torch.Tensor:
        return torch.log(self.variance.clamp_min(1e-12))


def bayesian_linear_regression_predict(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    tau: float = 1.0,
    sigma: float = 0.1,
) -> OraclePrediction:
    """Closed-form posterior predictive for Gaussian linear regression."""

    batch_size, _, x_dim = context_x.shape
    dtype = context_x.dtype
    device = context_x.device
    if context_x.shape[1] == 0:
        mean = torch.zeros(batch_size, device=device, dtype=dtype)
        variance = sigma**2 + tau**2 * query_x.square().sum(dim=-1)
        return OraclePrediction(mean=mean, variance=variance.clamp_min(1e-12))

    eye = torch.eye(x_dim, device=device, dtype=dtype).expand(batch_size, x_dim, x_dim)
    xtx = torch.bmm(context_x.transpose(1, 2), context_x)
    precision = eye / (tau**2) + xtx / (sigma**2)
    rhs = torch.bmm(context_x.transpose(1, 2), context_y.unsqueeze(-1)).squeeze(-1) / (sigma**2)
    posterior_mean = torch.linalg.solve(precision, rhs.unsqueeze(-1)).squeeze(-1)
    solved_query = torch.linalg.solve(precision, query_x.unsqueeze(-1)).squeeze(-1)
    mean = (query_x * posterior_mean).sum(dim=-1)
    variance = sigma**2 + (query_x * solved_query).sum(dim=-1)
    return OraclePrediction(mean=mean, variance=variance.clamp_min(1e-12))


def random_walk_kalman_predict(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    tau: float = 1.0,
    sigma: float = 0.1,
    q: float = 0.05,
) -> OraclePrediction:
    """Kalman-filter posterior predictive for random-walk regression weights."""

    batch_size, n_context, x_dim = context_x.shape
    dtype = context_x.dtype
    device = context_x.device
    eye = torch.eye(x_dim, device=device, dtype=dtype).expand(batch_size, x_dim, x_dim)
    mean_state = torch.zeros(batch_size, x_dim, device=device, dtype=dtype)
    cov_state = (tau**2) * eye.clone()
    process_cov = q * eye

    for t in range(n_context):
        x_t = context_x[:, t]
        y_t = context_y[:, t]
        cov_x = torch.bmm(cov_state, x_t.unsqueeze(-1)).squeeze(-1)
        innovation_var = sigma**2 + (x_t * cov_x).sum(dim=-1).clamp_min(1e-12)
        innovation = y_t - (x_t * mean_state).sum(dim=-1)
        kalman_gain = cov_x / innovation_var.unsqueeze(-1)
        mean_state = mean_state + kalman_gain * innovation.unsqueeze(-1)
        cov_state = cov_state - torch.bmm(cov_x.unsqueeze(-1), cov_x.unsqueeze(1)) / innovation_var[:, None, None]
        cov_state = 0.5 * (cov_state + cov_state.transpose(1, 2))
        cov_state = cov_state + process_cov

    query_cov = torch.bmm(cov_state, query_x.unsqueeze(-1)).squeeze(-1)
    mean = (query_x * mean_state).sum(dim=-1)
    variance = sigma**2 + (query_x * query_cov).sum(dim=-1)
    return OraclePrediction(mean=mean, variance=variance.clamp_min(1e-12))


def _segment_log_marginal(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    tau: float,
    sigma: float,
) -> torch.Tensor:
    batch_size, n_obs, _ = x.shape
    device = x.device
    dtype = x.dtype
    if n_obs == 0:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    eye = torch.eye(n_obs, device=device, dtype=dtype).expand(batch_size, n_obs, n_obs)
    kernel = sigma**2 * eye + tau**2 * torch.bmm(x, x.transpose(1, 2))
    _sign, logdet = torch.linalg.slogdet(kernel)
    solved = torch.linalg.solve(kernel, y.unsqueeze(-1)).squeeze(-1)
    quad = (y * solved).sum(dim=-1)
    return -0.5 * (n_obs * torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype)) + logdet + quad)


def changepoint_oracle_predict(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    tau: float = 1.0,
    sigma: float = 0.1,
) -> OraclePrediction:
    """Uniform-changepoint oracle that mixes current-regime posterior predictives."""

    _, n_context, _ = context_x.shape
    means = []
    variances = []
    log_weights = []
    for cp in range(n_context + 1):
        prefix_x = context_x[:, :cp]
        prefix_y = context_y[:, :cp]
        suffix_x = context_x[:, cp:]
        suffix_y = context_y[:, cp:]
        log_weight = _segment_log_marginal(prefix_x, prefix_y, tau=tau, sigma=sigma)
        log_weight = log_weight + _segment_log_marginal(suffix_x, suffix_y, tau=tau, sigma=sigma)
        pred = bayesian_linear_regression_predict(suffix_x, suffix_y, query_x, tau=tau, sigma=sigma)
        log_weights.append(log_weight)
        means.append(pred.mean)
        variances.append(pred.variance)

    weights = torch.softmax(torch.stack(log_weights, dim=0), dim=0)
    mean_stack = torch.stack(means, dim=0)
    var_stack = torch.stack(variances, dim=0)
    mean = (weights * mean_stack).sum(dim=0)
    second_moment = (weights * (var_stack + mean_stack.square())).sum(dim=0)
    variance = (second_moment - mean.square()).clamp_min(1e-12)
    return OraclePrediction(mean=mean, variance=variance)


def oracle_for_batch(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    task_name: str,
    *,
    tau: float = 1.0,
    sigma: float = 0.1,
    q: float = 0.05,
) -> OraclePrediction | None:
    normalized = task_name.replace("-", "_").lower()
    if normalized in {"exchangeable", "linear", "exchangeable_linear"}:
        return bayesian_linear_regression_predict(context_x, context_y, query_x, tau=tau, sigma=sigma)
    if normalized in {"random_walk", "drift", "drifting"}:
        return random_walk_kalman_predict(context_x, context_y, query_x, tau=tau, sigma=sigma, q=q)
    if normalized in {"changepoint", "change_point"}:
        return changepoint_oracle_predict(context_x, context_y, query_x, tau=tau, sigma=sigma)
    return None
