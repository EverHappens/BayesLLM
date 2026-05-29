from __future__ import annotations

from collections.abc import Mapping

import torch

from bayes_llm.tasks import RegressionBatch


def mse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (mean - target).square().mean()


def gaussian_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = variance.clamp_min(1e-12)
    return 0.5 * (torch.log(2.0 * torch.pi * var) + (target - mean).square() / var).mean()


def gaussian_kl(
    p_mean: torch.Tensor,
    p_variance: torch.Tensor,
    q_mean: torch.Tensor,
    q_variance: torch.Tensor,
) -> torch.Tensor:
    p_var = p_variance.clamp_min(1e-12)
    q_var = q_variance.clamp_min(1e-12)
    return 0.5 * (torch.log(q_var / p_var) + (p_var + (p_mean - q_mean).square()) / q_var - 1.0).mean()


def oracle_distance_metrics(
    pred: Mapping[str, torch.Tensor],
    oracle_mean: torch.Tensor,
    oracle_variance: torch.Tensor,
) -> dict[str, float]:
    mean = pred["mean"]
    variance = pred["variance"]
    return {
        "oracle_mean_mse": float((mean - oracle_mean).square().mean().detach().cpu()),
        "oracle_var_mse": float((variance - oracle_variance).square().mean().detach().cpu()),
        "oracle_kl": float(gaussian_kl(oracle_mean, oracle_variance, mean, variance).detach().cpu()),
    }


@torch.no_grad()
def permutation_gap(
    model: torch.nn.Module,
    batch: RegressionBatch,
    *,
    n_permutations: int = 8,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    means = []
    variances = []
    for _ in range(n_permutations):
        perm = torch.randperm(batch.context_x.shape[1], device=batch.context_x.device)
        out = model(batch.context_x[:, perm], batch.context_y[:, perm], batch.query_x)
        means.append(out["mean"])
        variances.append(out["variance"])
    mean_stack = torch.stack(means, dim=0)
    var_stack = torch.stack(variances, dim=0)
    if was_training:
        model.train()
    return {
        "permutation_mean_var": float(mean_stack.var(dim=0, unbiased=False).mean().detach().cpu()),
        "permutation_var_var": float(var_stack.var(dim=0, unbiased=False).mean().detach().cpu()),
        "permutation_mean_range": float((mean_stack.max(dim=0).values - mean_stack.min(dim=0).values).mean().detach().cpu()),
    }


@torch.no_grad()
def martingale_prediction_stats(
    model: torch.nn.Module,
    batch: RegressionBatch,
    *,
    min_context: int = 1,
) -> dict[str, float]:
    """Sample-path diagnostics for predictive drift as exchangeable context grows."""

    was_training = model.training
    model.eval()
    n_context = batch.context_x.shape[1]
    if n_context <= min_context:
        return {
            "martingale_mean_increment": 0.0,
            "martingale_abs_increment": 0.0,
            "martingale_path_variance": 0.0,
        }

    path = []
    for k in range(min_context, n_context + 1):
        out = model(batch.context_x[:, :k], batch.context_y[:, :k], batch.query_x)
        path.append(out["mean"])
    path_tensor = torch.stack(path, dim=0)
    increments = path_tensor[1:] - path_tensor[:-1]
    if was_training:
        model.train()
    return {
        "martingale_mean_increment": float(increments.mean().detach().cpu()),
        "martingale_abs_increment": float(increments.abs().mean().detach().cpu()),
        "martingale_path_variance": float(path_tensor.var(dim=0, unbiased=False).mean().detach().cpu()),
    }


def scalar_metrics(pred: Mapping[str, torch.Tensor], target: torch.Tensor) -> dict[str, float]:
    values = {
        "mse": mse(pred["mean"], target),
        "nll": gaussian_nll(pred["mean"], pred["variance"], target),
        "mean_variance": pred["variance"].mean(),
    }
    if "gate_alpha" in pred:
        values["gate_alpha_mean"] = pred["gate_alpha"].mean()
        values["gate_alpha_std"] = pred["gate_alpha"].std(unbiased=False)
    return {key: float(value.detach().cpu()) for key, value in values.items()}

