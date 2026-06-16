from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from bayes_llm.metrics import (
    gaussian_nll,
    martingale_prediction_stats,
    nearest_neighbor_mean,
    oracle_distance_metrics,
    permutation_gap,
    scalar_metrics,
)
from bayes_llm.models import ModelConfig, make_model
from bayes_llm.oracles import oracle_for_batch
from bayes_llm.tasks import (
    ChangepointRegression,
    ExchangeableLinearRegression,
    GaussianProcessRegression,
    HeteroscedasticTemporalRegression,
    RandomWalkLinearRegression,
    RegressionTask,
    make_torch_generator,
)
from bayes_llm.train import resolve_device


_MISSING_MATPLOTLIB_REPORTED = False


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def make_task(
    name: str,
    *,
    tau: float,
    sigma: float,
    q: float,
    min_segment: int,
    gp_amplitude: float,
    gp_lengthscale: float,
) -> RegressionTask:
    normalized = name.replace("-", "_").lower()
    if normalized in {"exchangeable", "linear", "exchangeable_linear"}:
        return ExchangeableLinearRegression(tau=tau, sigma=sigma)
    if normalized in {"random_walk", "drift", "drifting"}:
        return RandomWalkLinearRegression(tau=tau, sigma=sigma, q=q)
    if normalized in {"changepoint", "change_point"}:
        return ChangepointRegression(tau=tau, sigma=sigma, min_segment=min_segment)
    if normalized in {"heteroscedastic", "temporal_noise"}:
        return HeteroscedasticTemporalRegression(tau=tau)
    if normalized in {"gp", "gp_rbf"}:
        return GaussianProcessRegression(kernel="rbf", amplitude=gp_amplitude, lengthscale=gp_lengthscale, sigma=sigma)
    if normalized in {"gp_matern12", "gp_exponential"}:
        return GaussianProcessRegression(kernel="matern12", amplitude=gp_amplitude, lengthscale=gp_lengthscale, sigma=sigma)
    if normalized in {"gp_matern32"}:
        return GaussianProcessRegression(kernel="matern32", amplitude=gp_amplitude, lengthscale=gp_lengthscale, sigma=sigma)
    raise ValueError(f"unknown task: {name}")


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row[key] for row in rows if key in row) / max(1, sum(key in row for row in rows)))
        for key in keys
    }


def load_model_from_run(run_dir: Path, device: torch.device, max_context: int) -> tuple[str, torch.nn.Module]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_name = config.get("model", "qwen_adaptive")
    model = make_model(
        ModelConfig(
            model=model_name,
            x_dim=int(config.get("x_dim", 8)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            n_heads=int(config.get("n_heads", 4)),
            n_layers=int(config.get("n_layers", 2)),
            dropout=float(config.get("dropout", 0.0)),
            max_context=max_context,
            hf_model_id=str(config.get("hf_model_id", "Qwen/Qwen2.5-0.5B")),
            freeze_backbone=bool(config.get("freeze_backbone", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            prompt_precision=int(config.get("prompt_precision", 4)),
            max_prompt_length=int(config.get("max_prompt_length", 1024)),
            prompt_ordered=bool(config.get("prompt_ordered", True)),
            prompt_style=str(config.get("prompt_style", "compact")),
            task_name=str(config.get("task", "")),
        )
    ).to(device)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    state = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return f"{run_dir.name}:{model_name}", model


@torch.no_grad()
def evaluate_references(
    task: RegressionTask,
    args: argparse.Namespace,
    context: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> list[dict[str, float]]:
    oracle_rows = []
    nn_rows = []
    for _ in range(args.n_batches):
        batch = task.sample(args.batch_size, context, args.x_dim, device=device, generator=generator)
        oracle = oracle_for_batch(
            batch.context_x,
            batch.context_y,
            batch.query_x,
            args.task,
            tau=args.tau,
            sigma=args.sigma,
            q=args.q,
            gp_amplitude=args.gp_amplitude,
            gp_lengthscale=args.gp_lengthscale,
        )
        if oracle is not None:
            oracle_rows.append(
                {
                    "mse": float((oracle.mean - batch.query_y).square().mean().detach().cpu()),
                    "nll": float(gaussian_nll(oracle.mean, oracle.variance, batch.query_y).detach().cpu()),
                    "mean_variance": float(oracle.variance.mean().detach().cpu()),
                }
            )
        nn_mean = nearest_neighbor_mean(batch.context_x, batch.context_y, batch.query_x)
        nn_rows.append({"mse": float((nn_mean - batch.query_y).square().mean().detach().cpu())})

    rows = []
    if oracle_rows:
        rows.append({"model": "analytical_oracle", "context": float(context), **average(oracle_rows)})
    rows.append({"model": "nearest_neighbor", "context": float(context), **average(nn_rows)})
    return rows


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    label: str,
    task: RegressionTask,
    args: argparse.Namespace,
    context: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> dict[str, float]:
    rows = []
    for _ in range(args.n_batches):
        batch = task.sample(args.batch_size, context, args.x_dim, device=device, generator=generator)
        pred = model(batch.context_x, batch.context_y, batch.query_x)
        row = scalar_metrics(pred, batch.query_y)
        oracle = oracle_for_batch(
            batch.context_x,
            batch.context_y,
            batch.query_x,
            args.task,
            tau=args.tau,
            sigma=args.sigma,
            q=args.q,
            gp_amplitude=args.gp_amplitude,
            gp_lengthscale=args.gp_lengthscale,
        )
        if oracle is not None:
            row.update(oracle_distance_metrics(pred, oracle.mean, oracle.variance))
        if args.n_permutations > 1:
            row.update(permutation_gap(model, batch, n_permutations=args.n_permutations))
        if args.task.replace("-", "_").lower() in {"exchangeable", "linear", "exchangeable_linear"}:
            row.update(martingale_prediction_stats(model, batch))
        rows.append(row)
    return {"model": label, "context": float(context), **average(rows)}


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    global _MISSING_MATPLOTLIB_REPORTED
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if not _MISSING_MATPLOTLIB_REPORTED:
            print("matplotlib is not installed; wrote CSV and skipped PNG plots")
            _MISSING_MATPLOTLIB_REPORTED = True
        return

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if metric in row and row[metric] != "":
            by_model.setdefault(str(row["model"]), []).append(row)
    if not by_model:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, model_rows in sorted(by_model.items()):
        ordered = sorted(model_rows, key=lambda item: float(item["context"]))
        ax.plot(
            [float(item["context"]) for item in ordered],
            [float(item[metric]) for item in ordered],
            marker="o",
            label=model,
        )
    ax.set_xlabel("number of demonstrations")
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot analytical and model learning curves.")
    parser.add_argument("--task", default="exchangeable")
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--contexts", default="1,2,4,8,16,32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--n-permutations", type=int, default=8)
    parser.add_argument("--x-dim", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--gp-amplitude", type=float, default=1.0)
    parser.add_argument("--gp-lengthscale", type=float, default=1.0)
    parser.add_argument("--min-segment", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/plots"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    contexts = parse_int_list(args.contexts)
    if args.task.replace("-", "_").lower() in {"changepoint", "change_point"}:
        contexts = [context for context in contexts if context >= 2 * args.min_segment]
    if not contexts:
        raise ValueError("no valid context counts remain after task constraints")
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = make_torch_generator(args.seed, device) if device.type in {"cpu", "cuda"} else None
    task = make_task(
        args.task,
        tau=args.tau,
        sigma=args.sigma,
        q=args.q,
        min_segment=args.min_segment,
        gp_amplitude=args.gp_amplitude,
        gp_lengthscale=args.gp_lengthscale,
    )

    rows: list[dict[str, float | str]] = []
    for context in contexts:
        rows.extend(evaluate_references(task, args, context, device, generator))

    loaded_models = [
        load_model_from_run(Path(run_dir), device, max_context=max(contexts))
        for run_dir in args.run_dir
    ]
    for label, model in loaded_models:
        for context in contexts:
            rows.append(evaluate_model(model, label, task, args, context, device, generator))

    write_csv(args.out_dir / "eval_curves.csv", rows)
    for metric in [
        "mse",
        "nll",
        "oracle_kl",
        "oracle_mean_mse",
        "permutation_mean_var",
        "permutation_mean_range",
        "gate_alpha_mean",
        "mean_variance",
        "martingale_abs_increment",
        "martingale_path_variance",
    ]:
        plot_metric(rows, metric, args.out_dir / f"{metric}_vs_demonstrations.png")


if __name__ == "__main__":
    main()
