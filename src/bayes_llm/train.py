from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from bayes_llm.metrics import (
    gaussian_nll,
    martingale_prediction_stats,
    oracle_distance_metrics,
    permutation_gap,
    scalar_metrics,
)
from bayes_llm.models import ModelConfig, make_model
from bayes_llm.oracles import oracle_for_batch
from bayes_llm.tasks import (
    ChangepointRegression,
    ExchangeableLinearRegression,
    HeteroscedasticTemporalRegression,
    RandomWalkLinearRegression,
    RegressionTask,
    make_torch_generator,
)


def resolve_device(requested: str) -> torch.device:
    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
    return device


def make_task_from_args(args: argparse.Namespace) -> RegressionTask:
    name = args.task.replace("-", "_").lower()
    if name in {"exchangeable", "linear", "exchangeable_linear"}:
        return ExchangeableLinearRegression(tau=args.tau, sigma=args.sigma)
    if name in {"random_walk", "drift", "drifting"}:
        return RandomWalkLinearRegression(tau=args.tau, sigma=args.sigma, q=args.q)
    if name in {"changepoint", "change_point"}:
        return ChangepointRegression(tau=args.tau, sigma=args.sigma, min_segment=args.min_segment)
    if name in {"heteroscedastic", "temporal_noise"}:
        return HeteroscedasticTemporalRegression(
            tau=args.tau,
            sigma_start=args.sigma_start,
            sigma_end=args.sigma_end,
        )
    raise ValueError(f"unknown task: {args.task}")


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    if amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"unknown amp mode: {amp}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row[key] for row in rows if key in row) / max(1, sum(1 for row in rows if key in row)))
        for key in keys
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    task: RegressionTask,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    for _ in range(args.n_eval_batches):
        batch = task.sample(
            args.batch_size,
            args.context,
            args.x_dim,
            device=device,
            generator=generator,
        )
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
        )
        if oracle is not None:
            row.update(oracle_distance_metrics(pred, oracle.mean, oracle.variance))
        if args.n_permutations > 1:
            row.update(permutation_gap(model, batch, n_permutations=args.n_permutations))
        if args.task.replace("-", "_").lower() in {"exchangeable", "linear", "exchangeable_linear"}:
            row.update(martingale_prediction_stats(model, batch))
        rows.append(row)
    if was_training:
        model.train()
    return _average_metrics(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train synthetic Bayesian ICL regressors.")
    parser.add_argument("--task", default="exchangeable", choices=["exchangeable", "random_walk", "changepoint", "heteroscedastic"])
    parser.add_argument("--model", default="adaptive", choices=["regular", "ordered", "set", "set_llm", "adaptive"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", default="off", choices=["off", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", help="Wrap the model with torch.compile when available.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--x-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--sigma-start", type=float, default=0.3)
    parser.add_argument("--sigma-end", type=float, default=0.05)
    parser.add_argument("--min-segment", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--n-eval-batches", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/run"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = make_torch_generator(args.seed, device) if device.type in {"cpu", "cuda"} else None

    task = make_task_from_args(args)
    model = make_model(
        ModelConfig(
            model=args.model,
            x_dim=args.x_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_context=max(args.context, 1),
        )
    ).to(device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    metrics_path = args.out_dir / "metrics.jsonl"
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["out_dir"] = str(args.out_dir)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    model.train()
    for step in range(1, args.steps + 1):
        batch = task.sample(
            args.batch_size,
            args.context,
            args.x_dim,
            device=device,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            pred = model(batch.context_x, batch.context_y, batch.query_x)
            loss = gaussian_nll(pred["mean"], pred["variance"], batch.query_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "split": "train", "loss": float(loss.detach().cpu())}
            row.update(scalar_metrics(pred, batch.query_y))
            append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate(model, task, args, device, generator)
            row = {"step": step, "split": "eval", **eval_metrics}
            append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": config,
            "step": args.steps,
        },
        args.out_dir / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()

