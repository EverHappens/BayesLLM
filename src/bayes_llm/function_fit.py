from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch

from bayes_llm.plot_curves import load_model_from_run
from bayes_llm.train import resolve_device


_MISSING_MATPLOTLIB_REPORTED = False


def evaluate_named_function(name: str, x: torch.Tensor) -> torch.Tensor:
    normalized = name.replace("-", "_").lower()
    if normalized == "cosine":
        return torch.cos(x)
    if normalized == "sine":
        return torch.sin(x)
    if normalized == "linear":
        return x
    if normalized == "quadratic":
        return x.square()
    if normalized == "cubic":
        return x.pow(3)
    if normalized == "abs":
        return x.abs()
    if normalized == "step":
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    raise ValueError(f"unknown function: {name}")


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def make_feature_matrix(x: torch.Tensor, x_dim: int) -> torch.Tensor:
    features = torch.zeros(x.shape[0], x_dim, device=x.device, dtype=x.dtype)
    features[:, 0] = x
    return features


def make_context(
    function_name: str,
    demo_x: torch.Tensor,
    *,
    x_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return make_feature_matrix(demo_x, x_dim), evaluate_named_function(function_name, demo_x)


@torch.no_grad()
def predict_grid(
    model: torch.nn.Module,
    *,
    function_name: str,
    grid_x: torch.Tensor,
    demo_x: torch.Tensor,
    x_dim: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    context_x_single, context_y_single = make_context(function_name, demo_x, x_dim=x_dim)
    means = []
    variances = []
    for start in range(0, grid_x.shape[0], batch_size):
        chunk_x = grid_x[start : start + batch_size]
        query_x = make_feature_matrix(chunk_x, x_dim)
        context_x = context_x_single.unsqueeze(0).expand(chunk_x.shape[0], -1, -1)
        context_y = context_y_single.unsqueeze(0).expand(chunk_x.shape[0], -1)
        pred = model(context_x, context_y, query_x)
        means.append(pred["mean"].detach().cpu())
        variances.append(pred["variance"].detach().cpu())
    return {
        "mean": torch.cat(means, dim=0),
        "variance": torch.cat(variances, dim=0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_function_fit(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    demo_x: torch.Tensor,
    demo_y: torch.Tensor,
    function_name: str,
) -> None:
    global _MISSING_MATPLOTLIB_REPORTED
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if not _MISSING_MATPLOTLIB_REPORTED:
            print("matplotlib is not installed; wrote CSV and skipped PNG plot")
            _MISSING_MATPLOTLIB_REPORTED = True
        return

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    true_rows = sorted(by_model.pop("true", []), key=lambda row: float(row["x"]))
    if true_rows:
        ax.plot(
            [float(row["x"]) for row in true_rows],
            [float(row["y"]) for row in true_rows],
            color="black",
            linewidth=2.0,
            label=f"true {function_name}",
        )
    for model_name, model_rows in sorted(by_model.items()):
        ordered = sorted(model_rows, key=lambda row: float(row["x"]))
        xs = [float(row["x"]) for row in ordered]
        means = [float(row["mean"]) for row in ordered]
        ax.plot(xs, means, linewidth=1.6, label=model_name)
        if "std" in ordered[0]:
            std = torch.tensor([float(row["std"]) for row in ordered])
            mean = torch.tensor(means)
            x_tensor = torch.tensor(xs)
            ax.fill_between(
                x_tensor.numpy(),
                (mean - 2.0 * std).numpy(),
                (mean + 2.0 * std).numpy(),
                alpha=0.12,
            )
    ax.scatter(
        demo_x.detach().cpu().numpy(),
        demo_y.detach().cpu().numpy(),
        color="red",
        marker="o",
        zorder=5,
        label="demonstrations",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Function fit: {function_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize how checkpoints approximate a fixed 1D function.")
    parser.add_argument("--run-dir", action="append", default=[], help="Training run directory with config.json/checkpoint.pt.")
    parser.add_argument("--function", default="cosine", choices=["cosine", "sine", "linear", "quadratic", "cubic", "abs", "step"])
    parser.add_argument("--demo-x", default="-3.1416,-1.5708,0,1.5708,3.1416")
    parser.add_argument("--x-min", type=float, default=-math.pi)
    parser.add_argument("--x-max", type=float, default=math.pi)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--x-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/function_fit"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_x = torch.linspace(args.x_min, args.x_max, args.n_grid, device=device)
    demo_x = torch.tensor(parse_float_list(args.demo_x), device=device, dtype=torch.float32)
    demo_y = evaluate_named_function(args.function, demo_x)

    rows: list[dict[str, Any]] = []
    true_y = evaluate_named_function(args.function, grid_x)
    for x_value, y_value in zip(grid_x.detach().cpu().tolist(), true_y.detach().cpu().tolist(), strict=True):
        rows.append({"model": "true", "x": x_value, "y": y_value})

    for run_dir_text in args.run_dir:
        run_dir = Path(run_dir_text)
        x_dim = args.x_dim
        if x_dim is None:
            import json

            with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
                x_dim = int(json.load(handle).get("x_dim", 1))
        label, model = load_model_from_run(run_dir, device, max_context=demo_x.shape[0])
        pred = predict_grid(
            model,
            function_name=args.function,
            grid_x=grid_x,
            demo_x=demo_x,
            x_dim=x_dim,
            batch_size=args.batch_size,
        )
        mean = pred["mean"].detach().cpu()
        variance = pred["variance"].detach().cpu().clamp_min(0.0)
        true_cpu = true_y.detach().cpu()
        mse = float((mean - true_cpu).square().mean())
        print(f"{label}: grid_mse={mse:.6f}")
        for x_value, mean_value, var_value in zip(
            grid_x.detach().cpu().tolist(),
            mean.tolist(),
            variance.tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "model": label,
                    "x": x_value,
                    "mean": mean_value,
                    "variance": var_value,
                    "std": var_value**0.5,
                    "grid_mse": mse,
                }
            )

    csv_path = args.out_dir / "function_fit.csv"
    write_csv(csv_path, rows)
    plot_function_fit(
        output_path=args.out_dir / "function_fit.png",
        rows=rows,
        demo_x=demo_x,
        demo_y=demo_y,
        function_name=args.function,
    )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

