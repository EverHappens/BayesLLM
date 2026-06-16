from __future__ import annotations

import argparse

import torch

from bayes_llm.tasks import RegressionBatch, make_task, make_torch_generator


def format_vector(vector: torch.Tensor, *, precision: int = 4) -> str:
    values = vector.detach().cpu().tolist()
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def format_regression_prompt(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    task_name: str | None = None,
    precision: int = 4,
    ordered: bool = True,
    style: str = "human",
) -> str:
    """Render one synthetic regression prompt in a human-readable ICL format."""

    if style == "compact":
        prefix = f"Task: {task_name} regression.\n" if task_name is not None else ""
        order_note = "Ordered examples:\n" if ordered else "Unordered examples:\n"
        lines = [prefix + "Infer the scalar function from examples.", order_note]
        for index in range(context_x.shape[0]):
            lines.append(
                f"x={format_vector(context_x[index], precision=precision)} -> "
                f"y={float(context_y[index].detach().cpu()):.{precision}f}"
            )
        lines.append(f"Query x={format_vector(query_x, precision=precision)} -> y=")
        return "\n".join(lines)
    if style != "human":
        raise ValueError(f"unknown prompt style: {style}")

    lines = []
    if task_name is not None:
        lines.append(f"Task: {task_name} regression.")
    if ordered:
        lines.append("The demonstrations are given in order. Use order when it is meaningful.")
    else:
        lines.append("The demonstrations form an unordered set. The order of examples is arbitrary.")
    lines.append("Given demonstration pairs (x, y), predict the scalar y for the query x.")
    lines.append("")
    for index in range(context_x.shape[0]):
        lines.append(
            f"Example {index + 1}: x = {format_vector(context_x[index], precision=precision)}, "
            f"y = {float(context_y[index].detach().cpu()):.{precision}f}"
        )
    lines.append("")
    lines.append(f"Query: x = {format_vector(query_x, precision=precision)}")
    lines.append("Answer with the predicted scalar y.")
    return "\n".join(lines)


def format_batch_prompt(
    batch: RegressionBatch,
    *,
    batch_index: int = 0,
    task_name: str | None = None,
    precision: int = 4,
    ordered: bool = True,
    style: str = "human",
) -> str:
    return format_regression_prompt(
        batch.context_x[batch_index],
        batch.context_y[batch_index],
        batch.query_x[batch_index],
        task_name=task_name,
        precision=precision,
        ordered=ordered,
        style=style,
    )


def numeric_prompt_explanation(n_context: int) -> str:
    return "\n".join(
        [
            "Current trained Qwen paths:",
            "- qwen_text tokenizes this prompt with AutoTokenizer and runs Qwen on input_ids.",
            "- qwen/qwen_set/qwen_adaptive map tensors through learned adapters and call the pretrained model with inputs_embeds.",
            f"- Ordered path shape before Qwen: [batch, {n_context + 1}, hidden], representing [example_1, ..., example_N, query].",
            f"- Set-LLM path shape before Qwen: [batch, {2 * n_context + 1}, hidden], representing [x1, y1, ..., xN, yN, query].",
            "- The text prompt above is also useful for inspection and future proprietary-model baselines.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a human-readable synthetic ICL prompt.")
    parser.add_argument("--task", default="exchangeable")
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--x-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--ordered", action="store_true", help="Describe demonstrations as ordered.")
    parser.add_argument("--style", default="human", choices=["human", "compact"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    task = make_task(args.task)
    generator = make_torch_generator(args.seed, "cpu")
    batch = task.sample(
        max(1, args.batch_index + 1),
        args.context,
        args.x_dim,
        device="cpu",
        generator=generator,
    )
    prompt = format_batch_prompt(
        batch,
        batch_index=args.batch_index,
        task_name=args.task,
        precision=args.precision,
        ordered=args.ordered,
        style=args.style,
    )
    print(prompt)
    print("")
    print(f"True sampled query_y: {float(batch.query_y[args.batch_index]):.{args.precision}f}")
    print("")
    print(numeric_prompt_explanation(args.context))


if __name__ == "__main__":
    main()
