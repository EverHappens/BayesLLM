from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be at least 1")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth - 1):
        layers.extend([nn.Linear(current, hidden_dim), nn.GELU()])
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _make_transformer(
    hidden_dim: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=n_heads,
        dim_feedforward=4 * hidden_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class GaussianHead(nn.Module):
    def __init__(self, input_dim: int, min_variance: float = 1e-4) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 2)
        self.min_variance = min_variance

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.proj(hidden)
        mean = out[..., 0]
        variance = F.softplus(out[..., 1]) + self.min_variance
        return {
            "mean": mean,
            "variance": variance,
            "log_variance": torch.log(variance),
        }


class QueryEncoder(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(x_dim, hidden_dim, hidden_dim)

    def forward(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.net(query_x)


class SetLLMStyleContextEncoder(nn.Module):
    """Invariant set encoder with shared local positions and invariant pooling."""

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        local_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.x_token = nn.Linear(x_dim, hidden_dim)
        self.y_token = nn.Linear(1, hidden_dim)
        self.within_example_position = nn.Parameter(torch.zeros(2, hidden_dim))
        self.local_encoder = _make_transformer(hidden_dim, n_heads, local_layers, dropout)
        self.set_encoder = _make_transformer(hidden_dim, n_heads, n_layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        batch_size, n_context, _ = context_x.shape
        x_tok = self.x_token(context_x)
        y_tok = self.y_token(context_y.unsqueeze(-1))
        local_tokens = torch.stack([x_tok, y_tok], dim=2)
        local_tokens = local_tokens + self.within_example_position[None, None, :, :]
        local_tokens = local_tokens.reshape(batch_size * n_context, 2, -1)
        local_encoded = self.local_encoder(local_tokens).mean(dim=1)
        example_tokens = local_encoded.reshape(batch_size, n_context, -1)
        set_encoded = self.set_encoder(example_tokens)
        return self.norm(set_encoded.mean(dim=1))


class OrderedContextEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_context: int = 256,
    ) -> None:
        super().__init__()
        self.max_context = max_context
        self.example_encoder = _mlp(x_dim + 1, hidden_dim, hidden_dim)
        self.position = nn.Embedding(max_context, hidden_dim)
        self.transformer = _make_transformer(hidden_dim, n_heads, n_layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        batch_size, n_context, _ = context_x.shape
        if n_context > self.max_context:
            raise ValueError(f"n_context={n_context} exceeds max_context={self.max_context}")
        examples = torch.cat([context_x, context_y.unsqueeze(-1)], dim=-1)
        tokens = self.example_encoder(examples)
        positions = torch.arange(n_context, device=context_x.device)
        tokens = tokens + self.position(positions)[None, :, :]
        encoded = self.transformer(tokens)
        return self.norm(encoded[:, -1])


class PositionAwareTransformerRegressor(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_context: int = 256,
    ) -> None:
        super().__init__()
        self.context_encoder = OrderedContextEncoder(
            x_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_context=max_context,
        )
        self.query_encoder = QueryEncoder(x_dim, hidden_dim)
        self.head = GaussianHead(2 * hidden_dim)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context_encoder(context_x, context_y)
        query = self.query_encoder(query_x)
        return self.head(torch.cat([context, query], dim=-1))


class SetLLMStyleInvariantRegressor(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_encoder = SetLLMStyleContextEncoder(
            x_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.query_encoder = QueryEncoder(x_dim, hidden_dim)
        self.head = GaussianHead(2 * hidden_dim)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context_encoder(context_x, context_y)
        query = self.query_encoder(query_x)
        return self.head(torch.cat([context, query], dim=-1))


class AdaptiveTwoBranchRegressor(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_context: int = 256,
    ) -> None:
        super().__init__()
        self.set_encoder = SetLLMStyleContextEncoder(
            x_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.ordered_encoder = OrderedContextEncoder(
            x_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_context=max_context,
        )
        self.query_encoder = QueryEncoder(x_dim, hidden_dim)
        self.set_head = GaussianHead(2 * hidden_dim)
        self.ordered_head = GaussianHead(2 * hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> dict[str, torch.Tensor]:
        set_context = self.set_encoder(context_x, context_y)
        ordered_context = self.ordered_encoder(context_x, context_y)
        query = self.query_encoder(query_x)
        set_pred = self.set_head(torch.cat([set_context, query], dim=-1))
        ordered_pred = self.ordered_head(torch.cat([ordered_context, query], dim=-1))
        alpha = torch.sigmoid(self.gate(torch.cat([set_context, ordered_context, query], dim=-1))).squeeze(-1)
        mean = alpha * set_pred["mean"] + (1.0 - alpha) * ordered_pred["mean"]
        second_moment = alpha * (set_pred["variance"] + set_pred["mean"].square())
        second_moment = second_moment + (1.0 - alpha) * (ordered_pred["variance"] + ordered_pred["mean"].square())
        variance = (second_moment - mean.square()).clamp_min(1e-8)
        return {
            "mean": mean,
            "variance": variance,
            "log_variance": torch.log(variance),
            "set_mean": set_pred["mean"],
            "set_variance": set_pred["variance"],
            "ordered_mean": ordered_pred["mean"],
            "ordered_variance": ordered_pred["variance"],
            "gate_alpha": alpha,
        }


def _load_hf_backbone(model_id: str, trust_remote_code: bool) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face pretrained models require transformers. "
            "Install with `pip install -e '.[hf]'` or `pip install transformers`."
        ) from exc
    return AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def _hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("could not infer hidden size from Hugging Face backbone config")


class PretrainedCausalBackboneEncoder(nn.Module):
    """Numeric ICL encoder that reuses a pretrained decoder backbone."""

    def __init__(
        self,
        x_dim: int,
        model_id: str,
        *,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = _load_hf_backbone(model_id, trust_remote_code=trust_remote_code)
        self.hidden_dim = _hidden_size(self.backbone)
        self.example_adapter = nn.Linear(x_dim + 1, self.hidden_dim)
        self.query_adapter = nn.Linear(x_dim, self.hidden_dim)
        self.local_x_adapter = nn.Linear(x_dim, self.hidden_dim)
        self.local_y_adapter = nn.Linear(1, self.hidden_dim)
        self.ordered_example_marker = nn.Parameter(torch.zeros(self.hidden_dim))
        self.ordered_query_marker = nn.Parameter(torch.zeros(self.hidden_dim))
        self.local_position = nn.Parameter(torch.zeros(2, self.hidden_dim))
        self.set_norm = nn.LayerNorm(self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "PretrainedCausalBackboneEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _run_backbone(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        try:
            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Hugging Face backbone did not return hidden states")
            last_hidden = hidden_states[-1]
        return last_hidden

    def encode_ordered(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        examples = torch.cat([context_x, context_y.unsqueeze(-1)], dim=-1)
        example_tokens = self.example_adapter(examples) + self.ordered_example_marker[None, None, :]
        query_token = self.query_adapter(query_x).unsqueeze(1) + self.ordered_query_marker[None, None, :]
        hidden = self._run_backbone(torch.cat([example_tokens, query_token], dim=1))
        return hidden[:, -1]

    def encode_query(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.query_norm(self.query_adapter(query_x))

    def encode_set(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        batch_size, n_context, _ = context_x.shape
        x_token = self.local_x_adapter(context_x)
        y_token = self.local_y_adapter(context_y.unsqueeze(-1))
        local_tokens = torch.stack([x_token, y_token], dim=2)
        local_tokens = local_tokens + self.local_position[None, None, :, :]
        local_tokens = local_tokens.reshape(batch_size * n_context, 2, self.hidden_dim)
        local_hidden = self._run_backbone(local_tokens).mean(dim=1)
        example_hidden = local_hidden.reshape(batch_size, n_context, self.hidden_dim)
        return self.set_norm(example_hidden.mean(dim=1))


class PretrainedOrderedRegressor(nn.Module):
    """Ordered regressor using a pretrained Qwen/HF decoder as the sequence backbone."""

    def __init__(
        self,
        x_dim: int,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        *,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = PretrainedCausalBackboneEncoder(
            x_dim,
            model_id,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
        )
        self.head = GaussianHead(self.encoder.hidden_dim)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.encoder.encode_ordered(context_x, context_y, query_x))


class PretrainedSetRegressor(nn.Module):
    """Permutation-invariant regressor using a pretrained backbone per example."""

    def __init__(
        self,
        x_dim: int,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        *,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = PretrainedCausalBackboneEncoder(
            x_dim,
            model_id,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
        )
        self.head = GaussianHead(2 * self.encoder.hidden_dim)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encoder.encode_set(context_x, context_y)
        query = self.encoder.encode_query(query_x)
        return self.head(torch.cat([context, query], dim=-1))


class PretrainedAdaptiveRegressor(nn.Module):
    """Adaptive set/ordered regressor with a shared pretrained Qwen/HF backbone."""

    def __init__(
        self,
        x_dim: int,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        *,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = PretrainedCausalBackboneEncoder(
            x_dim,
            model_id,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
        )
        hidden_dim = self.encoder.hidden_dim
        self.set_head = GaussianHead(2 * hidden_dim)
        self.ordered_head = GaussianHead(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor) -> dict[str, torch.Tensor]:
        set_context = self.encoder.encode_set(context_x, context_y)
        ordered_context = self.encoder.encode_ordered(context_x, context_y, query_x)
        query = self.encoder.encode_query(query_x)
        set_pred = self.set_head(torch.cat([set_context, query], dim=-1))
        ordered_pred = self.ordered_head(ordered_context)
        alpha = torch.sigmoid(self.gate(torch.cat([set_context, ordered_context, query], dim=-1))).squeeze(-1)
        mean = alpha * set_pred["mean"] + (1.0 - alpha) * ordered_pred["mean"]
        second_moment = alpha * (set_pred["variance"] + set_pred["mean"].square())
        second_moment = second_moment + (1.0 - alpha) * (ordered_pred["variance"] + ordered_pred["mean"].square())
        variance = (second_moment - mean.square()).clamp_min(1e-8)
        return {
            "mean": mean,
            "variance": variance,
            "log_variance": torch.log(variance),
            "set_mean": set_pred["mean"],
            "set_variance": set_pred["variance"],
            "ordered_mean": ordered_pred["mean"],
            "ordered_variance": ordered_pred["variance"],
            "gate_alpha": alpha,
        }


@dataclass(frozen=True)
class ModelConfig:
    model: str
    x_dim: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    max_context: int = 256
    hf_model_id: str = "Qwen/Qwen2.5-0.5B"
    freeze_backbone: bool = False
    trust_remote_code: bool = False


def make_model(config: ModelConfig) -> nn.Module:
    name = config.model.replace("-", "_").lower()
    if name in {"qwen", "hf", "pretrained", "pretrained_ordered"}:
        return PretrainedOrderedRegressor(
            config.x_dim,
            model_id=config.hf_model_id,
            freeze_backbone=config.freeze_backbone,
            trust_remote_code=config.trust_remote_code,
        )
    if name in {"qwen_set", "hf_set", "pretrained_set"}:
        return PretrainedSetRegressor(
            config.x_dim,
            model_id=config.hf_model_id,
            freeze_backbone=config.freeze_backbone,
            trust_remote_code=config.trust_remote_code,
        )
    if name in {"qwen_adaptive", "hf_adaptive", "pretrained_adaptive"}:
        return PretrainedAdaptiveRegressor(
            config.x_dim,
            model_id=config.hf_model_id,
            freeze_backbone=config.freeze_backbone,
            trust_remote_code=config.trust_remote_code,
        )
    if name in {"regular", "ordered", "position_aware", "transformer"}:
        return PositionAwareTransformerRegressor(
            config.x_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            max_context=config.max_context,
        )
    if name in {"set", "set_only", "set_llm", "invariant"}:
        return SetLLMStyleInvariantRegressor(
            config.x_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
    if name in {"adaptive", "two_branch", "adaptive_two_branch"}:
        return AdaptiveTwoBranchRegressor(
            config.x_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            max_context=config.max_context,
        )
    raise ValueError(f"unknown model: {config.model}")
