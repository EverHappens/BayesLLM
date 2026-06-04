from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


TensorDict = dict[str, Any]


@dataclass(frozen=True)
class RegressionBatch:
    """A batch of in-context regression prompts."""

    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    metadata: TensorDict

    def to(self, device: torch.device | str) -> "RegressionBatch":
        target = torch.device(device)
        metadata: TensorDict = {}
        for key, value in self.metadata.items():
            metadata[key] = value.to(target) if torch.is_tensor(value) else value
        return RegressionBatch(
            context_x=self.context_x.to(target),
            context_y=self.context_y.to(target),
            query_x=self.query_x.to(target),
            query_y=self.query_y.to(target),
            metadata=metadata,
        )


class RegressionTask(Protocol):
    def sample(
        self,
        batch_size: int,
        n_context: int,
        x_dim: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> RegressionBatch:
        ...


def make_torch_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a generator on the target device when PyTorch supports it."""

    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.Generator(device=dev).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _randn(
    *shape: int,
    device: torch.device,
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, device=device, generator=_compatible_generator(generator, device), dtype=dtype)


def _compatible_generator(
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Generator | None:
    if generator is None:
        return None
    try:
        generator_device = torch.device(generator.device)
    except AttributeError:
        generator_device = torch.device("cpu")
    if device.type == "cpu" and generator_device.type == "cpu":
        return generator
    if device.type == "cuda" and generator_device == device:
        return generator
    return None


@dataclass(frozen=True)
class ExchangeableLinearRegression:
    tau: float = 1.0
    sigma: float = 0.1
    dtype: torch.dtype = torch.float32

    def sample(
        self,
        batch_size: int,
        n_context: int,
        x_dim: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> RegressionBatch:
        dev = torch.device(device)
        w = self.tau * _randn(batch_size, x_dim, device=dev, generator=generator, dtype=self.dtype)
        x = _randn(batch_size, n_context + 1, x_dim, device=dev, generator=generator, dtype=self.dtype)
        noise = self.sigma * _randn(batch_size, n_context + 1, device=dev, generator=generator, dtype=self.dtype)
        y = torch.einsum("btd,bd->bt", x, w) + noise
        return RegressionBatch(
            context_x=x[:, :n_context],
            context_y=y[:, :n_context],
            query_x=x[:, n_context],
            query_y=y[:, n_context],
            metadata={
                "task": "exchangeable",
                "weights": w,
                "tau": torch.tensor(self.tau, device=dev, dtype=self.dtype),
                "sigma": torch.tensor(self.sigma, device=dev, dtype=self.dtype),
            },
        )


@dataclass(frozen=True)
class RandomWalkLinearRegression:
    tau: float = 1.0
    sigma: float = 0.1
    q: float = 0.05
    dtype: torch.dtype = torch.float32

    def sample(
        self,
        batch_size: int,
        n_context: int,
        x_dim: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> RegressionBatch:
        dev = torch.device(device)
        total_steps = n_context + 1
        initial = self.tau * _randn(batch_size, x_dim, device=dev, generator=generator, dtype=self.dtype)
        increments = (self.q**0.5) * _randn(
            batch_size,
            total_steps - 1,
            x_dim,
            device=dev,
            generator=generator,
            dtype=self.dtype,
        )
        weights = torch.empty(batch_size, total_steps, x_dim, device=dev, dtype=self.dtype)
        weights[:, 0] = initial
        if total_steps > 1:
            weights[:, 1:] = initial[:, None, :] + torch.cumsum(increments, dim=1)
        x = _randn(batch_size, total_steps, x_dim, device=dev, generator=generator, dtype=self.dtype)
        noise = self.sigma * _randn(batch_size, total_steps, device=dev, generator=generator, dtype=self.dtype)
        y = torch.einsum("btd,btd->bt", x, weights) + noise
        return RegressionBatch(
            context_x=x[:, :n_context],
            context_y=y[:, :n_context],
            query_x=x[:, n_context],
            query_y=y[:, n_context],
            metadata={
                "task": "random_walk",
                "weights": weights,
                "tau": torch.tensor(self.tau, device=dev, dtype=self.dtype),
                "sigma": torch.tensor(self.sigma, device=dev, dtype=self.dtype),
                "q": torch.tensor(self.q, device=dev, dtype=self.dtype),
            },
        )


@dataclass(frozen=True)
class ChangepointRegression:
    tau: float = 1.0
    sigma: float = 0.1
    min_segment: int = 2
    dtype: torch.dtype = torch.float32

    def sample(
        self,
        batch_size: int,
        n_context: int,
        x_dim: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> RegressionBatch:
        if n_context < 2 * self.min_segment:
            raise ValueError("n_context must be at least 2 * min_segment for changepoint tasks")
        dev = torch.device(device)
        total_steps = n_context + 1
        low = self.min_segment
        high = n_context - self.min_segment + 1
        changepoint = torch.randint(
            low,
            high,
            (batch_size,),
            device=dev,
            generator=_compatible_generator(generator, dev),
        )
        w_old = self.tau * _randn(batch_size, x_dim, device=dev, generator=generator, dtype=self.dtype)
        w_new = self.tau * _randn(batch_size, x_dim, device=dev, generator=generator, dtype=self.dtype)
        x = _randn(batch_size, total_steps, x_dim, device=dev, generator=generator, dtype=self.dtype)
        t = torch.arange(total_steps, device=dev)[None, :]
        use_new = t >= changepoint[:, None]
        weights = torch.where(use_new[..., None], w_new[:, None, :], w_old[:, None, :])
        noise = self.sigma * _randn(batch_size, total_steps, device=dev, generator=generator, dtype=self.dtype)
        y = torch.einsum("btd,btd->bt", x, weights) + noise
        return RegressionBatch(
            context_x=x[:, :n_context],
            context_y=y[:, :n_context],
            query_x=x[:, n_context],
            query_y=y[:, n_context],
            metadata={
                "task": "changepoint",
                "changepoint": changepoint,
                "w_old": w_old,
                "w_new": w_new,
                "tau": torch.tensor(self.tau, device=dev, dtype=self.dtype),
                "sigma": torch.tensor(self.sigma, device=dev, dtype=self.dtype),
            },
        )


@dataclass(frozen=True)
class HeteroscedasticTemporalRegression:
    tau: float = 1.0
    sigma_start: float = 0.3
    sigma_end: float = 0.05
    dtype: torch.dtype = torch.float32

    def sample(
        self,
        batch_size: int,
        n_context: int,
        x_dim: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> RegressionBatch:
        dev = torch.device(device)
        total_steps = n_context + 1
        w = self.tau * _randn(batch_size, x_dim, device=dev, generator=generator, dtype=self.dtype)
        x = _randn(batch_size, total_steps, x_dim, device=dev, generator=generator, dtype=self.dtype)
        sigma_t = torch.linspace(
            self.sigma_start,
            self.sigma_end,
            total_steps,
            device=dev,
            dtype=self.dtype,
        )
        noise = sigma_t[None, :] * _randn(batch_size, total_steps, device=dev, generator=generator, dtype=self.dtype)
        y = torch.einsum("btd,bd->bt", x, w) + noise
        return RegressionBatch(
            context_x=x[:, :n_context],
            context_y=y[:, :n_context],
            query_x=x[:, n_context],
            query_y=y[:, n_context],
            metadata={
                "task": "heteroscedastic",
                "weights": w,
                "sigma_t": sigma_t,
                "tau": torch.tensor(self.tau, device=dev, dtype=self.dtype),
            },
        )


def _pairwise_squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    diff = left[:, :, None, :] - right[:, None, :, :]
    return diff.square().sum(dim=-1)


def gp_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    kernel: str,
    amplitude: float,
    lengthscale: float,
) -> torch.Tensor:
    dist2 = _pairwise_squared_distance(left, right).clamp_min(0.0)
    if kernel == "rbf":
        return amplitude**2 * torch.exp(-0.5 * dist2 / (lengthscale**2))
    dist = torch.sqrt(dist2 + 1e-12)
    if kernel == "matern12":
        return amplitude**2 * torch.exp(-dist / lengthscale)
    if kernel == "matern32":
        scale = 3.0**0.5 * dist / lengthscale
        return amplitude**2 * (1.0 + scale) * torch.exp(-scale)
    raise ValueError(f"unknown GP kernel: {kernel}")


@dataclass(frozen=True)
class GaussianProcessRegression:
    kernel: str = "rbf"
    amplitude: float = 1.0
    lengthscale: float = 1.0
    sigma: float = 0.05
    dtype: torch.dtype = torch.float32

    def sample(
        self,
        batch_size: int,
        n_context: int,
        x_dim: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> RegressionBatch:
        dev = torch.device(device)
        total_steps = n_context + 1
        x = _randn(batch_size, total_steps, x_dim, device=dev, generator=generator, dtype=self.dtype)
        kernel = gp_kernel(
            x,
            x,
            kernel=self.kernel,
            amplitude=self.amplitude,
            lengthscale=self.lengthscale,
        )
        eye = torch.eye(total_steps, device=dev, dtype=self.dtype).expand(batch_size, total_steps, total_steps)
        cov = kernel + (self.sigma**2 + 1e-6) * eye
        chol = torch.linalg.cholesky(cov)
        eps = _randn(batch_size, total_steps, device=dev, generator=generator, dtype=self.dtype)
        y = torch.bmm(chol, eps.unsqueeze(-1)).squeeze(-1)
        return RegressionBatch(
            context_x=x[:, :n_context],
            context_y=y[:, :n_context],
            query_x=x[:, n_context],
            query_y=y[:, n_context],
            metadata={
                "task": f"gp_{self.kernel}",
                "kernel": self.kernel,
                "amplitude": torch.tensor(self.amplitude, device=dev, dtype=self.dtype),
                "lengthscale": torch.tensor(self.lengthscale, device=dev, dtype=self.dtype),
                "sigma": torch.tensor(self.sigma, device=dev, dtype=self.dtype),
            },
        )


def make_task(name: str) -> RegressionTask:
    normalized = name.replace("-", "_").lower()
    if normalized in {"exchangeable", "linear", "exchangeable_linear"}:
        return ExchangeableLinearRegression()
    if normalized in {"random_walk", "drift", "drifting"}:
        return RandomWalkLinearRegression()
    if normalized in {"changepoint", "change_point"}:
        return ChangepointRegression()
    if normalized in {"heteroscedastic", "temporal_noise"}:
        return HeteroscedasticTemporalRegression()
    if normalized in {"gp", "gp_rbf"}:
        return GaussianProcessRegression(kernel="rbf")
    if normalized in {"gp_matern12", "gp_exponential"}:
        return GaussianProcessRegression(kernel="matern12")
    if normalized in {"gp_matern32"}:
        return GaussianProcessRegression(kernel="matern32")
    raise ValueError(f"unknown task: {name}")
