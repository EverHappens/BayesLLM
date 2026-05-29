from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayes_llm.metrics import gaussian_nll
from bayes_llm.models import AdaptiveTwoBranchRegressor, SetLLMStyleInvariantRegressor
from bayes_llm.oracles import bayesian_linear_regression_predict, random_walk_kalman_predict
from bayes_llm.tasks import (
    ChangepointRegression,
    ExchangeableLinearRegression,
    RandomWalkLinearRegression,
    make_torch_generator,
)


class TaskGeneratorTests(unittest.TestCase):
    def test_generators_return_expected_shapes_and_finite_values(self) -> None:
        tasks = [
            ExchangeableLinearRegression(),
            RandomWalkLinearRegression(),
            ChangepointRegression(min_segment=2),
        ]
        for task in tasks:
            with self.subTest(task=task.__class__.__name__):
                batch = task.sample(4, 6, 3, device="cpu", generator=make_torch_generator(7))
                self.assertEqual(batch.context_x.shape, (4, 6, 3))
                self.assertEqual(batch.context_y.shape, (4, 6))
                self.assertEqual(batch.query_x.shape, (4, 3))
                self.assertEqual(batch.query_y.shape, (4,))
                self.assertTrue(torch.isfinite(batch.context_x).all())
                self.assertTrue(torch.isfinite(batch.context_y).all())
                self.assertEqual(batch.context_x.device.type, "cpu")

    def test_exchangeable_sampling_is_deterministic_on_cpu(self) -> None:
        task = ExchangeableLinearRegression()
        a = task.sample(2, 5, 3, device="cpu", generator=make_torch_generator(123))
        b = task.sample(2, 5, 3, device="cpu", generator=make_torch_generator(123))
        self.assertTrue(torch.allclose(a.context_x, b.context_x))
        self.assertTrue(torch.allclose(a.context_y, b.context_y))
        self.assertTrue(torch.allclose(a.query_y, b.query_y))


class OracleTests(unittest.TestCase):
    def test_exchangeable_oracle_is_permutation_invariant(self) -> None:
        task = ExchangeableLinearRegression(tau=1.0, sigma=0.2)
        batch = task.sample(8, 7, 4, device="cpu", generator=make_torch_generator(4))
        pred = bayesian_linear_regression_predict(batch.context_x, batch.context_y, batch.query_x, tau=1.0, sigma=0.2)
        perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
        pred_perm = bayesian_linear_regression_predict(
            batch.context_x[:, perm],
            batch.context_y[:, perm],
            batch.query_x,
            tau=1.0,
            sigma=0.2,
        )
        self.assertTrue(torch.allclose(pred.mean, pred_perm.mean, atol=1e-5))
        self.assertTrue(torch.allclose(pred.variance, pred_perm.variance, atol=1e-5))

    def test_random_walk_oracle_is_order_sensitive(self) -> None:
        context_x = torch.ones(1, 3, 1)
        context_y = torch.tensor([[0.0, 0.0, 5.0]])
        query_x = torch.ones(1, 1)
        pred = random_walk_kalman_predict(context_x, context_y, query_x, tau=1.0, sigma=0.1, q=1.0)
        pred_rev = random_walk_kalman_predict(
            context_x.flip(1),
            context_y.flip(1),
            query_x,
            tau=1.0,
            sigma=0.1,
            q=1.0,
        )
        self.assertGreater(torch.abs(pred.mean - pred_rev.mean).item(), 0.1)


class ModelTests(unittest.TestCase):
    def test_set_llm_style_regressor_is_permutation_invariant(self) -> None:
        torch.manual_seed(0)
        batch = ExchangeableLinearRegression().sample(3, 5, 4, device="cpu", generator=make_torch_generator(0))
        model = SetLLMStyleInvariantRegressor(x_dim=4, hidden_dim=16, n_heads=4, n_layers=1, dropout=0.0)
        model.eval()
        perm = torch.tensor([4, 2, 0, 3, 1])
        with torch.no_grad():
            pred = model(batch.context_x, batch.context_y, batch.query_x)
            pred_perm = model(batch.context_x[:, perm], batch.context_y[:, perm], batch.query_x)
        self.assertTrue(torch.allclose(pred["mean"], pred_perm["mean"], atol=1e-5))
        self.assertTrue(torch.allclose(pred["variance"], pred_perm["variance"], atol=1e-5))

    def test_adaptive_model_forward_backward_and_positive_variance(self) -> None:
        torch.manual_seed(1)
        batch = RandomWalkLinearRegression().sample(4, 6, 3, device="cpu", generator=make_torch_generator(1))
        model = AdaptiveTwoBranchRegressor(x_dim=3, hidden_dim=16, n_heads=4, n_layers=1, dropout=0.0, max_context=8)
        pred = model(batch.context_x, batch.context_y, batch.query_x)
        self.assertTrue(torch.isfinite(pred["mean"]).all())
        self.assertTrue(torch.isfinite(pred["variance"]).all())
        self.assertTrue((pred["variance"] > 0).all())
        loss = gaussian_nll(pred["mean"], pred["variance"], batch.query_y)
        loss.backward()
        grad_norm = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()

