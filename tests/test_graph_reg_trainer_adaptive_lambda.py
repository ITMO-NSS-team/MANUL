"""
Regression tests for GraphRegTrainer's Sobol-based adaptive lambda logic.

Covers the bug where get_adaptive_lambda_sobol() only ever looked at the
first 6 epochs of loss history regardless of the requested window, and the
lambda recalibration only ever happened once for the whole training run.

Two supported modes are checked:
- adaptive_lambda_recompute=False (default): lambdas are computed once, using
  the full window of loss history, and then held fixed for the rest of
  training.
- adaptive_lambda_recompute=True: lambdas are recomputed again at every
  window boundary, from the most recent window of loss history.
"""
import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch import nn

from regularizator.GraphRegTrainer import GraphRegTrainer

warnings.filterwarnings('ignore', category=FutureWarning,
                        message='unique with argument that is not not a Series')


def _train_and_get_graph_lambda(n_points, n_features, n_base, window, total_epochs,
                                recompute, cache_folder, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_points, n_features).astype(np.float64)
    y = rng.randn(n_points).astype(np.float64)
    base_indices = rng.choice(n_points, n_base, replace=False)
    weights_matrix = np.abs(rng.randn(n_base, n_base))
    weights_matrix = (weights_matrix + weights_matrix.T) / 2
    np.fill_diagonal(weights_matrix, 0)

    model = nn.Sequential(
        nn.Linear(n_features, 16, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(16, 1, dtype=torch.float64),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = GraphRegTrainer(
        train_features=X,
        train_target=y,
        weights_matrix=weights_matrix,
        base_indices=base_indices,
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        num_epochs=total_epochs,
        batch_size=32,
        device='cpu',
        cache_folder=cache_folder,
    )
    trainer.train(
        plot_convergence=False,
        adaptive_lambda='sobol',
        early_stopping_patience=total_epochs,  # disabled for this test
        adaptive_lambda_window=window,
        adaptive_lambda_recompute=recompute,
    )
    return trainer.convergence_history['graph_lambda']


class AdaptiveLambdaModeTests(unittest.TestCase):
    """Small ("synthetic-like") and larger ("mnist-like") feature scales,
    matching the two example pipelines this trainer is used by."""

    SCALES = {
        'synthetic_like': dict(n_points=120, n_features=3, n_base=24, window=10, total_epochs=35),
        'mnist_like': dict(n_points=400, n_features=784, n_base=50, window=10, total_epochs=35),
    }

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_folder = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_fixed_mode_holds_single_value_after_window(self):
        for name, params in self.SCALES.items():
            with self.subTest(scale=name):
                graph_lambda = _train_and_get_graph_lambda(
                    recompute=False, cache_folder=self.cache_folder, **params)
                after_window = np.round(graph_lambda[params['window']:], 8)
                self.assertEqual(
                    len(set(after_window)), 1,
                    f"[{name}] fixed mode should hold one constant lambda after the "
                    f"one-shot recompute, got {set(after_window)}")

    def test_recompute_mode_produces_multiple_values(self):
        for name, params in self.SCALES.items():
            with self.subTest(scale=name):
                graph_lambda = _train_and_get_graph_lambda(
                    recompute=True, cache_folder=self.cache_folder, **params)
                after_window = np.round(graph_lambda[params['window']:], 8)
                self.assertGreaterEqual(
                    len(set(after_window)), 2,
                    f"[{name}] recompute mode should keep changing lambda across "
                    f"window boundaries, got {set(after_window)}")


if __name__ == '__main__':
    unittest.main()
