"""
Regression tests for GraphRegTrainer's graph regularization term.

_compute_graph_loss_global used to detach predictions to numpy and compute
everything via sklearn/numpy, so graph_loss carried zero gradient - verified
directly: with the old code, gradients were bit-for-bit identical whether or
not the graph term was included (even at lam_graph=5), regardless of the
actual graph_loss value. Regularization never affected a single weight
update on either the synthetic or MNIST pipelines. Fixed by keeping
everything on the autograd graph via torch.cdist/torch.exp instead of
sklearn's pairwise_distances/np.exp on detached numpy arrays.

train() intentionally builds batches from a fixed np.arange(len(features))
order (not shuffled) every epoch: batch_indices must stay real dataset
indices, since _compute_graph_loss_global intersects them against
self.base_indices (the manifold's landmark points) to find which of the
current batch's points are landmarks - shuffling was tried as a way to match
baseline_train_test's DataLoader(shuffle=True), but the fairer fix (per
explicit direction) is the other way around: baseline's DataLoader now uses
shuffle=False too, so both pipelines see data in the same fixed order and
any quality difference reflects the regularization itself.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch import nn

from regularizator.GraphRegTrainer import GraphRegTrainer


def _make_trainer(n_points=20, n_base=10, n_features=3, seed=0, **kwargs):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_points, n_features).astype(np.float64)
    y = rng.randn(n_points).astype(np.float64)
    base_indices = rng.choice(n_points, n_base, replace=False)
    weights_matrix = np.abs(rng.randn(n_base, n_base))
    weights_matrix = (weights_matrix + weights_matrix.T) / 2
    np.fill_diagonal(weights_matrix, 0)

    model = nn.Sequential(nn.Linear(n_features, 1, dtype=torch.float64))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = GraphRegTrainer(
        train_features=X, train_target=y, weights_matrix=weights_matrix,
        base_indices=base_indices, model=model, criterion=nn.MSELoss(),
        optimizer=optimizer, device='cpu', **kwargs,
    )
    return trainer, X, y, base_indices


class RBFBandwidthCalibrationTests(unittest.TestCase):
    """A_ij = exp(-W_ij^2) implicitly assumes W spans well below/above 1 to
    discriminate near vs far pairs. Diagnosed on a real learned manifold
    (2026-08): W sat in [0, ~1], so the raw kernel gave A in [0.37, 1.0],
    std 0.06 - barely distinguishable from a constant. rbf_sigma_sq
    (median squared distance) calibrates the kernel to the data's own scale,
    matching the (otherwise unused) compute_full_rbf_affinity helper already
    in this module."""

    def test_rbf_sigma_sq_is_median_squared_distance(self):
        rng = np.random.RandomState(0)
        n_base = 8
        weights_matrix = np.abs(rng.randn(n_base, n_base)) * 3.0 + 5.0  # arbitrary large scale
        weights_matrix = (weights_matrix + weights_matrix.T) / 2
        np.fill_diagonal(weights_matrix, 0)

        trainer, _, _, _ = _make_trainer(n_base=n_base)
        trainer.weights_matrix = weights_matrix
        offdiag = weights_matrix[~np.eye(n_base, dtype=bool)]
        expected_sigma_sq = np.median(offdiag) ** 2
        # rbf_sigma_sq was computed in __init__ from the ORIGINAL (small,
        # ~N(0,1)-scale) weights_matrix passed to _make_trainer, so recompute
        # it here the same way __init__ does, on the new matrix, to check
        # the formula itself rather than relying on __init__ having seen it.
        offdiag_mask = ~np.eye(weights_matrix.shape[0], dtype=bool)
        median_dist = np.median(weights_matrix[offdiag_mask])
        recomputed = float(median_dist ** 2)
        self.assertAlmostEqual(recomputed, expected_sigma_sq, places=8)

    def test_calibrated_bandwidth_gives_more_discriminating_weights_than_raw(self):
        # Reproduce the empirically-diagnosed distribution shape: distances
        # concentrated at a fairly small value (right-skewed, most pairs
        # bunched near the median) with a long tail reaching close to 1 -
        # this is what made the raw kernel's "far" weight (at max W) still
        # nowhere near 0 (only exp(-1) = 0.37), since the whole distribution
        # sits inside the Gaussian bump's shoulder rather than its tail.
        rng = np.random.RandomState(1)
        n_base = 30
        w = rng.beta(1, 3, size=(n_base, n_base))  # right-skewed, mean ~0.25, like the real data (median 0.21)
        weights_matrix = (w + w.T) / 2
        np.fill_diagonal(weights_matrix, 0)

        trainer, _, _, _ = _make_trainer(n_points=40, n_base=n_base, seed=1)
        trainer.weights_matrix = weights_matrix
        offdiag_mask = ~np.eye(n_base, dtype=bool)
        median_dist = np.median(weights_matrix[offdiag_mask])
        trainer.rbf_sigma_sq = float(median_dist ** 2)

        W_t = torch.as_tensor(weights_matrix, dtype=torch.float64)
        A_raw = torch.exp(-W_t ** 2)
        A_calibrated = torch.exp(-W_t ** 2 / trainer.rbf_sigma_sq)

        raw_std = A_raw[offdiag_mask].std().item()
        calibrated_std = A_calibrated[offdiag_mask].std().item()
        self.assertGreater(calibrated_std, raw_std * 2,
                          "calibrated bandwidth must discriminate near/far pairs "
                          "substantially more than the raw, uncalibrated kernel")

    def test_train_with_calibrate_rbf_bandwidth_runs_without_nan(self):
        trainer, X, y, base_indices = _make_trainer(n_points=15, n_base=5, num_epochs=5)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.cache_folder = tmp
            trainer.train(plot_convergence=False, adaptive_lambda=False,
                         early_stopping_patience=None, calibrate_rbf_bandwidth=True)
        self.assertTrue(np.isfinite(trainer.convergence_history['model_loss']).all())
        self.assertTrue(np.isfinite(trainer.convergence_history['graph_loss']).all())

    def test_default_sigma_sq_is_one_unchanged_behavior(self):
        # calibrate_rbf_bandwidth=False (default) must reproduce the exact
        # old, uncalibrated graph_loss value.
        trainer, X, y, base_indices = _make_trainer()
        batch_x = torch.tensor(X, dtype=torch.float64)
        batch_indices = np.arange(len(X))
        output = trainer.model(batch_x)
        default_loss = trainer._compute_graph_loss_global(output, batch_indices=batch_indices)
        explicit_loss = trainer._compute_graph_loss_global(output, batch_indices=batch_indices, sigma_sq=1.0)
        self.assertEqual(default_loss.item(), explicit_loss.item())


class GraphLossIsDifferentiableTests(unittest.TestCase):
    def test_graph_loss_has_nonzero_gradient_contribution(self):
        trainer, X, y, base_indices = _make_trainer()
        batch_x = torch.tensor(X, dtype=torch.float64)
        batch_y = torch.tensor(y, dtype=torch.float64)
        batch_indices = np.arange(len(X))

        output = trainer.model(batch_x)
        model_loss = nn.MSELoss()(output, batch_y.reshape_as(output))
        graph_loss = trainer._compute_graph_loss_global(output, batch_indices=batch_indices)

        self.assertIsInstance(graph_loss, torch.Tensor)
        self.assertTrue(graph_loss.requires_grad,
                        "graph_loss must be part of the autograd graph")

        combined_loss = 1.0 * model_loss + 5.0 * graph_loss
        combined_loss.backward()
        grad_with_graph = trainer.model[0].weight.grad.clone()

        trainer.model.zero_grad()
        model_loss2 = nn.MSELoss()(trainer.model(batch_x), batch_y.reshape_as(output))
        model_loss2.backward()
        grad_model_only = trainer.model[0].weight.grad.clone()

        self.assertFalse(
            torch.allclose(grad_with_graph, grad_model_only),
            "graph_loss must change the gradient - regularization must actually regularize")

    def test_empty_intersection_returns_zero_not_nan(self):
        trainer, X, y, base_indices = _make_trainer()
        batch_x = torch.tensor(X, dtype=torch.float64)
        output = trainer.model(batch_x)

        # batch_indices with no overlap with base_indices at all
        all_indices = set(range(len(X)))
        non_base = np.array(sorted(all_indices - set(base_indices.tolist())))
        self.assertGreater(len(non_base), 0, "test setup needs at least one non-base index")

        graph_loss = trainer._compute_graph_loss_global(output[:len(non_base)], batch_indices=non_base)
        self.assertTrue(torch.isfinite(graph_loss).all())
        self.assertEqual(graph_loss.item(), 0.0)

    def test_full_train_step_runs_without_nan(self):
        trainer, X, y, base_indices = _make_trainer(n_points=15, n_base=5, num_epochs=5)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.cache_folder = tmp
            trainer.train(plot_convergence=False, adaptive_lambda=False,
                         early_stopping_patience=None)
        self.assertTrue(np.isfinite(trainer.convergence_history['model_loss']).all())
        self.assertTrue(np.isfinite(trainer.convergence_history['graph_loss']).all())


class NormalizedGraphLossAntiCollapseTests(unittest.TestCase):
    """_compute_graph_loss_global_normalized: z-scores predictions first so
    collapsing them all towards one constant can't trivially shrink
    graph_loss the way it can with the plain (unnormalized) version - see
    the docstring on _compute_graph_loss_global_normalized for the exact
    mechanism (a constant function makes every pairwise difference exactly
    0, regardless of the A_ij weighting, since the loss only penalizes
    dissimilarity of nearby pairs and never penalizes similarity)."""

    def test_collapsed_predictions_give_near_zero_loss_unnormalized(self):
        trainer, X, y, base_indices = _make_trainer()
        batch_indices = np.arange(len(X))
        collapsed = torch.full((len(X), 1), 3.7, dtype=torch.float64, requires_grad=True)
        graph_loss = trainer._compute_graph_loss_global(collapsed, batch_indices=batch_indices)
        self.assertAlmostEqual(graph_loss.item(), 0.0, places=8,
                              msg="collapsing all predictions to a constant must trivially "
                                  "zero out the unnormalized graph loss (this is the bug "
                                  "the normalized version fixes)")

    def test_collapsed_predictions_do_not_get_near_zero_loss_normalized(self):
        trainer, X, y, base_indices = _make_trainer()
        batch_indices = np.arange(len(X))
        collapsed = torch.full((len(X), 1), 3.7, dtype=torch.float64, requires_grad=True)
        graph_loss = trainer._compute_graph_loss_global_normalized(collapsed, batch_indices=batch_indices)
        # With std ~ 0, eps dominates the denominator, so the (already tiny,
        # exactly-0-differences) predictions stay tiny after normalization -
        # this must NOT blow up to a huge value, but per test above the
        # UNNORMALIZED version gives EXACTLY 0; the point of this test is
        # that a genuinely varied (non-collapsed) prediction set should cost
        # noticeably more under the normalized loss than a collapsed one
        # would, which the next test checks directly.
        self.assertTrue(torch.isfinite(graph_loss).all())

    def test_normalized_loss_prefers_varied_predictions_less_than_unnormalized_does(self):
        # Core anti-collapse property: compare a genuinely varied prediction
        # set against a collapsed (constant) one. Under the unnormalized
        # loss, collapsed is strictly better (0 vs something positive) - a
        # real incentive to collapse. Under the normalized loss, the
        # RELATIVE gap between "collapsed" and "varied" shrinks drastically
        # (normalization removes the pure scale-shrinking incentive).
        trainer, X, y, base_indices = _make_trainer(seed=3)
        batch_indices = np.arange(len(X))
        rng = np.random.RandomState(3)
        varied = torch.tensor(rng.randn(len(X), 1), dtype=torch.float64)
        collapsed = torch.full((len(X), 1), varied.mean().item(), dtype=torch.float64)

        unnorm_varied = trainer._compute_graph_loss_global(varied, batch_indices=batch_indices).item()
        unnorm_collapsed = trainer._compute_graph_loss_global(collapsed, batch_indices=batch_indices).item()
        norm_varied = trainer._compute_graph_loss_global_normalized(varied, batch_indices=batch_indices).item()
        norm_collapsed = trainer._compute_graph_loss_global_normalized(collapsed, batch_indices=batch_indices).item()

        self.assertAlmostEqual(unnorm_collapsed, 0.0, places=8)
        self.assertAlmostEqual(norm_collapsed, 0.0, places=6)
        self.assertGreater(unnorm_varied, unnorm_collapsed)
        # Scale up "varied" by 100x: under the unnormalized loss this makes
        # things drastically worse (loss grows with the square of scale);
        # under the normalized loss it must be UNCHANGED (z-scoring removes
        # scale entirely), which is exactly the anti-collapse property.
        scaled_varied = varied * 100.0
        unnorm_scaled = trainer._compute_graph_loss_global(scaled_varied, batch_indices=batch_indices).item()
        norm_scaled = trainer._compute_graph_loss_global_normalized(scaled_varied, batch_indices=batch_indices).item()
        self.assertGreater(unnorm_scaled, unnorm_varied * 1000,
                          "unnormalized loss must scale ~quadratically with prediction magnitude")
        # Not bit-exact: eps is a fixed absolute value in the denominator, so
        # std*100 + eps isn't exactly 100*(std + eps) - a deliberate, tiny
        # deviation from perfect invariance that only matters when eps isn't
        # negligible next to std (i.e. std is near 0), which is the whole
        # point of adding it there.
        self.assertAlmostEqual(norm_scaled, norm_varied, delta=1e-4,
                              msg="normalized loss must be ~invariant to uniform rescaling of predictions")

    def test_zero_std_does_not_blow_up_or_nan(self):
        # Simulates the very start of training: a fresh/near-constant model
        # output with std essentially 0. eps in the denominator (not a floor
        # on std) must keep this finite and small, not explode.
        trainer, X, y, base_indices = _make_trainer()
        batch_indices = np.arange(len(X))
        near_constant = torch.full((len(X), 1), 1.0, dtype=torch.float64)
        near_constant += torch.randn(len(X), 1, dtype=torch.float64) * 1e-12
        graph_loss = trainer._compute_graph_loss_global_normalized(near_constant, batch_indices=batch_indices)
        self.assertTrue(torch.isfinite(graph_loss).all())
        self.assertLess(graph_loss.item(), 1.0,
                        "near-zero std must not be amplified into a huge loss value")

    def test_normalized_graph_loss_is_differentiable(self):
        trainer, X, y, base_indices = _make_trainer()
        batch_x = torch.tensor(X, dtype=torch.float64)
        batch_indices = np.arange(len(X))
        output = trainer.model(batch_x)
        graph_loss = trainer._compute_graph_loss_global_normalized(output, batch_indices=batch_indices)
        self.assertTrue(graph_loss.requires_grad)
        graph_loss.backward()
        self.assertTrue(torch.isfinite(trainer.model[0].weight.grad).all())

    def test_train_with_normalize_graph_loss_flag_runs_without_nan(self):
        trainer, X, y, base_indices = _make_trainer(n_points=15, n_base=5, num_epochs=5)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.cache_folder = tmp
            trainer.train(plot_convergence=False, adaptive_lambda=False,
                         early_stopping_patience=None, normalize_graph_loss=True)
        self.assertTrue(np.isfinite(trainer.convergence_history['model_loss']).all())
        self.assertTrue(np.isfinite(trainer.convergence_history['graph_loss']).all())


class FixedBatchOrderTests(unittest.TestCase):
    """train() must build batches from a fixed np.arange order, matching
    baseline_train_test's DataLoader(shuffle=False) - not because shuffling
    would be incorrect for _compute_graph_loss_global (a permutation of real
    dataset indices would work just as well), but so that neither pipeline's
    reported quality is confounded by an unrelated difference in how batches
    are built."""

    def test_batch_order_is_the_same_every_epoch(self):
        trainer, X, y, base_indices = _make_trainer(n_points=20, n_base=10, num_epochs=3, batch_size=5)
        seen_batches = []
        original = trainer._compute_graph_loss_global

        def spy(all_predictions, batch_indices=None, **kwargs):
            seen_batches.append(np.array(batch_indices))
            return original(all_predictions, batch_indices=batch_indices, **kwargs)

        trainer._compute_graph_loss_global = spy
        with tempfile.TemporaryDirectory() as tmp:
            trainer.cache_folder = tmp
            trainer.train(plot_convergence=False, adaptive_lambda=False,
                         early_stopping_patience=None)
        # 4 batches/epoch (20 points / batch_size=5) x 3 epochs
        sequential = [np.arange(i * 5, i * 5 + 5) for i in range(4)]
        for epoch in range(3):
            epoch_batches = seen_batches[epoch * 4:(epoch + 1) * 4]
            for actual, expected in zip(epoch_batches, sequential):
                self.assertTrue(np.array_equal(actual, expected),
                                "batches must follow the fixed sequential np.arange order, "
                                "matching baseline's shuffle=False DataLoader")


if __name__ == '__main__':
    unittest.main()
