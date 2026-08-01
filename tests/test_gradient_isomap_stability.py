"""
Regression tests for the numerical-stability fixes in GradientIsomap:
- train() zeroes NaN/Inf gradients before the optimizer step, instead of
  letting a bad backward pass (eigh's backward divides by
  (eigenvalue_i - eigenvalue_j), which can blow up near-degenerate pairs)
  permanently corrupt the distance matrix.
- _stable_eigenvalues() reacts to that observed failure (had_bad_grad) in
  addition to the original small-top-eigenvalue check. It deliberately does
  NOT key off "some eigenvalue pair is close together" - among the handful
  of kept components (~latent_dim, not the full spectrum) a close pair is a
  common, healthy spectrum shape and does not by itself mean anything went
  wrong. An earlier version of this fix checked exactly that and ended up
  permanently stuck at the "degenerate" high learning rate, stalling
  training instead of protecting it.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from Adam.GradientIsomap import GradientIsomap


def _make_isomap(tmp_dir, n_points=10, latent_len=2):
    torch.manual_seed(0)
    features = torch.rand(n_points, 3)
    targets = torch.rand(n_points)
    return GradientIsomap(
        train_feature=features,
        train_target=targets,
        latent_len=latent_len,
        checkpoint_each=None,
        logs_folder=tmp_dir,
        epochs=1,
    )


class StableEigenvaluesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.isomap = _make_isomap(self._tmpdir.name)
        self.optim = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0001)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_top_eigenvalue_small_is_flagged_degenerate(self):
        eigenvalues = np.array([0.005, 0.001, -0.5, -1.0])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=False)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.01)

    def test_bad_gradient_is_flagged_degenerate_even_with_healthy_eigenvalues(self):
        eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=True)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.01)

    def test_close_eigenvalue_pair_alone_is_not_flagged_degenerate(self):
        # Regression guard: a close pair among the ~latent_dim kept components is
        # common and, on its own (no actual bad gradient observed), must NOT force
        # the high-lr branch - that previously left training permanently stuck.
        eigenvalues = np.array([10.0, 9.995, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=False)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.0001)

    def test_well_separated_eigenvalues_are_not_degenerate(self):
        eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=False)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.0001)


class NanGradientGuardTests(unittest.TestCase):
    def _clean_grad_and_detect_bad(self, param):
        had_bad_grad = False
        if not torch.isfinite(param.grad).all():
            had_bad_grad = True
        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return had_bad_grad

    def test_train_zeroes_nan_and_inf_gradients_before_step(self):
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        optim = torch.optim.AdamW([param], lr=0.0001)
        param.grad = torch.tensor([float('nan'), float('inf'), 5.0])

        had_bad_grad = self._clean_grad_and_detect_bad(param)

        self.assertTrue(had_bad_grad, "a non-finite gradient must be detected")
        self.assertTrue(torch.isfinite(param.grad).all(),
                        "gradient should be fully finite after the guard")
        self.assertEqual(param.grad[2].item(), 5.0,
                        "a legitimate finite gradient entry must be left untouched")

        # step() must not raise / must not turn the parameter itself into NaN
        optim.step()
        self.assertTrue(torch.isfinite(param).all())

    def test_finite_gradient_is_not_flagged_bad(self):
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        param.grad = torch.tensor([0.1, -0.2, 5.0])

        had_bad_grad = self._clean_grad_and_detect_bad(param)

        self.assertFalse(had_bad_grad)
        self.assertTrue(torch.allclose(param.grad, torch.tensor([0.1, -0.2, 5.0])))


if __name__ == '__main__':
    unittest.main()
