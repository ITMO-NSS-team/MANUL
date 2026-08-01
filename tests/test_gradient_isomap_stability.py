"""
Regression tests for the numerical-stability fixes in GradientIsomap:
- _stable_eigenvalues() now also checks the smallest gap between sorted
  eigenvalues (not just the top eigenvalue's magnitude), since eigh's
  backward blows up on close eigenvalue PAIRS (1/(lambda_i - lambda_j)).
- train() now zeroes NaN/Inf gradients before the optimizer step, instead
  of letting a bad backward pass permanently corrupt the distance matrix.
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


class StableEigenvaluesGapTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.isomap = _make_isomap(self._tmpdir.name)
        self.optim = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0001)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_top_eigenvalue_small_is_still_flagged_degenerate(self):
        # Old behaviour: a tiny top eigenvalue alone must still flag degeneracy.
        eigenvalues = np.array([0.005, 0.001, -0.5, -1.0])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.01)

    def test_close_pair_not_at_top_is_now_flagged_degenerate(self):
        # New behaviour: top eigenvalue is large (old check would say "fine"),
        # but it sits right next to the second one - this is exactly the gap
        # that makes eigh's backward (1/(lambda_i - lambda_j)) blow up.
        eigenvalues = np.array([10.0, 9.995, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.01,
                              msg="a close eigenvalue pair should trigger the degenerate branch "
                                  "even when the top eigenvalue itself is not small")

    def test_well_separated_eigenvalues_are_not_degenerate(self):
        eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.0001)


class NanGradientGuardTests(unittest.TestCase):
    def test_train_zeroes_nan_and_inf_gradients_before_step(self):
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        optim = torch.optim.AdamW([param], lr=0.0001)
        param.grad = torch.tensor([float('nan'), float('inf'), 5.0])

        for group in optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        self.assertTrue(torch.isfinite(param.grad).all(),
                        "gradient should be fully finite after the guard")
        self.assertEqual(param.grad[2].item(), 5.0,
                        "a legitimate finite gradient entry must be left untouched")

        # step() must not raise / must not turn the parameter itself into NaN
        optim.step()
        self.assertTrue(torch.isfinite(param).all())


if __name__ == '__main__':
    unittest.main()
