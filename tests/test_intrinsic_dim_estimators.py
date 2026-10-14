"""
Sanity checks for the geometric intrinsic-dimension estimator (MLE) against
data with EXACTLY known ground-truth dimension: a k-dimensional linear
subspace has zero curvature, so any reasonable estimator must recover k
(within statistical noise) regardless of the ambient embedding dimension.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from utils.intrinsic_dim_estimators import mle_dimension


def _random_linear_subspace(n_samples, true_dim, ambient_dim, seed=0):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(-1, 1, size=(n_samples, true_dim))
    # random orthonormal embedding into a higher ambient dimension - a pure
    # rotation/embedding must not change the intrinsic dimension.
    random_matrix = rng.randn(ambient_dim, true_dim)
    q, _ = np.linalg.qr(random_matrix)
    return coords @ q.T


class MleDimensionTests(unittest.TestCase):
    def test_recovers_known_dimension_of_a_flat_subspace(self):
        for true_dim in (2, 5, 10):
            with self.subTest(true_dim=true_dim):
                X = _random_linear_subspace(3000, true_dim, ambient_dim=true_dim * 4, seed=true_dim)
                estimate = mle_dimension(X, k1=10, k2=20)
                self.assertAlmostEqual(estimate, true_dim, delta=max(1.0, 0.3 * true_dim))

    def test_embedding_dimension_does_not_change_estimate(self):
        # Same intrinsic points, embedded into two different ambient
        # dimensions - the estimate should not depend on the ambient dim.
        rng = np.random.RandomState(1)
        coords = rng.uniform(-1, 1, size=(2000, 5))
        low = coords @ np.linalg.qr(rng.randn(8, 5))[0].T
        high = coords @ np.linalg.qr(rng.randn(50, 5))[0].T
        est_low = mle_dimension(low)
        est_high = mle_dimension(high)
        self.assertAlmostEqual(est_low, est_high, delta=1.0)


if __name__ == '__main__':
    unittest.main()
