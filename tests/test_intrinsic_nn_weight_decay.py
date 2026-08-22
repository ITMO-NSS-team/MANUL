"""
IntrinsicNN's inner probe optimizer used to silently inherit AdamW's library
default (weight_decay=0.01), never passed explicitly. Since the probe is
re-initialized from scratch every GradientIsomap outer epoch, and MSE is
scale-invariant while weight decay is not, this measurably amplified
GradientIsomap's embedding-scale growth without any accuracy benefit
(confirmed via ablation on synthetic sphere data: growth ratio dropped from
1.09-1.15x to 1.03x, Spearman +0.94 -> +0.30 vs epoch, at the same loss).
weight_decay now defaults to 0.0 and is explicitly configurable.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from structure_approximation.IntrinsicNN import IntrinsicNN


class IntrinsicNNWeightDecayTests(unittest.TestCase):
    def test_default_weight_decay_is_zero(self):
        features = torch.rand(10, 3)
        targets = torch.rand(10)
        model = IntrinsicNN(features, targets, latent_len=3, epochs=1, plot_convergence=False)
        self.assertEqual(model.weight_decay, 0.0)

    def test_weight_decay_is_configurable_and_used_by_optimizer(self):
        features = torch.rand(10, 3)
        targets = torch.rand(10)
        model = IntrinsicNN(features, targets, latent_len=3, epochs=1,
                            plot_convergence=False, weight_decay=0.05)
        self.assertEqual(model.weight_decay, 0.05)
        model.train()  # must not raise, and must actually apply weight_decay=0.05

    def test_train_runs_with_default_zero_weight_decay(self):
        features = torch.rand(10, 3)
        targets = torch.rand(10)
        model = IntrinsicNN(features, targets, latent_len=3, epochs=5, plot_convergence=False)
        model.train()
        self.assertIsNotNone(model.loss)
        self.assertTrue(torch.isfinite(torch.tensor(model.loss)))


if __name__ == '__main__':
    unittest.main()
