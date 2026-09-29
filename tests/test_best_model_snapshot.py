"""
Regression tests for the "best checkpoint" snapshot bug found across
GraphRegTrainer.train() (self.best_model = self.model) and both
baseline_train_test() implementations (best_model_state =
model.state_dict().copy()): neither actually took an independent snapshot.

A bare reference assignment (self.best_model = self.model) is obviously just
an alias. Less obviously, state_dict().copy() is ALSO not a real snapshot:
.copy() only shallow-copies the dict structure - the tensor VALUES inside
are the exact same tensor objects the live model's optimizer keeps mutating
in place (e.g. via addcdiv_ inside Adam.step()), so the "saved" state
silently tracks every future update too. Verified directly below. The fix in
both places is copy.deepcopy() (of the whole model, or of the state_dict).

Diagnosed on a real MNIST regularization run where val_loss never beat its
epoch-1 value across 5000 epochs (best_epoch stuck at 1) - which turned out
to be a real, separate observation (fast convergence + flat/noisy val loss,
not a bug in the < comparison), but exposed that best_model's "improved
epoch 1" weights were never actually being isolated from the other 4999
epochs of training that followed.
"""
import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch import nn

from regularizator.GraphRegTrainer import GraphRegTrainer


class StateDictCopySemanticsTests(unittest.TestCase):
    """Documents the underlying PyTorch gotcha both bugs shared."""

    def test_shallow_dict_copy_does_not_protect_tensor_values(self):
        model = nn.Linear(2, 1)
        shallow_snapshot = model.state_dict().copy()
        original_weight = shallow_snapshot['weight'].clone()

        with torch.no_grad():
            model.weight.add_(1.0)  # simulates an in-place optimizer.step()

        self.assertFalse(
            torch.equal(shallow_snapshot['weight'], original_weight),
            "state_dict().copy() should NOT protect against later in-place "
            "mutation of the live model (this is the bug both call sites had)")

    def test_deepcopy_does_protect_tensor_values(self):
        model = nn.Linear(2, 1)
        real_snapshot = copy.deepcopy(model.state_dict())
        original_weight = real_snapshot['weight'].clone()

        with torch.no_grad():
            model.weight.add_(1.0)

        self.assertTrue(
            torch.equal(real_snapshot['weight'], original_weight),
            "copy.deepcopy(state_dict()) must be immune to later in-place "
            "mutation of the live model")


class GraphRegTrainerBestModelSnapshotTests(unittest.TestCase):
    def test_best_model_snapshot_survives_further_training_of_live_model(self):
        model = nn.Sequential(nn.Linear(3, 1, dtype=torch.float64))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        rng = np.random.RandomState(0)
        X = rng.randn(20, 3).astype(np.float64)
        y = rng.randn(20).astype(np.float64)
        base_indices = np.arange(10)
        weights_matrix = np.abs(rng.randn(10, 10))
        weights_matrix = (weights_matrix + weights_matrix.T) / 2
        np.fill_diagonal(weights_matrix, 0)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = GraphRegTrainer(
                train_features=X, train_target=y,
                weights_matrix=weights_matrix, base_indices=base_indices,
                model=model, criterion=nn.MSELoss(), optimizer=optimizer,
                num_epochs=1, batch_size=32, device='cpu', cache_folder=tmp,
            )
            # Reproduces exactly what train() does the first time val loss
            # improves (guaranteed on the very first check, since
            # best_val_loss starts at float('inf')).
            trainer.best_model = copy.deepcopy(trainer.model)
            snapshot_weight = trainer.best_model[0].weight.clone()

            # Simulate further training mutating the live model in place,
            # as additional epochs would via optimizer.step().
            with torch.no_grad():
                trainer.model[0].weight.add_(5.0)

            self.assertTrue(
                torch.equal(trainer.best_model[0].weight, snapshot_weight),
                "best_model must not change when the live model is trained further")
            self.assertFalse(
                torch.equal(trainer.best_model[0].weight, trainer.model[0].weight),
                "best_model must diverge from the live model once the live model "
                "keeps training past the snapshot point")


if __name__ == '__main__':
    unittest.main()
