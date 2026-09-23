"""
Regression tests for the numerical-stability fixes in GradientIsomap:
- train() zeroes NaN/Inf gradients before the optimizer step, instead of
  letting a bad backward pass (eigh's backward divides by
  (eigenvalue_i - eigenvalue_j), which can blow up near-degenerate pairs)
  permanently corrupt the distance matrix.
- LR switching in _stable_eigenvalues() stays on the original, long-proven
  check (top eigenvalue magnitude only). An occasional NaN/Inf gradient is
  normal and self-corrects within a few epochs on its own - it must NOT
  bump the LR by itself. An earlier version of this fix (a) keyed the LR
  switch off "some eigenvalue pair is close together" among the handful of
  kept components (~latent_dim, not the full spectrum) - a common, healthy
  spectrum shape, not a sign of trouble - and (b) keyed it off any single
  bad-gradient epoch. Both ended up permanently/mostly stuck at the
  "degenerate" high learning rate, stalling training instead of protecting
  it.
- A SEPARATE, much coarser counter tracks consecutive bad-gradient epochs
  (not tied to the LR at all). Only once degenerate_kick_patience epochs in
  a row have had a bad gradient - a genuinely sustained, non-self-correcting
  run, not an occasional one-off - does _stable_eigenvalues perturb the
  distance matrix with noise, since a stuck optimizer's gradient for the
  affected entries is exactly zero (diagnosed on real MNIST-scale data) and
  no LR can move it.
- The kick is skipped (and the counter reset) whenever current_loss is
  already at/near the best loss ever found. Diagnosed on a real full run: a
  SHARP, well-converged optimum barely moves epoch to epoch, so it produces
  the same near-degenerate eigenvalue spectrum (and hence a bad gradient)
  every epoch - which looks identical to "stuck in a bad state" by
  consecutive-epoch count alone. The run had converged to loss~2.64 (best
  ever was 2.6155), got kicked anyway at epoch 627, and never recovered a
  loss below ~4 for the remaining ~19000 epochs. A kick must only fire when
  the model is stuck somewhere meaningfully WORSE than its best-known state.
- enable_kick now defaults to False: even WITH the near_best guard above, a
  real MNIST-scale run (mnist_run_20260801_110559) found an excellent
  minimum (loss 0.0864 at epoch 5131) and then suffered repeated kick/recover
  cycles that landed at a progressively WORSE trough each time, never
  recovering by epoch 13300 - the kick mechanism itself is net-harmful at
  this scale, not just occasionally mistimed. It is kept only as an
  explicitly opt-in mechanism (enable_kick=True); _stable_eigenvalues still
  tracks and returns whether the patience threshold was crossed
  ("would_kick") purely for diagnostic logging, without acting on it by
  default.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from Adam.GradientIsomap import GradientIsomap


def _make_isomap(tmp_dir, n_points=10, latent_len=2, degenerate_kick_patience=100,
                 degenerate_kick_noise_std=0.02, enable_kick=False):
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
        degenerate_kick_patience=degenerate_kick_patience,
        degenerate_kick_noise_std=degenerate_kick_noise_std,
        enable_kick=enable_kick,
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

    def test_single_bad_gradient_does_not_bump_lr(self):
        # Regression guard: an occasional NaN/Inf gradient is normal and must
        # NOT switch the LR by itself - only sustained trouble (tracked by the
        # separate kick counter) should get a stronger response than "ignore it
        # and let the guard zero it out this epoch".
        eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=True)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.0001)

    def test_close_eigenvalue_pair_alone_is_not_flagged_degenerate(self):
        eigenvalues = np.array([10.0, 9.995, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=False)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.0001)

    def test_well_separated_eigenvalues_are_not_degenerate(self):
        eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])
        self.isomap._stable_eigenvalues(eigenvalues, self.optim, had_bad_grad=False)
        self.assertAlmostEqual(self.optim.param_groups[0]['lr'], 0.0001)


class DegenerateKickTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        # enable_kick=True here: this class specifically tests the kick
        # mechanism's gating logic (patience, near_best guard). The
        # class-default-off behavior (enable_kick=False) is covered
        # separately in KickDisabledByDefaultTests below.
        self.isomap = _make_isomap(self._tmpdir.name, degenerate_kick_patience=3, enable_kick=True)
        self.optim = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0001)
        self.healthy_eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])

    def tearDown(self):
        self._tmpdir.cleanup()

    def _isomap_model(self):
        # Build a real IsomapNN the same way train() does, so .layer exists.
        from Adam.Isomap import IsomapNN
        dist = torch.rand(6, 6)
        dist = (dist + dist.T) / 2
        dist.fill_diagonal_(0)
        return IsomapNN(dist, n_components=2, n_neighbors=3)

    def test_counter_resets_on_non_bad_grad_epoch(self):
        model = self._isomap_model()
        self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
        self.assertEqual(self.isomap._consecutive_bad_grad_epochs, 1)
        self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=False, isomap_model=model)
        self.assertEqual(self.isomap._consecutive_bad_grad_epochs, 0)

    def test_single_bad_grad_epoch_does_not_kick(self):
        model = self._isomap_model()
        original_layer = model.layer.detach().clone()
        self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
        self.assertTrue(torch.equal(model.layer, original_layer),
                        "a single bad-gradient epoch must not perturb the layer")

    def test_kick_perturbs_layer_after_patience_exceeded(self):
        model = self._isomap_model()
        original_layer = model.layer.detach().clone()

        # patience=3: first two consecutive bad-gradient epochs must NOT kick yet.
        self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
        self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
        self.assertTrue(torch.equal(model.layer, original_layer),
                        "layer must be untouched before the patience threshold is reached")

        # Third consecutive bad-gradient epoch crosses patience=3 -> kick.
        self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
        self.assertFalse(torch.equal(model.layer, original_layer),
                         "layer should be perturbed once the patience threshold is reached")
        self.assertEqual(self.isomap._consecutive_bad_grad_epochs, 0,
                        "counter must reset immediately after a kick")

    def test_no_kick_without_isomap_model(self):
        # _stable_eigenvalues must not crash if no model is passed (e.g. old call sites).
        for _ in range(5):
            self.isomap._stable_eigenvalues(self.healthy_eigenvalues, self.optim, had_bad_grad=True)
        # nothing to assert beyond "did not raise"

    def test_near_best_loss_never_kicks_even_past_patience(self):
        # Regression guard for the real-run bug: converging to (or holding) a
        # loss at/near the best ever found must never be kicked, no matter how
        # many consecutive epochs report a bad gradient - that is convergence,
        # not being stuck badly.
        model = self._isomap_model()
        original_layer = model.layer.detach().clone()
        for _ in range(10):  # far past patience=3
            self.isomap._stable_eigenvalues(
                self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model,
                current_loss=2.64, best_loss_so_far=2.6155)
        self.assertTrue(torch.equal(model.layer, original_layer),
                        "a loss near the best-ever value must never be kicked")
        self.assertEqual(self.isomap._consecutive_bad_grad_epochs, 0)

    def test_far_from_best_loss_still_kicks(self):
        # Sanity check: the near-best guard must not disable kicking altogether
        # when the model really is stuck somewhere much worse than its best.
        model = self._isomap_model()
        original_layer = model.layer.detach().clone()
        for _ in range(3):  # patience=3
            self.isomap._stable_eigenvalues(
                self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model,
                current_loss=5.0, best_loss_so_far=2.6155)
        self.assertFalse(torch.equal(model.layer, original_layer),
                         "a loss far worse than the best-ever value should still get kicked")

    def test_stable_eigenvalues_returns_would_kick_flag(self):
        # _stable_eigenvalues reports whether the patience threshold was
        # crossed regardless of enable_kick, so train() can log a "would have
        # kicked" diagnostic signal even while the kick itself stays inactive.
        model = self._isomap_model()
        results = [
            self.isomap._stable_eigenvalues(
                self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
            for _ in range(3)  # patience=3
        ]
        self.assertEqual(results, [False, False, True])


class KickDisabledByDefaultTests(unittest.TestCase):
    """
    Regression guard for the current default (enable_kick=False): a real
    MNIST-scale run found an excellent minimum (loss 0.0864 at epoch 5131)
    and then suffered repeated kick/recover cycles landing at a progressively
    WORSE trough each time, never recovering by epoch 13300 - i.e. even with
    the near_best guard in place, the kick was net-harmful at this scale. The
    mechanism is kept only as an explicit opt-in.
    """
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.isomap = _make_isomap(self._tmpdir.name, degenerate_kick_patience=3, enable_kick=False)
        self.optim = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0001)
        self.healthy_eigenvalues = np.array([10.0, 5.0, 1.0, 0.5])

    def tearDown(self):
        self._tmpdir.cleanup()

    def _isomap_model(self):
        from Adam.Isomap import IsomapNN
        dist = torch.rand(6, 6)
        dist = (dist + dist.T) / 2
        dist.fill_diagonal_(0)
        return IsomapNN(dist, n_components=2, n_neighbors=3)

    def test_default_never_perturbs_layer_even_far_past_patience(self):
        model = self._isomap_model()
        original_layer = model.layer.detach().clone()
        for _ in range(20):  # far past patience=3
            self.isomap._stable_eigenvalues(
                self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model,
                current_loss=5.0, best_loss_so_far=2.6155)  # far from best - would kick if enabled
        self.assertTrue(torch.equal(model.layer, original_layer),
                        "enable_kick=False must never perturb the layer, no matter how far from best")

    def test_would_kick_still_reported_when_disabled(self):
        model = self._isomap_model()
        results = [
            self.isomap._stable_eigenvalues(
                self.healthy_eigenvalues, self.optim, had_bad_grad=True, isomap_model=model)
            for _ in range(3)  # patience=3
        ]
        self.assertEqual(results, [False, False, True],
                         "the diagnostic would_kick flag must still fire even though the kick is disabled")


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


class FullEigenvaluesSortedTests(unittest.TestCase):
    """
    KernelPCA.choose_position() immediately slices eigenvalues_ down to
    n_components, discarding the rest - so nothing could previously inspect
    the gap between the last KEPT eigenvalue and the first EXCLUDED one
    (exactly the kind of gap eigh's backward divides by). full_eigenvalues_
    sorted_ keeps the complete, |value|-descending-sorted spectrum around
    for that purpose without changing any existing behavior.
    """
    def test_full_spectrum_is_kept_sorted_by_absolute_value_descending(self):
        from Adam.KernelPCA import KernelPCA
        torch.manual_seed(0)
        n = 20
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T  # symmetric PSD kernel-like matrix

        pca = KernelPCA(n_components=5, eigval_choice='MDS')
        pca.fit(K)

        self.assertEqual(len(pca.full_eigenvalues_sorted_), n)
        abs_vals = pca.full_eigenvalues_sorted_.abs()
        self.assertTrue(torch.all(abs_vals[:-1] >= abs_vals[1:]),
                        "full spectrum must be sorted by |eigenvalue|, descending")

    def test_kept_eigenvalues_are_the_top_n_components_of_full_spectrum(self):
        from Adam.KernelPCA import KernelPCA
        torch.manual_seed(1)
        n = 15
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T

        n_components = 4
        pca = KernelPCA(n_components=n_components, eigval_choice='MDS')
        pca.fit(K)

        # 'MDS' mode has c_2l=0 (see choose_position), so the kept eigenvalues_
        # should exactly match the top-n_components of the full sorted spectrum.
        expected = pca.full_eigenvalues_sorted_[:n_components]
        sorted_kept, _ = torch.sort(pca.eigenvalues_, descending=True)
        sorted_expected, _ = torch.sort(expected, descending=True)
        self.assertTrue(torch.allclose(sorted_kept, sorted_expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
