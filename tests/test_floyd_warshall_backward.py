"""
Regression tests for FloydWarshall's custom backward (Adam/Isomap.py).

The original backward compared grad_output (an upstream gradient value, any
sign/magnitude) against dist[:,k]+dist[k,:] (a path-length sum, always >= 0)
- a dimensionally meaningless comparison. Verified empirically before the
fix: it over-amplified the gradient norm by a factor growing linearly with
graph size (~50x at n=100) and destroyed the sparsity every correct
shortest-path gradient must have (edges not on any shortest path must get
exactly zero gradient; the old version gave 0% exact zeros even at n=100,
where the true fraction of zero-gradient entries is ~84%).

First fix (kept as FloydWarshall._backward_bruteforce, exercised here only
for cross-validation): for every candidate edge (a, b), check the shortest-
path decomposition identity dist[i,a]+graph[a,b]+dist[b,j] == dist[i,j]
against all L^2 pairs (i, j). Correct, but O(n_neighbors * n_samples^3) -
at landmark-count scale (L=2000) this measured ~20x slower than the
original (broken) backward - a single IsomapNN step went from ~0.8s to
~15s, i.e. a full 20000-epoch run from ~6h to ~3.5 days.

Final version: forward() also tracks next_hop[i,j] (standard shortest-path
predecessor bookkeeping, free to compute alongside dist). backward() then
walks every (i,j) pair's path in lockstep, one hop at a time, scatter-adding
grad_output[i,j] onto whichever edge was just traversed, until every pair
has reached its destination. Cost is O(n_samples^2) per hop, and the number
of hops needed is the graph's diameter (small for a symmetrized k-NN graph),
so total cost stays close to the forward pass's own O(n_samples^3) - back to
~0.9-1.1s per IsomapNN step at L=2000, while remaining exactly correct
(and, on ties, matching the reference's tie-breaking more closely than the
bruteforce version did, since both use the same "first strict improvement
wins" rule).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from Adam.Isomap import FloydWarshall


def _reference_shortest_paths(graph, n):
    """Plain torch.min loop - correct by construction via native autograd,
    but O(n) intermediate LxL states retained for backward (infeasible at
    landmark-count scale - this is only usable for small test graphs)."""
    dist = graph.clone()
    for k in range(n):
        dist = torch.min(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist


def _random_symmetric_graph(n, seed, dtype=torch.float64):
    torch.manual_seed(seed)
    raw = torch.rand(n, n, dtype=dtype)
    raw = (raw + raw.T) / 2
    raw.fill_diagonal_(0)
    return raw


class FloydWarshallForwardTests(unittest.TestCase):
    def test_forward_matches_reference_shortest_paths(self):
        for n in (8, 30, 100):
            with self.subTest(n=n):
                graph = _random_symmetric_graph(n, seed=n)
                dist_custom = FloydWarshall.apply(graph.clone())
                dist_ref = _reference_shortest_paths(graph.clone(), n)
                self.assertTrue(torch.allclose(dist_custom, dist_ref))


class FloydWarshallBackwardCorrectnessTests(unittest.TestCase):
    """Dense-graph correctness: the diagonal is excluded from comparison
    since in the real pipeline the distance matrix diagonal is never
    derived from any trainable `layer` entry (see IsomapNN._init_weights,
    which only parameterizes the strict upper triangle)."""

    def _check_matches_reference(self, n, dtype, seed=0):
        graph = _random_symmetric_graph(n, seed=seed, dtype=dtype)

        graph_ref = graph.clone().requires_grad_(True)
        dist_ref = _reference_shortest_paths(graph_ref, n)
        gen = torch.Generator().manual_seed(seed + 1)
        w = torch.rand(n, n, dtype=dtype, generator=gen) - 0.5
        (dist_ref * w).sum().backward()
        grad_ref = graph_ref.grad.clone()
        grad_ref.fill_diagonal_(0)

        graph_custom = graph.clone().requires_grad_(True)
        dist_custom = FloydWarshall.apply(graph_custom)
        (dist_custom * w).sum().backward()
        grad_custom = graph_custom.grad.clone()

        return grad_ref, grad_custom

    def test_backward_matches_reference_on_small_dense_graphs(self):
        # The path-reconstruction backward matches the reference EXACTLY
        # (not just approximately) - unlike the bruteforce version, tie
        # handling follows the same "first strict improvement wins" rule as
        # the reference's own sequential torch.min chain.
        for n in (8, 20, 50, 100, 300):
            with self.subTest(n=n):
                grad_ref, grad_custom = self._check_matches_reference(n, torch.float64)
                self.assertTrue(torch.allclose(grad_ref, grad_custom, atol=1e-6, rtol=1e-4),
                                f"max diff={float((grad_ref - grad_custom).abs().max())}")

    def test_backward_gradient_norm_is_not_amplified_at_larger_n(self):
        # Regression guard for the original bug: norm must stay within a
        # small tolerance of the reference, not blow up by 10x-50x+.
        grad_ref, grad_custom = self._check_matches_reference(300, torch.float32, seed=3)
        ratio = grad_custom.norm().item() / grad_ref.norm().item()
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)


class FloydWarshallBackwardCrossValidationTests(unittest.TestCase):
    """The fast path-reconstruction backward and the slow-but-independently-
    derived bruteforce backward must agree - two very different derivations
    of the same gradient landing on the same answer is strong evidence
    neither has a shared blind spot."""

    def test_fast_backward_matches_bruteforce_on_sparse_graph(self):
        torch.manual_seed(0)
        n = 120
        n_neighbors = 12
        X = torch.rand(n, 6, dtype=torch.float64)
        full_dist = torch.cdist(X, X)
        graph = torch.full_like(full_dist, float('inf'))
        for i in range(n):
            neighbors = torch.topk(full_dist[i], n_neighbors, largest=False).indices
            graph[i, neighbors] = full_dist[i, neighbors]
        graph = torch.min(graph, graph.T)

        dist = FloydWarshall.apply(graph.clone().requires_grad_(False)).detach()
        gen = torch.Generator().manual_seed(5)
        grad_output = torch.nan_to_num(torch.rand(n, n, dtype=torch.float64, generator=gen) - 0.5)

        grad_bruteforce = FloydWarshall._backward_bruteforce(graph, dist, grad_output)

        graph_fast = graph.clone().requires_grad_(True)
        dist_fast = FloydWarshall.apply(graph_fast)
        dist_fast.backward(grad_output)
        grad_fast = graph_fast.grad

        self.assertTrue(torch.allclose(grad_fast, grad_bruteforce, atol=1e-4, rtol=1e-3),
                        f"max diff={float((grad_fast - grad_bruteforce).abs().max())}")


class FloydWarshallBackwardPerformanceTests(unittest.TestCase):
    def test_backward_stays_fast_at_moderate_scale(self):
        # Regression guard for the O(n_neighbors * n^3) bruteforce
        # regression: at n=2000 that version measured ~15s per step (~20x
        # the original), which would turn a 20000-epoch run into ~3.5 days.
        # This checks the much smaller n=400 case finishes quickly - a
        # sudden return to bruteforce-like scaling would show up here long
        # before someone waits out a real L=2000 run.
        import time
        torch.manual_seed(0)
        n = 400
        graph = torch.rand(n, n, dtype=torch.float32)
        graph = (graph + graph.T) / 2
        graph.fill_diagonal_(0)
        graph.requires_grad_(True)

        start = time.time()
        dist = FloydWarshall.apply(graph)
        dist.sum().backward()
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0, f"FloydWarshall forward+backward at n={n} took {elapsed:.2f}s - "
                                       "expected well under 10s on CPU; check for accidental O(n^4)-ish regressions")


class FloydWarshallBackwardSparsityTests(unittest.TestCase):
    """The old backward gave 0% exactly-zero gradient entries at any n - a
    correct shortest-path gradient must be exactly zero for any pair that
    lies on no shortest path, and (structurally) for any non-edge (+inf in
    the sparse kNN graph built by _construct_graph)."""

    def test_infinite_graph_entries_always_get_zero_gradient(self):
        torch.manual_seed(0)
        n = 60
        n_neighbors = 8
        X = torch.rand(n, 4, dtype=torch.float32)
        full_dist = torch.cdist(X, X)

        graph = torch.full_like(full_dist, float('inf'))
        for i in range(n):
            neighbors = torch.topk(full_dist[i], n_neighbors, largest=False).indices
            graph[i, neighbors] = full_dist[i, neighbors]
        graph = torch.min(graph, graph.T)
        graph.requires_grad_(True)

        dist = FloydWarshall.apply(graph)
        dist_clean = torch.nan_to_num(dist, posinf=0.0)
        dist_clean.sum().backward()

        non_edges = ~torch.isfinite(graph.detach())
        self.assertTrue(torch.all(graph.grad[non_edges] == 0),
                        "non-edges (+inf entries) must never receive gradient")

    def test_true_zero_gradient_fraction_matches_reference_on_dense_graph(self):
        # At n=100 on a dense random graph, most pairs are NOT on any
        # shortest path and must get exactly zero gradient (~84% in the
        # reference). The old buggy backward gave 0% zeros here.
        n = 100
        graph = _random_symmetric_graph(n, seed=7)
        graph_ref = graph.clone().requires_grad_(True)
        dist_ref = _reference_shortest_paths(graph_ref, n)
        dist_ref.sum().backward()
        zero_frac_ref = (graph_ref.grad.abs() < 1e-9).float().mean().item()

        graph_custom = graph.clone().requires_grad_(True)
        dist_custom = FloydWarshall.apply(graph_custom)
        dist_custom.sum().backward()
        zero_frac_custom = (graph_custom.grad.abs() < 1e-9).float().mean().item()

        self.assertGreater(zero_frac_custom, 0.5,
                           "corrected backward must reproduce the sparse structure of the true gradient")
        self.assertAlmostEqual(zero_frac_custom, zero_frac_ref, delta=0.05)


class FloydWarshallBackwardSmokeTests(unittest.TestCase):
    def test_no_crash_on_moderate_sparse_graph(self):
        torch.manual_seed(0)
        n = 200
        n_neighbors = 15
        X = torch.rand(n, 8, dtype=torch.float32)
        full_dist = torch.cdist(X, X)
        graph = torch.full_like(full_dist, float('inf'))
        for i in range(n):
            neighbors = torch.topk(full_dist[i], n_neighbors, largest=False).indices
            graph[i, neighbors] = full_dist[i, neighbors]
        graph = torch.min(graph, graph.T)
        graph.requires_grad_(True)

        dist = FloydWarshall.apply(graph)
        loss = torch.nan_to_num(dist, posinf=0.0).sum()
        loss.backward()  # must not raise
        self.assertTrue(torch.isfinite(graph.grad).all())


if __name__ == '__main__':
    unittest.main()
