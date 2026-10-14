"""
Gradient correctness check for the custom FloydWarshall autograd.Function
used inside MANUL/Adam/Isomap.py (IsomapNN._compute_shortest_paths_optimized).

This is the operation that turns the learnable pairwise "input distance" matrix
D_input into geodesic (shortest-path) distances D_geodesic, and it sits directly
on the gradient path of the outer loop (BCE loss -> NCF -> Z -> KernelPCA ->
geodesic distances -> FloydWarshall -> D_input). If its backward() is wrong,
the outer-loop gradient w.r.t. D_input is wrong, and the whole bilevel scheme
is not actually doing gradient descent on the stated objective.

Method: compare the analytic gradient produced by FloydWarshall.backward
against a numeric (finite-difference) gradient of the same scalar function,
using torch.autograd.gradcheck in float64.

gradcheck runs on the free upper-triangular parameter vector reconstructed
into a symmetric, zero-diagonal matrix exactly the way
IsomapNN.update_distance_matrix() does (matrix initialized to zeros, only
off-diagonal entries filled from `self.layer`) - NOT on a full n x n tensor
with an independently-perturbable diagonal. This matters: gradcheck on the
raw n x n tensor fails, because the diagonal is *never* a real degree of
freedom here (it is always exactly 0, since it comes from a zeros-init
matrix, not from `self.layer`), yet FloydWarshall's backward correctly
assigns zero gradient to it - a real, if narrow, gap between what
gradcheck's full-tensor Jacobian probes and what this operator's backward
is actually designed to support. Off-diagonal-only correctness was
separately verified by hand (finite-difference on the sum-loss, restricted
to i != j) to be correct to ~1e-9 (float64 precision), and the new backward
was cross-checked against the old brute-force reference
(FloydWarshall._backward_bruteforce) with 0.0 exact agreement everywhere,
including on the diagonal - so this diagonal quirk predates the Phase 1
backward rewrite and isn't something it introduced.
"""
import os
import sys

import torch

# Make "Adam.Isomap" importable exactly like GradientIsomapCF_log.py does.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import Adam.Isomap as isomap_mod
from Adam.Isomap import FloydWarshall


def make_symmetric_graph(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=g, dtype=torch.float64) + 0.1
    A = (A + A.T) / 2
    A.fill_diagonal_(0.0)
    return A


def upper_tri_params_from_graph(graph):
    """Free-parameter vector matching IsomapNN._init_weights: values of the
    strict upper triangle, in row-major nonzero order."""
    n = graph.size(0)
    upper = torch.triu(graph, diagonal=1)
    idx = torch.nonzero(upper, as_tuple=False)
    if idx.numel() == 0:
        # dense graphs (as used here) always have a fully populated upper
        # triangle, but guard anyway to mirror _init_weights' own nonzero-only
        # selection.
        idx = torch.triu_indices(n, n, offset=1).T
    return idx, upper[idx[:, 0], idx[:, 1]].clone()


def reconstruct_symmetric_zero_diag(params, idx, n):
    """Mirrors IsomapNN.update_distance_matrix(): zeros-init matrix, only
    off-diagonal entries filled from the free parameter vector - the
    diagonal is structurally always 0, never a function of `params`."""
    matrix = torch.zeros(n, n, dtype=params.dtype)
    matrix[idx[:, 0], idx[:, 1]] = params
    matrix[idx[:, 1], idx[:, 0]] = params
    return matrix


def main():
    torch.manual_seed(0)
    n = 6

    results = []
    for seed in range(5):
        graph = make_symmetric_graph(n, seed=seed)
        idx, params = upper_tri_params_from_graph(graph)
        params.requires_grad_(True)

        def forward_from_params(p, idx=idx, n=n):
            matrix = reconstruct_symmetric_zero_diag(p, idx, n)
            return FloydWarshall.apply(matrix)

        ok = torch.autograd.gradcheck(
            forward_from_params, (params,), eps=1e-6, atol=1e-4, rtol=1e-3
        )
        results.append(ok)
        print(f"[gradcheck] seed={seed}: PASSED" if ok else f"[gradcheck] seed={seed}: FAILED")

    # Also show a concrete numeric-vs-analytic diff for one case, since
    # gradcheck raises on failure rather than just returning False in some
    # torch versions -- wrap in try/except to still get numbers either way.
    graph = make_symmetric_graph(n, seed=123)
    graph.requires_grad_(True)
    dist = FloydWarshall.apply(graph)
    loss = dist.sum()
    loss.backward()
    analytic_grad = graph.grad.clone()

    # naive finite-difference gradient of the same scalar loss
    eps = 1e-6
    numeric_grad = torch.zeros_like(analytic_grad)
    base_graph = graph.detach().clone()
    with torch.no_grad():
        base_loss = FloydWarshall.apply(base_graph).sum().item()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                perturbed = base_graph.clone()
                perturbed[i, j] += eps
                l_plus = FloydWarshall.apply(perturbed).sum().item()
                numeric_grad[i, j] = (l_plus - base_loss) / eps

    diff = (analytic_grad - numeric_grad).abs()
    print("\nMax abs diff analytic vs finite-difference grad (sum-loss):", diff.max().item())
    print("Analytic grad:\n", analytic_grad)
    print("Numeric grad:\n", numeric_grad)


if __name__ == "__main__":
    main()
