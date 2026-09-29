"""
Sanity checks for geometry_diagnostics.py on small, hand-interpretable cases
before trusting it on real MovieLens matrices - mirrors the spirit of
check_floydwarshall_grad.py (verify the primitive in isolation first).

Cases:
- Tree (star graph): the canonical negatively-curved / hyperbolic discrete
  structure - expect ORC strongly negative on every edge, and small
  algebraic connectivity (a star's leaves are only connected through the
  hub, a classic bottleneck).
- Complete graph (clique): the canonical positively-curved structure -
  expect ORC strongly positive on every edge, and large algebraic
  connectivity (maximally, uniformly connected).
- Circle point cloud (points on a ring): expect exactly one strongly
  persistent H1 feature (the ring itself) via Vietoris-Rips.
- Filled disk (uniformly sampled 2D blob, no hole): expect no persistent H1
  feature of comparable strength - any H1 bars present should be short-lived
  (sampling noise), not a strong, long-lived feature that would be filtered
  by an amplitude test on the geometry module's own noise level.
"""
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_diagnostics import (
    ollivier_ricci_curvature,
    graph_laplacian_spectral,
    persistent_homology,
)


def dense_adj_from_nx(G, n):
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        adj[u, v] = w
        adj[v, u] = w
    return adj


def check_tree_vs_clique():
    """Star-graph hub-leaf edges are actually a degenerate ORC case (~0, not
    negative): a leaf's only neighbor already IS the hub, so transporting the
    hub's neighbor-distribution onto the leaf's is nearly free - confirmed by
    hand-derivation, not a bug (see docs/recsys_paper_diary.md 2026-08-24).
    The standard negative-curvature textbook example needs branching on BOTH
    endpoints of the tested edge: a "double star" (two hubs, each with their
    own leaves, joined by a bridge edge) - the bridge forces transport
    between genuinely disjoint leaf sets, which is where trees actually cost
    more than a single-degree leaf edge does.
    """
    n = 12
    star = nx.star_graph(n - 1)  # hub = node 0, n-1 leaves
    nx.set_edge_attributes(star, 1.0, "weight")
    star_adj = dense_adj_from_nx(star, n)

    clique = nx.complete_graph(n)
    nx.set_edge_attributes(clique, 1.0, "weight")
    clique_adj = dense_adj_from_nx(clique, n)

    # Double star: hub A=0, hub B=1, each with 5 leaves (nodes 2-6 for A,
    # 7-11 for B), plus the bridge edge (0,1). n=12 total, matching the
    # scale of star/clique above.
    double_star = nx.Graph()
    double_star.add_edge(0, 1)
    for leaf in range(2, 7):
        double_star.add_edge(0, leaf)
    for leaf in range(7, 12):
        double_star.add_edge(1, leaf)
    nx.set_edge_attributes(double_star, 1.0, "weight")
    double_star_adj = dense_adj_from_nx(double_star, n)

    orc_star = ollivier_ricci_curvature(star_adj)
    orc_clique = ollivier_ricci_curvature(clique_adj)
    orc_bridge = ollivier_ricci_curvature(double_star_adj)
    spec_star = graph_laplacian_spectral(star_adj)
    spec_clique = graph_laplacian_spectral(clique_adj)

    # Isolate the bridge edge (0,1) specifically - the graph-wide mean is
    # dominated by the 10 hub-leaf edges (each ~0, same degenerate case as
    # the plain star above), which would wash out the bridge's own signal.
    bridge_curvature = next(c for i, j, c in orc_bridge["edges"] if {i, j} == {0, 1})

    print(f"[star/tree]     ORC mean={orc_star['mean']:.4f} f_neg={orc_star['f_neg']:.2f} | "
          f"lambda2={spec_star['lambda2']:.4f} gap={spec_star['spectral_gap']:.4f}")
    print(f"[clique]        ORC mean={orc_clique['mean']:.4f} f_neg={orc_clique['f_neg']:.2f} | "
          f"lambda2={spec_clique['lambda2']:.4f} gap={spec_clique['spectral_gap']:.4f}")
    print(f"[double-star]   bridge edge (0,1) ORC={bridge_curvature:.4f} "
          f"(graph-wide mean={orc_bridge['mean']:.4f}, dominated by hub-leaf edges)")

    assert bridge_curvature < 0, "double-star bridge edge must have negative ORC"
    assert orc_clique["mean"] > 0, "clique must have positive mean ORC"
    assert bridge_curvature < orc_clique["mean"], "double-star bridge must be more negatively curved than clique"
    assert spec_clique["lambda2"] > spec_star["lambda2"], \
        "clique must be better-connected (higher algebraic connectivity) than a star/tree"
    print("check_tree_vs_clique: PASSED\n")


def check_persistent_homology_ring_vs_disk():
    rng = np.random.default_rng(0)
    n = 120

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1) + rng.normal(scale=0.02, size=(n, 2))
    ring_dist = np.linalg.norm(ring[:, None, :] - ring[None, :, :], axis=-1)

    disk = rng.uniform(-1, 1, size=(n, 2))
    disk = disk[np.linalg.norm(disk, axis=1) <= 1]
    disk_dist = np.linalg.norm(disk[:, None, :] - disk[None, :, :], axis=-1)

    ph_ring = persistent_homology(ring_dist, max_edge_length=1.5)
    ph_disk = persistent_homology(disk_dist, max_edge_length=1.5)

    print(f"[ring]  H1 max persistence={ph_ring['h1_max_persistence']:.4f}  H1 count={ph_ring['h1_count']}")
    print(f"[disk]  H1 max persistence={ph_disk['h1_max_persistence']:.4f}  H1 count={ph_disk['h1_count']}")

    assert ph_ring["h1_max_persistence"] > 3 * max(ph_disk["h1_max_persistence"], 1e-6), \
        "a ring's single loop must be far more persistent than any noise-driven H1 feature in a filled disk"
    print("check_persistent_homology_ring_vs_disk: PASSED\n")


if __name__ == "__main__":
    check_tree_vs_clique()
    check_persistent_homology_ring_vs_disk()
    print("All geometry_diagnostics sanity checks PASSED.")
