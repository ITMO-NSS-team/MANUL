"""
Geometric/topological diagnostics of the distance matrices produced by
GradientIsomapCF (recsys/GradIsomapCF_movielens/GradientIsomapCF_log.py),
covering the measures described in the paper draft's Section 3.3 that
delta_hyperbolicity (analyze_hyperbolicity.py) does not: Ollivier-Ricci
curvature, graph-Laplacian spectral analysis, isometric embedding quality
(Kruskal stress / Spearman rho / residual variance), and persistent
homology (H0/H1 via Vietoris-Rips).

Same input contract as analyze_hyperbolicity.py: reads matrices_epoch{k}.npz
files saved by GradientIsomapCF_log.save_epoch_matrices(), which contain
D_input, D_geodesic, knn_adj, Z, D_latent.

Ollivier-Ricci curvature and the graph-Laplacian spectral gap are computed
on the kNN graph (knn_adj) - the natural sparse graph structure the paper
calls "the item interaction graph" - not on the dense D_geodesic/D_latent
matrices, which have no intrinsic edge set of their own. Isometric-embedding
quality and persistent homology are computed on D_geodesic and D_latent, per
Section 3.3.4/3.3.5 of the draft.
"""
import argparse
import glob
import os
import re
import sys
import time

import gudhi
import networkx as nx
import numpy as np
import ot
from scipy.stats import spearmanr, pearsonr


def _progress(msg: str) -> None:
    """Flushed, timestamped progress line - so a long-running call keeps
    producing visible output instead of going silent until it returns
    (which is indistinguishable from a hang when watching the terminal)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def graph_from_knn_adj(knn_adj: np.ndarray) -> nx.Graph:
    """Build an undirected weighted NetworkX graph from a dense, symmetric
    kNN adjacency matrix (zero = no edge), as saved in knn_adj."""
    G = nx.Graph()
    n = knn_adj.shape[0]
    G.add_nodes_from(range(n))
    iu, ju = np.triu_indices(n, k=1)
    weights = knn_adj[iu, ju]
    mask = weights > 0
    edges = [(int(i), int(j), {"weight": float(w)}) for i, j, w in zip(iu[mask], ju[mask], weights[mask])]
    G.add_edges_from(edges)
    return G


def _neighbor_distribution(G: nx.Graph, node: int, alpha: float) -> dict:
    """Lazy random-walk neighbor distribution used by Ollivier-Ricci
    curvature: mass alpha stays at `node` itself, mass (1-alpha) is spread
    uniformly over its graph neighbors. Returns {node_id: probability}."""
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return {node: 1.0}
    dist = {node: alpha}
    share = (1.0 - alpha) / len(neighbors)
    for nb in neighbors:
        dist[nb] = dist.get(nb, 0.0) + share
    return dist


def ollivier_ricci_curvature(knn_adj: np.ndarray, alpha: float = 0.5) -> dict:
    """Edge-level Ollivier-Ricci curvature on the kNN graph:
    kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j), per Section 3.3.2 of the draft
    (mu_i/mu_j are lazy-random-walk neighbor distributions, W1 is exact
    Wasserstein-1/earth-mover distance between them under shortest-path
    ground-truth distances).

    Returns summary statistics (mean, median, fraction negative) plus the
    raw per-edge curvature array, matching the paper draft's reporting
    (mean/median ORC, f_neg) in Section 4.

    SAFETY / WHY THIS IS HAND-ROLLED rather than using the
    `GraphRicciCurvature` package (which implements exactly this
    computation): two separate, serious problems surfaced when it was tried
    first (see docs/recsys_paper_diary.md 2026-08-24 for the full incident):
    (1) its default proc=os.cpu_count() (=28 on this machine) unconditionally
    spawns that many multiprocessing worker processes even for a trivial
    graph, which caused a real, severe system-wide freeze; (2) even at
    proc=1 it still goes through multiprocessing.Pool, which is broken on
    Windows for this library regardless of worker count (spawned workers
    don't inherit the module-level globals it relies on, unlike fork on
    Linux) - and its OTD solver path additionally hardcodes cvxpy's
    "ECOS_BB", which isn't installed here. Rather than keep patching around
    a library not well-suited to this environment, this implements the
    same well-defined formula directly and safely: no multiprocessing
    anywhere, using `ot.emd2` (POT, already a project dependency) for the
    exact Wasserstein-1 solve, which is both simpler to reason about and
    faster than routing through a generic LP solver for this problem
    structure.
    """
    G = graph_from_knn_adj(knn_adj)
    if G.number_of_edges() == 0:
        return {"mean": float("nan"), "median": float("nan"), "f_neg": float("nan"),
                "n_edges": 0, "values": np.array([])}

    n_edges_total = G.number_of_edges()
    _progress(f"ORC: computing all-pairs shortest paths ({G.number_of_nodes()} nodes)...")
    apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    _progress(f"ORC: shortest paths done, computing curvature for {n_edges_total} edges...")

    # At most ~4 progress lines regardless of graph size - at this project's
    # scale (a few thousand edges) the whole loop finishes in ~1s, so
    # printing every N edges (a fixed N) produced far more lines than the
    # actual runtime warranted and triggered Monitor's noise-rate-limit
    # during the eta_outer sweep (see docs/recsys_paper_diary.md 2026-08-24).
    progress_every = max(1, n_edges_total // 4)
    t0 = time.time()
    for edge_idx, (i, j) in enumerate(G.edges()):
        if edge_idx > 0 and edge_idx % progress_every == 0:
            rate = edge_idx / (time.time() - t0)
            eta_s = (n_edges_total - edge_idx) / rate if rate > 0 else float("nan")
            _progress(f"ORC: edge {edge_idx}/{n_edges_total} ({rate:.0f}/s, ETA {eta_s:.0f}s)")

        d_ij = apsp[i][j]
        if d_ij < 1e-12:
            G[i][j]["ricciCurvature"] = 0.0
            continue

        mu_i = _neighbor_distribution(G, i, alpha)
        mu_j = _neighbor_distribution(G, j, alpha)
        support_i = list(mu_i.keys())
        support_j = list(mu_j.keys())

        cost = np.array([[apsp[a][b] for b in support_j] for a in support_i], dtype=np.float64)
        p = np.array([mu_i[a] for a in support_i], dtype=np.float64)
        q = np.array([mu_j[b] for b in support_j], dtype=np.float64)

        w1 = float(ot.emd2(p, q, cost))
        G[i][j]["ricciCurvature"] = 1.0 - w1 / d_ij

    edges = [(i, j, data["ricciCurvature"]) for i, j, data in G.edges(data=True)]
    values = np.array([c for _, _, c in edges])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "f_neg": float(np.mean(values < 0)),
        "n_edges": int(len(values)),
        "values": values,
        "edges": edges,
    }


def graph_laplacian_spectral(knn_adj: np.ndarray) -> dict:
    """Algebraic connectivity (lambda_2, the Fiedler value) and spectral gap
    (gamma = lambda_2 / lambda_max) of the combinatorial graph Laplacian
    L = D - A of the kNN graph, per Section 3.3.3 of the draft.

    Uses the largest connected component if the graph is disconnected (a
    disconnected kNN graph has lambda_2 = 0 by definition over the whole
    graph, which would trivially always report "disconnected" rather than
    describing the connectivity structure of the graph's actual bulk).
    """
    G = graph_from_knn_adj(knn_adj)
    if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
        return {"lambda2": float("nan"), "lambda_max": float("nan"), "spectral_gap": float("nan"),
                "n_components": int(nx.number_connected_components(G))}

    n_components = nx.number_connected_components(G)
    if n_components > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    if G.number_of_nodes() < 3:
        return {"lambda2": float("nan"), "lambda_max": float("nan"), "spectral_gap": float("nan"),
                "n_components": int(n_components)}

    L = nx.laplacian_matrix(G, weight="weight").toarray().astype(np.float64)
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(np.clip(eigvals, a_min=0, a_max=None))  # numerical noise can give tiny negatives

    lambda2 = float(eigvals[1]) if len(eigvals) > 1 else float("nan")
    lambda_max = float(eigvals[-1])
    spectral_gap = (lambda2 / lambda_max) if lambda_max > 0 else float("nan")

    return {"lambda2": lambda2, "lambda_max": lambda_max, "spectral_gap": spectral_gap,
            "n_components": int(n_components)}


def isometric_embedding_quality(D_geodesic: np.ndarray, D_latent: np.ndarray) -> dict:
    """Kruskal Stress-1, Spearman rank correlation, and residual variance
    between geodesic and latent pairwise distances, per Section 3.3.4.
    """
    n = D_geodesic.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    d_geo = D_geodesic[iu, ju].astype(np.float64)
    d_lat = D_latent[iu, ju].astype(np.float64)

    finite = np.isfinite(d_geo) & np.isfinite(d_lat)
    d_geo, d_lat = d_geo[finite], d_lat[finite]

    denom = np.sum(d_geo ** 2)
    stress = float(np.sqrt(np.sum((d_geo - d_lat) ** 2) / denom)) if denom > 0 else float("nan")

    if len(d_geo) > 1 and np.std(d_geo) > 0 and np.std(d_lat) > 0:
        rho, _ = spearmanr(d_geo, d_lat)
        r, _ = pearsonr(d_geo, d_lat)
        residual_variance = 1.0 - r ** 2
    else:
        rho, residual_variance = float("nan"), float("nan")

    return {"kruskal_stress": stress, "spearman_rho": float(rho), "residual_variance": float(residual_variance)}


def _max_persistence(intervals, cap: float) -> float:
    """Max (death - birth) over a set of persistence intervals, treating an
    infinite death as "still alive at the filtration cutoff" (i.e. capped at
    `cap`, the max_edge_length the complex was built with) rather than
    discarding it.

    BUG THIS FIXES: a first version filtered to `np.isfinite(d)` only and
    dropped infinite-death intervals entirely, silently reporting 0.0
    persistence for them - which is exactly backwards, since a feature that
    never dies within the observed filtration range is the STRONGEST
    possible persistence signal, not an absent one. Caught by a sanity test
    (a point cloud on a ring showed H1 max persistence = 0.0 despite having
    exactly one H1 feature - the loop - which turned out to still be alive,
    i.e. un-triangulated-over, at max_edge_length).
    """
    if len(intervals) == 0:
        return 0.0
    capped = [(b, cap if not np.isfinite(d) else d) for b, d in intervals]
    return float(max(d - b for b, d in capped))


def persistent_homology(D: np.ndarray, max_edge_length: float = None, max_dimension: int = 2,
                        max_simplices: int = 300_000) -> dict:
    """H0/H1 persistent homology via Vietoris-Rips on a distance matrix,
    per Section 3.3.5. Reports max persistence per homological degree and
    the number of H1 (loop) features - both directly comparable to the
    paper draft's reporting (max H0/H1 persistence, H1 cycle count).

    SAFETY: a naive "just pick a generous max_edge_length and build the
    complex" approach is dangerous at real scale (n~800) - a median-distance
    threshold admits roughly half of all ~n^2/2 pairs, and expanding that
    into 2-simplices (triangles) can blow up to millions of simplices,
    which is exactly the kind of unbounded memory/CPU spike that can hang a
    machine, not just crash the one process (this was caught in review
    BEFORE ever being run against real data - see docs/recsys_paper_diary.md
    2026-08-24). So the complex is built in stages with a hard size check
    before each expensive step, instead of trusting a distance threshold
    alone to keep it small:
      1. build only the 1-skeleton (nodes+edges, cheap, O(n^2) worst case)
      2. if the edge count already implies a dense-enough graph that
         triangle expansion could blow up, fall back to H0-only (H1 is
         reported as NaN rather than risking the expansion)
      3. otherwise expand to max_dimension and re-check simplex count
         against max_simplices BEFORE compute_persistence() - if it's still
         over budget, abort the same way rather than proceeding

    max_edge_length defaults to the 3rd percentile of pairwise distances.
    Tuned empirically against the real n=800 smoke-test matrices
    (2026-08-24): the original 10th-percentile default gave avg degree ~80,
    which always exceeded max_simplices and silently fell back to H0-only -
    H1 was never actually computed at real scale. Swept 1st-5th percentile:
    1st-3rd all land safely under budget (24-100% margin) and already
    saturate H1 count (~1247 at 1%, ~2187 from 2% up - see
    docs/recsys_paper_diary.md 2026-08-24), and 3% conveniently lands in the
    same H1-cycle-count ballpark the paper draft itself reports
    (1291-2048 across its four experiments) with a comfortable 4x safety
    margin below max_simplices (71699 vs 300000). The staged size checks
    remain the real safety mechanism regardless - this default is just a
    starting point chosen to avoid needing that fallback in the common case.
    """
    n = D.shape[0]
    D = D.astype(np.float64)
    if max_edge_length is None:
        iu, ju = np.triu_indices(n, k=1)
        max_edge_length = float(np.percentile(D[iu, ju], 3))

    _progress(f"PH: building 1-skeleton (n={n}, max_edge_length={max_edge_length:.4f})...")
    rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge_length)

    # Stage 1: 1-skeleton only (nodes + edges) - cheap and bounded by n^2/2.
    simplex_tree = rips.create_simplex_tree(max_dimension=1)
    n_edges = simplex_tree.num_simplices() - simplex_tree.num_vertices()
    avg_degree = (2 * n_edges / n) if n > 0 else 0.0
    _progress(f"PH: 1-skeleton done ({n_edges} edges, avg degree {avg_degree:.1f})")

    # Rough worst-case triangle count for a graph with this average degree
    # (avg_degree choose 2 per node, halved for double-counting) - used only
    # to decide whether it's safe to attempt the dimension-2 expansion at
    # all, not as an exact bound.
    def h0_from_current_tree():
        st_copy = simplex_tree.copy()
        st_copy.compute_persistence()
        h0 = st_copy.persistence_intervals_in_dimension(0)
        return _max_persistence(h0, cap=max_edge_length)

    estimated_triangles = n * avg_degree * avg_degree / 6.0
    if max_dimension >= 2 and estimated_triangles > max_simplices:
        _progress(f"PH: estimated triangles ({estimated_triangles:.0f}) exceeds max_simplices "
                  f"({max_simplices}) - skipping H1 for safety, H0-only.")
        return {
            "h0_max_persistence": h0_from_current_tree(),
            "h1_max_persistence": float("nan"),
            "h1_count": -1,  # sentinel: skipped for safety, not "zero cycles"
            "h0_count": -1,
        }

    if max_dimension >= 2:
        _progress("PH: expanding to dimension 2 (triangles)...")
        simplex_tree.expansion(max_dimension)
        _progress(f"PH: expansion done ({simplex_tree.num_simplices()} simplices total)")
        if simplex_tree.num_simplices() > max_simplices:
            _progress(f"PH: expanded complex ({simplex_tree.num_simplices()} simplices) exceeds "
                      f"max_simplices ({max_simplices}) - skipping H1 for safety, H0-only.")
            return {
                "h0_max_persistence": h0_from_current_tree(),
                "h1_max_persistence": float("nan"),
                "h1_count": -1,
                "h0_count": -1,
            }

    _progress("PH: computing persistence...")
    simplex_tree.compute_persistence()
    _progress("PH: persistence computation done")

    h0 = simplex_tree.persistence_intervals_in_dimension(0)
    h1 = simplex_tree.persistence_intervals_in_dimension(1) if max_dimension >= 2 else []

    return {
        "h0_max_persistence": _max_persistence(h0, cap=max_edge_length),
        "h1_max_persistence": _max_persistence(h1, cap=max_edge_length),
        "h1_count": int(len(h1)),
        "h0_count": int(len(h0)),
    }


def summarize_epoch(data: dict) -> dict:
    """Run all diagnostics on one epoch's saved matrices dict (as loaded
    from a matrices_epoch{k}.npz file)."""
    out = {}

    if "knn_adj" in data:
        _progress("stage: Ollivier-Ricci curvature")
        orc = ollivier_ricci_curvature(data["knn_adj"])
        out.update({f"orc_{k}": v for k, v in orc.items() if k != "values"})
        _progress("stage: graph Laplacian spectral analysis")
        spec = graph_laplacian_spectral(data["knn_adj"])
        out.update({f"spectral_{k}": v for k, v in spec.items()})

    if "D_geodesic" in data and "D_latent" in data:
        _progress("stage: isometric embedding quality")
        iso = isometric_embedding_quality(data["D_geodesic"], data["D_latent"])
        out.update(iso)

    if "D_geodesic" in data:
        _progress("stage: persistent homology (D_geodesic)")
        ph_geo = persistent_homology(data["D_geodesic"])
        out.update({f"geo_{k}": v for k, v in ph_geo.items()})
    if "D_latent" in data:
        _progress("stage: persistent homology (D_latent)")
        ph_lat = persistent_homology(data["D_latent"])
        out.update({f"lat_{k}": v for k, v in ph_lat.items()})

    return out


def analyze_run(logs_folder: str, max_epochs_to_show: int = None) -> list:
    print(f"\n=== Geometry diagnostics for {logs_folder} ===")
    results = []

    epoch_files = glob.glob(os.path.join(logs_folder, "matrices_epoch*.npz"))

    def epoch_num(path):
        m = re.search(r"matrices_epoch(\d+)\.npz", os.path.basename(path))
        return int(m.group(1)) if m else -1

    epoch_files = sorted(epoch_files, key=epoch_num)
    if max_epochs_to_show is not None:
        shown = sorted(set(epoch_num(p) for p in epoch_files))
        keep = set(shown[:1] + shown[-max_epochs_to_show:]) if len(shown) > max_epochs_to_show else set(shown)
        epoch_files = [p for p in epoch_files if epoch_num(p) in keep]

    for file_idx, path in enumerate(epoch_files):
        k = epoch_num(path)
        _progress(f"=== epoch {k} ({file_idx + 1}/{len(epoch_files)} files) ===")
        data = np.load(path)
        row = {"epoch": k}
        row.update(summarize_epoch(data))
        results.append(row)
        print(f"[epoch {k}] ORC mean={row.get('orc_mean', float('nan')):.4f} "
              f"f_neg={row.get('orc_f_neg', float('nan')):.3f} | "
              f"lambda2={row.get('spectral_lambda2', float('nan')):.4f} "
              f"gap={row.get('spectral_spectral_gap', float('nan')):.4f} | "
              f"stress={row.get('kruskal_stress', float('nan')):.4f} "
              f"rho={row.get('spearman_rho', float('nan')):.4f} | "
              f"H1(geo)={row.get('geo_h1_count', 'n/a')} H1(lat)={row.get('lat_h1_count', 'n/a')}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True,
                         help="e.g. .../logs_movielens_isomap_cf/main01")
    parser.add_argument("--max_epochs_to_show", type=int, default=None)
    parser.add_argument("--out_csv", default=None,
                         help="if given, write per-epoch results as CSV here")
    args = parser.parse_args()

    results = analyze_run(args.logs_folder, args.max_epochs_to_show)

    if args.out_csv:
        import pandas as pd
        pd.DataFrame(results).to_csv(args.out_csv, index=False)
        print(f"\n[Save] {args.out_csv}")


if __name__ == "__main__":
    main()
