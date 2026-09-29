import argparse
import os
import time

import numpy as np
import networkx as nx
import ot
import gudhi


# ---------- delta-hyperbolicity ----------
def delta_hyperbolicity(D: np.ndarray, basepoint: int = 0) -> float:
    n = D.shape[0]
    w = basepoint
    dw = D[w, :]
    A = 0.5 * (dw[:, None] + dw[None, :] - D)
    M = np.full((n, n), -np.inf, dtype=D.dtype)
    for y in range(n):
        cand = np.minimum(A[:, y:y + 1], A[y:y + 1, :])
        np.maximum(M, cand, out=M)
    delta = float(np.max(M - A))
    return delta


def calc_delta_rel(D: np.ndarray):
    diam = float(np.max(D))
    delta = delta_hyperbolicity(D, basepoint=0)
    delta_rel = (2.0 * delta / diam) if diam > 0 else float("nan")
    return delta, diam, delta_rel


# ---------- utilities ----------
def _progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_distance_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape={D.shape}")
    # симметризуем на всякий случай
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    # отрицательные из-за числ. шума -> 0
    D[D < 0] = 0.0
    return D


def select_D_from_npz(data: np.lib.npyio.NpzFile, preferred_key: str | None = None):
    if preferred_key is not None:
        if preferred_key not in data:
            raise KeyError(f"Key {preferred_key!r} not in npz. Available: {data.files}")
        return _ensure_distance_matrix(data[preferred_key]), preferred_key

    for k in ["D", "D_latent", "D_geodesic", "D_input"]:
        if k in data:
            return _ensure_distance_matrix(data[k]), k
    raise KeyError(f"No distance matrix found. Available keys: {data.files}")


def knn_adj_from_D(D: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Строим симметричную kNN-матрицу смежности из dense distance matrix D.
    Вес ребра = distance (чем меньше, тем ближе).
    """
    D = _ensure_distance_matrix(D)
    n = D.shape[0]
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")

    knn_adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # берём k ближайших (исключая self)
        idx = np.argpartition(D[i], kth=min(k + 1, n - 1))[: k + 1]
        idx = idx[idx != i]
        if len(idx) > k:
            # аккуратно режем до k по реальной сортировке
            idx = idx[np.argsort(D[i, idx])[:k]]

        for j in idx:
            w = float(D[i, j])
            if w <= 0:
                w = 1e-12
            knn_adj[i, j] = w

    # симметризация: если i выбрал j или j выбрал i — добавляем ребро
    knn_adj = np.minimum(
        np.where(knn_adj > 0, knn_adj, np.inf),
        np.where(knn_adj.T > 0, knn_adj.T, np.inf),
    )
    knn_adj[~np.isfinite(knn_adj)] = 0.0
    np.fill_diagonal(knn_adj, 0.0)
    return knn_adj


def graph_from_knn_adj(knn_adj: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    n = knn_adj.shape[0]
    G.add_nodes_from(range(n))
    iu, ju = np.triu_indices(n, k=1)
    weights = knn_adj[iu, ju]
    mask = weights > 0
    edges = [(int(i), int(j), {"weight": float(w)}) for i, j, w in zip(iu[mask], ju[mask], weights[mask])]
    G.add_edges_from(edges)
    return G


# ---------- ORC ----------
def _neighbor_distribution(G: nx.Graph, node: int, alpha: float) -> dict:
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return {node: 1.0}
    dist = {node: alpha}
    share = (1.0 - alpha) / len(neighbors)
    for nb in neighbors:
        dist[nb] = dist.get(nb, 0.0) + share
    return dist


def ollivier_ricci_curvature(knn_adj: np.ndarray, alpha: float = 0.5) -> dict:
    G = graph_from_knn_adj(knn_adj)
    if G.number_of_edges() == 0:
        return {"mean": float("nan"), "median": float("nan"), "f_neg": float("nan"),
                "n_edges": 0}

    _progress(f"ORC: APSP on graph (n={G.number_of_nodes()}, m={G.number_of_edges()}) ...")
    apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    values = []
    for (i, j) in G.edges():
        d_ij = apsp[i][j]
        if d_ij < 1e-12:
            values.append(0.0)
            continue

        mu_i = _neighbor_distribution(G, i, alpha)
        mu_j = _neighbor_distribution(G, j, alpha)
        support_i = list(mu_i.keys())
        support_j = list(mu_j.keys())

        cost = np.array([[apsp[a][b] for b in support_j] for a in support_i], dtype=np.float64)
        p = np.array([mu_i[a] for a in support_i], dtype=np.float64)
        q = np.array([mu_j[b] for b in support_j], dtype=np.float64)

        w1 = float(ot.emd2(p, q, cost))
        values.append(1.0 - w1 / d_ij)

    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "f_neg": float(np.mean(values < 0)),
        "n_edges": int(values.size),
    }


# ---------- spectral ----------
def graph_laplacian_spectral(knn_adj: np.ndarray) -> dict:
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
    eigvals = np.sort(np.clip(eigvals, a_min=0, a_max=None))
    lambda2 = float(eigvals[1]) if len(eigvals) > 1 else float("nan")
    lambda_max = float(eigvals[-1])
    gap = (lambda2 / lambda_max) if lambda_max > 0 else float("nan")
    return {"lambda2": lambda2, "lambda_max": lambda_max, "spectral_gap": gap,
            "n_components": int(n_components)}


# ---------- persistent homology ----------
def _max_persistence(intervals, cap: float) -> float:
    if len(intervals) == 0:
        return 0.0
    capped = [(b, cap if not np.isfinite(d) else d) for b, d in intervals]
    return float(max(d - b for b, d in capped))


def persistent_homology(D: np.ndarray, max_edge_length: float = None, max_dimension: int = 2,
                        max_simplices: int = 300_000) -> dict:
    D = _ensure_distance_matrix(D)
    n = D.shape[0]
    if max_edge_length is None:
        iu, ju = np.triu_indices(n, k=1)
        max_edge_length = float(np.percentile(D[iu, ju], 3))

    _progress(f"PH: building Rips (n={n}, max_edge_length={max_edge_length:.4f})")
    rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge_length)

    simplex_tree = rips.create_simplex_tree(max_dimension=1)
    n_edges = simplex_tree.num_simplices() - simplex_tree.num_vertices()
    avg_degree = (2 * n_edges / n) if n > 0 else 0.0

    estimated_triangles = n * avg_degree * avg_degree / 6.0
    if max_dimension >= 2 and estimated_triangles > max_simplices:
        simplex_tree.compute_persistence()
        h0 = simplex_tree.persistence_intervals_in_dimension(0)
        return {
            "h0_max_persistence": _max_persistence(h0, cap=max_edge_length),
            "h1_max_persistence": float("nan"),
            "h1_count": -1,
        }

    if max_dimension >= 2:
        simplex_tree.expansion(max_dimension)
        if simplex_tree.num_simplices() > max_simplices:
            st2 = rips.create_simplex_tree(max_dimension=1)
            st2.compute_persistence()
            h0 = st2.persistence_intervals_in_dimension(0)
            return {
                "h0_max_persistence": _max_persistence(h0, cap=max_edge_length),
                "h1_max_persistence": float("nan"),
                "h1_count": -1,
            }

    simplex_tree.compute_persistence()
    h0 = simplex_tree.persistence_intervals_in_dimension(0)
    h1 = simplex_tree.persistence_intervals_in_dimension(1) if max_dimension >= 2 else []
    return {
        "h0_max_persistence": _max_persistence(h0, cap=max_edge_length),
        "h1_max_persistence": _max_persistence(h1, cap=max_edge_length),
        "h1_count": int(len(h1)),
    }


def parse_input(s: str):
    """
    Формат:
      NAME=/path/file.npz
      NAME=/path/file.npz:KEY
    """
    if "=" not in s:
        raise ValueError(f"Bad --input {s!r}. Expected NAME=PATH[:KEY]")
    name, rest = s.split("=", 1)
    if ":" in rest:
        path, key = rest.rsplit(":", 1)
        key = key.strip()
    else:
        path, key = rest, None
    return name.strip(), path.strip(), key


def analyze_one_npz(name: str, path: str, d_key: str | None, knn_k: int):
    data = np.load(path)
    D, used_key = select_D_from_npz(data, preferred_key=d_key)

    if "knn_adj" in data:
        knn_adj = np.asarray(data["knn_adj"], dtype=np.float64)
        knn_source = "npz:knn_adj"
    else:
        knn_adj = knn_adj_from_D(D, k=knn_k)
        knn_source = f"built_from_D(k={knn_k})"

    _progress(f"[{name}] loaded {os.path.basename(path)} | D_key={used_key} | knn={knn_source}")

    delta, diam, delta_rel = calc_delta_rel(D)
    orc = ollivier_ricci_curvature(knn_adj)
    spec = graph_laplacian_spectral(knn_adj)
    ph = persistent_homology(D)

    return {
        "name": name,
        "path": path,
        "D_key": used_key,
        "knn_source": knn_source,
        "delta": delta,
        "diam": diam,
        "delta_rel": delta_rel,
        "orc_mean": orc["mean"],
        "orc_median": orc["median"],
        "orc_f_neg": orc["f_neg"],
        "spectral_lambda2": spec["lambda2"],
        "spectral_gap": spec["spectral_gap"],
        "spectral_n_components": spec["n_components"],
        "ph_h0_max_persistence": ph["h0_max_persistence"],
        "ph_h1_max_persistence": ph["h1_max_persistence"],
        "ph_h1_count": ph["h1_count"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", action="append", required=True,
        help='Repeatable. Format: NAME=PATH[:KEY]. Example: "poincare=/.../file.npz:D"'
    )
    p.add_argument("--knn_k", type=int, default=10, help="k for kNN graph if knn_adj отсутствует в npz")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    rows = []
    for s in args.input:
        name, path, key = parse_input(s)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows.append(analyze_one_npz(name, path, key, knn_k=args.knn_k))

    print("\n=== SUMMARY (single D per npz) ===")
    cols = [
        "name", "delta_rel",
        "orc_mean", "orc_f_neg",
        "spectral_gap", "spectral_n_components",
        "ph_h1_count", "ph_h1_max_persistence",
    ]
    print(" | ".join(f"{c:>22s}" for c in cols))
    print("-" * (26 * len(cols)))
    for r in rows:
        print(" | ".join(f"{r[c]:22.6f}" if isinstance(r[c], (float, np.floating)) else f"{str(r[c]):>22s}" for c in cols))

    if args.out_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\n[Save] {args.out_csv}")


if __name__ == "__main__":
    main()