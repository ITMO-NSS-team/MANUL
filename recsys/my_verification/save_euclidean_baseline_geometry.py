"""
Retrains the Euclidean NeuMF baseline (same harness as
ablation_geometry_vs_optimization.train_and_eval_euclidean_baseline) and
persists its learned item-embedding geometry (concatenated GMF+MLP item
embedding tables) to an .npz, so full_hyperbolicity_table.py can fold this
arm into the unified hyperbolicity+loss table on the same footing as the
manifold-based arms (pure_init, GINCF etas, Poincare).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ablation_geometry_vs_optimization import build_data, train_and_eval_euclidean_baseline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    tmp_logs_folder = os.path.join(HERE, "euclidean_baseline_tmp_logs")
    data = build_data(300, 800, min_seq_len=2, num_ng=2, dataset_dir_name="ml-1m",
                      device=device, tmp_logs_folder=tmp_logs_folder)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    hr, ndcg, val_loss, item_emb = train_and_eval_euclidean_baseline(
        data, device, return_item_embeddings=True)
    print(f"Euclidean NeuMF baseline: test HR@10={hr:.4f} NDCG@10={ndcg:.4f} "
          f"best_val_loss={val_loss:.4f}", flush=True)

    item_emb_np = item_emb.numpy()
    D = np.linalg.norm(item_emb_np[:, None, :] - item_emb_np[None, :, :], axis=-1)

    out_path = os.path.join(HERE, "euclidean_baseline_geometry.npz")
    np.savez(out_path, D=D, val_loss=val_loss, hr=hr, ndcg=ndcg)
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
