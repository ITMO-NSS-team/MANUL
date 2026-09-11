"""
Retrains the Euclidean NeuMF baseline (same harness as
ablation_geometry_vs_optimization.train_and_eval_euclidean_baseline) and
persists its learned item-embedding geometry (concatenated GMF+MLP item
embedding tables) to an .npz, so full_hyperbolicity_table.py can fold this
arm into the unified hyperbolicity+loss table on the same footing as the
manifold-based arms (pure_init, GINCF etas, Poincare).
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ablation_geometry_vs_optimization import build_data, train_and_eval_euclidean_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="ml-1m")
    parser.add_argument("--dataset_type", default="movielens", choices=["movielens", "amazon"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--tag", default="",
                        help="Optional suffix for the saved geometry filename "
                             "(e.g. 'ml10m'), so runs at different scales don't "
                             "overwrite each other's euclidean_baseline_geometry*.npz.")
    parser.add_argument("--select_by", default="loss", choices=["loss", "hr"])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    tmp_logs_folder = os.path.join(HERE, f"euclidean_baseline_tmp_logs{suffix}")
    data = build_data(args.max_users, args.max_movies, min_seq_len=2, num_ng=2,
                      dataset_dir_name=args.dataset_dir_name, device=device,
                      tmp_logs_folder=tmp_logs_folder,
                      dataset_type=args.dataset_type, amazon_category=args.amazon_category)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    hr, ndcg, val_loss, item_emb = train_and_eval_euclidean_baseline(
        data, device, return_item_embeddings=True, select_by=args.select_by,
        patience=args.patience, epochs=args.epochs, dropout=args.dropout,
        weight_decay=args.weight_decay, seed=args.seed)
    print(f"Euclidean NeuMF baseline: test HR@10={hr:.4f} NDCG@10={ndcg:.4f} "
          f"best_val_loss={val_loss:.4f}", flush=True)

    item_emb_np = item_emb.numpy()
    D = np.linalg.norm(item_emb_np[:, None, :] - item_emb_np[None, :, :], axis=-1)

    out_path = os.path.join(HERE, f"euclidean_baseline_geometry{suffix}.npz")
    np.savez(out_path, D=D, val_loss=val_loss, hr=hr, ndcg=ndcg)
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
