"""
SASRec-версия save_euclidean_baseline_geometry.py.

Обучает чистый SASRec (source.py — обучаемый item_emb, без манифолда)
и сохраняет геометрию обученных item embeddings (.npz), чтобы
full_hyperbolicity_table.py мог включить этот arm в единую таблицу
гиперболичности+метрик наравне с GradientIsomapSASRec и Poincaré.

Геометрия = попарные евклидовы расстояния между весами item_emb.weight —
те координаты айтемов в скрытом пространстве, которые SASRec нашёл сам
через gradient descent, без каких-либо геометрических предположений.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ablation_geometry_vs_optimization_sasrec import (
    build_data_sasrec,
    train_and_eval_sasrec_euclidean_baseline,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="ml-1m")
    parser.add_argument("--dataset_type", default="movielens",
                        choices=["movielens", "amazon"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--tag", default="",
                        help="Суффикс для имени файла (напр. 'beauty'), чтобы "
                             "запуски на разных датасетах не перезаписывали друг друга.")
    parser.add_argument("--select_by", default="hr", choices=["loss", "hr"])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    sasrec_config = {
        'hidden_units': args.hidden_units,
        'maxlen': args.maxlen,
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate,
        'l2_emb': 0.0,
    }

    suffix = f"_{args.tag}" if args.tag else ""
    tmp_logs_folder = os.path.join(HERE, f"sasrec_euclidean_tmp_logs{suffix}")

    data = build_data_sasrec(
        max_users=args.max_users, max_movies=args.max_movies,
        min_seq_len=2, num_ng=2,
        dataset_dir_name=args.dataset_dir_name, device=device,
        tmp_logs_folder=tmp_logs_folder,
        dataset_type=args.dataset_type,
        amazon_category=args.amazon_category,
        maxlen=args.maxlen,
    )
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    hr, ndcg, val_loss, item_emb = train_and_eval_sasrec_euclidean_baseline(
        data=data, device=device, sasrec_config=sasrec_config,
        lr=1e-3, epochs=args.epochs, patience=args.patience,
        seed=args.seed, select_by=args.select_by,
        batch_size=args.batch_size,
        return_item_embeddings=True,
        verbose=True,
    )

    print(f"Euclidean SASRec baseline: "
          f"test HR@10={hr:.4f} NDCG@10={ndcg:.4f} "
          f"best_val_loss={val_loss:.4f}", flush=True)

    # Попарные евклидовы расстояния между item embeddings
    item_emb_np = item_emb.numpy()   # [num_movies, hidden_units]
    D = np.linalg.norm(
        item_emb_np[:, None, :] - item_emb_np[None, :, :], axis=-1
    )  # [num_movies, num_movies]

    out_path = os.path.join(HERE, f"sasrec_euclidean_baseline_geometry{suffix}.npz")
    np.savez(out_path, D=D, val_loss=val_loss, hr=hr, ndcg=ndcg)
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
    