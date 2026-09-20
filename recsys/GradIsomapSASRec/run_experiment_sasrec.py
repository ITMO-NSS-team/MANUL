"""
SASRec-вариант run_experiment.py
Поддерживает amazon_beauty через dataset_type="amazon".
"""
import os
import sys
import json
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
NSS_LAB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")

sys.path.insert(0, NSS_LAB_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from GradientIsomapSASRec_best2 import GradientIsomapSASRec
from sasrec_manifold_sampler import SASRecManifoldTestDataset
from evaluation_sasrec_manifold import evaluate_topk_sasrec_isomap
from recsys.GradIsomapCF_movielens.prepare_data import (
    prepare_sequences, subsample_users_items,
    train_val_test_split_next_item, build_movie_user_matrix,
)


def load_movielens_ratings(movielens_dir):
    ratings_path = os.path.join(movielens_dir, "ratings.dat")
    df = pd.read_csv(
        ratings_path, sep="::", engine="python", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    return df


def load_amazon_ratings(amazon_dir, category):
    """5-core Amazon Reviews'23 CSV."""
    ratings_path = os.path.join(amazon_dir, f"{category}.csv")
    df = pd.read_csv(ratings_path)
    df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
    df = df[["userId", "movieId", "rating", "timestamp"]]
    return df


def _build_user_seq_train(train_events):
    """Строим {user: [item1, item2, ...]} из train_events."""
    from collections import defaultdict
    user_train = defaultdict(list)
    for (u, m, r) in train_events:
        user_train[int(u)].append(int(m))
    return dict(user_train)


def main(
    max_users=300,
    max_movies=800,
    min_seq_len=5,
    top_k=10,
    gradisomap_epochs=30,
    cf_epochs=30,
    final_cf_epochs=50,
    lr_isomap=5e-3,
    n_run="sasrec_verify01",
    seed=0,
    dataset_dir_name="ml-1m",
    dataset_type="movielens",
    amazon_category="Beauty_and_Personal_Care",
    select_by="hr",
    final_patience=5,
    inner_patience=5,
    warm_start_inner=False,
    freeze_item_projection=False,
    maxlen=50,
    hidden_units=64,
    num_blocks=2,
    num_heads=1,
    sasrec_dropout=0.2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)
    if dataset_type == "movielens":
        ratings_df = load_movielens_ratings(dataset_dir)
    elif dataset_type == "amazon":
        ratings_df = load_amazon_ratings(dataset_dir, amazon_category)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    df_mapped, user2seq = prepare_sequences(ratings_df)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies,
        min_seq_len=min_seq_len,
    )

    print("\nTrain/Val/Test split...")
    train_events, val_next, test_next = train_val_test_split_next_item(
        user2seq_sub, min_len=3,
    )

    # TRAIN ONLY sets (no leakage)
    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))

    user_train = _build_user_seq_train(train_events)

    user_hist_val_set = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    val_dataset = SASRecManifoldTestDataset(
        user_train=user_train, next_triples=val_next,
        num_items=num_movies, user_pos_all_set=user_hist_val_set,
        maxlen=maxlen, num_ng=99, seed=123,
    )
    test_dataset = SASRecManifoldTestDataset(
        user_train=user_train, next_triples=test_next,
        num_items=num_movies, user_pos_all_set=user_hist_test_set,
        maxlen=maxlen, num_ng=99, seed=456,
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"num_users={num_users}, num_items={num_movies}, "
          f"val={len(val_next)}, test={len(test_next)}")

    # Feature matrix for D_input
    movie_user_mat = build_movie_user_matrix(
        train_events, num_movies, num_users, use_ratings=True,
    )
    features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

    logs_folder = os.path.join(HERE, "logs_sasrec_isomap", str(n_run))

    sasrec_config = {
        'hidden_units': hidden_units, 'maxlen': maxlen,
        'num_blocks': num_blocks, 'num_heads': num_heads,
        'dropout_rate': sasrec_dropout, 'l2_emb': 0.0,
    }

    gi_sasrec = GradientIsomapSASRec(
        train_feature=features_t,
        user_train=user_train,
        num_users=num_users,
        num_items=num_movies,
        user_pos_set=user_pos_train_set,  # TRAIN ONLY — no leakage
        latent_len=64,
        n_neighbors=10,
        epochs=gradisomap_epochs,
        lr_isomap=lr_isomap,
        sasrec_config=sasrec_config,
        cf_epochs=cf_epochs,
        final_cf_epochs=final_cf_epochs,
        lr_sasrec=1e-3,
        batch_size=2048,  # 256,
        #n_batches_per_outer_step=4,
        select_by=select_by,
        inner_patience=inner_patience,
        final_patience=final_patience,
        warm_start_inner=warm_start_inner,
        freeze_item_projection=freeze_item_projection,
        logs_folder=logs_folder,
        device=str(device),
        #seed=seed,
    )

    isomap_model, sasrec_model = gi_sasrec.train(
        val_loader=val_loader, top_k=top_k, device=device,
    )

    hr_test, ndcg_test = evaluate_topk_sasrec_isomap(
        sasrec_model, isomap_model, test_loader, top_k, device,
    )
    print(f"\nGradientIsomapSASRec final TEST: "
          f"HR@{top_k}={hr_test:.4f} NDCG@{top_k}={ndcg_test:.4f}")

    with open(os.path.join(logs_folder, "history.json"), "w", encoding="utf-8") as f:
        json.dump(gi_sasrec.history, f, ensure_ascii=False, indent=4)

    results = {
        "gi_sasrec": {
            "test_hr": hr_test, "test_ndcg": ndcg_test,
            "outer_history": gi_sasrec.history,
        }
    }
    results_path = os.path.join(HERE, f"results_{n_run}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Save] Results → {results_path}")

    return results


if __name__ == "__main__":
    # Amazon Beauty run
    main(
        max_users=300, 
        max_movies=800, 
        min_seq_len=5,
        top_k=10, 
        gradisomap_epochs=30,
        cf_epochs=30, 
        final_cf_epochs=50,
        lr_isomap=5e-3, 
        n_run="sasrec_amazon_beauty_06",
        seed=0,
        dataset_dir_name="amazon_beauty",
        dataset_type="amazon",
        amazon_category="Beauty_and_Personal_Care",
        select_by="hr",              
        final_patience=5, 
        inner_patience=5,
        warm_start_inner=True,
        freeze_item_projection=False,  
        maxlen=50, 
        hidden_units=64,
        num_blocks=2, 
        num_heads=1, 
        sasrec_dropout=0.2,
    )
