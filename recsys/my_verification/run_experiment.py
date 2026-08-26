"""
Verification run of the GradIsomapCF_movielens pipeline.

This is a COPY of the logic in recsys/GradIsomapCF_movielens/main_new2.py,
placed in a separate folder so the original project files are left untouched.
It imports the original, unmodified modules (NCF, GradientIsomapCF_log,
evaluation, prepare_data, new_datasets, manifold_visualization) from
recsys/GradIsomapCF_movielens and just adds the sys.path bookkeeping needed to
run it from a different working directory, plus writes all logs into this
folder (my_verification/logs_*) instead of the original logs_* directories.

No behavioral changes were made to the training/eval logic itself relative to
main_new2.py -- the point of this script is to faithfully reproduce the
original pipeline's behavior for verification purposes.
"""
import os
import sys
import time
import json
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
NSS_LAB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # .../NSS_lab
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")

# needed so that "from MANUL.Adam.Isomap import IsomapNN" inside
# GradientIsomapCF_log.py resolves (MANUL must be importable as a package,
# i.e. its parent dir must be on sys.path)
sys.path.insert(0, NSS_LAB_DIR)
# needed so that plain "import NCF", "import evaluation", etc. inside the
# original modules resolve, exactly as if main_new2.py were run from its own
# folder
sys.path.insert(0, GINCF_DIR)

from GradientIsomapCF_log import GradientIsomapCF
from evaluation import evaluate_topk_isomap, evaluate_topk_pure
from NCF import NCF
from prepare_data import (
    prepare_sequences,
    subsample_users_items,
    train_val_test_split_next_item,
    build_movie_user_matrix,
)
from new_datasets import NCFTrainDatasetFutureBlind, NCFTestDatasetSampled


def load_movielens_ratings(movielens_dir):
    """Loads ratings.dat from any MovieLens release using the ``::``-separated
    UserID::MovieID::Rating::Timestamp format - shared by ml-1m and ml-10m
    (verified against both; GroupLens kept this format through ml-10m,
    later releases switch to CSV, would need a different parser)."""
    ratings_path = os.path.join(movielens_dir, "ratings.dat")
    print(f"Loading ratings from {ratings_path} ...")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    print("Unique_users:", df["userId"].nunique(), "Unique_items:", df["movieId"].nunique())
    print(f"Loaded {len(df)} ratings")
    return df


def load_amazon_ratings(amazon_dir, category):
    """Loads a decompressed Amazon Reviews'23 5-core pure-ID CSV
    (userId/parent_asin/rating/timestamp, see
    https://amazon-reviews-2023.github.io/data_processing/5core.html) and
    renames columns to the same userId/movieId/rating/timestamp contract
    load_movielens_ratings() produces, so prepare_sequences()/
    subsample_users_items() work unmodified on either dataset. Timestamps
    here are already integer milliseconds (unlike MovieLens's seconds) -
    prepare_sequences only uses them for relative ordering (sort_values),
    so the unit difference doesn't matter downstream."""
    ratings_path = os.path.join(amazon_dir, f"{category}.csv")
    print(f"Loading ratings from {ratings_path} ...")
    df = pd.read_csv(ratings_path)
    df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
    df = df[["userId", "movieId", "rating", "timestamp"]]
    print("Unique_users:", df["userId"].nunique(), "Unique_items:", df["movieId"].nunique())
    print(f"Loaded {len(df)} ratings")
    return df


def main(
    max_users=300,
    max_movies=800,
    min_seq_len=2,
    num_ng=2,
    top_k=10,
    epochs_pure=100,
    gradisomap_epochs=10,
    cf_epochs=30,
    final_cf_epochs=30,
    lr_isomap=5e-2,
    run_ncf=True,
    run_gincf=True,
    n_run="verify01",
    seed=0,
    dataset_dir_name="ml-1m",
    dataset_type="movielens",
    amazon_category="Beauty_and_Personal_Care",
    select_by="loss",
    final_patience=3,
    inner_patience=5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)
    if dataset_type == "movielens":
        ratings_df = load_movielens_ratings(dataset_dir)
    elif dataset_type == "amazon":
        ratings_df = load_amazon_ratings(dataset_dir, amazon_category)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nBuilding per-user sequences...")
    df_mapped, user2seq = prepare_sequences(ratings_df)

    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped,
        max_users=max_users,
        max_movies=max_movies,
        min_seq_len=min_seq_len,
    )

    user_pos_set = {u: set(m for (m, ts, r) in seq) for u, seq in user2seq_sub.items()}

    print("\nTrain/Val/Test split (next-item)...")
    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)

    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))

    user_hist_val_set = user_pos_train_set

    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    train_pairs = [(int(u), int(m)) for (u, m, r) in train_events]

    train_dataset = NCFTrainDatasetFutureBlind(
        features_pos=train_pairs,
        num_items=num_movies,
        user_pos_train_set=user_pos_train_set,
        num_ng=num_ng,
        seed=42,
    )

    val_dataset = NCFTestDatasetSampled(
        next_triples=val_next,
        num_items=num_movies,
        user_pos_all_set=user_hist_val_set,
        num_ng=99,
        seed=123,
    )

    test_dataset = NCFTestDatasetSampled(
        next_triples=test_next,
        num_items=num_movies,
        user_pos_all_set=user_hist_test_set,
        num_ng=99,
        seed=456,
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    print(f"\n num_users={num_users}, num_movies={num_movies}, "
          f"train_pairs={len(train_pairs)}, val_users={len(val_next)}, test_users={len(test_next)}")

    results = {}

    if run_ncf:
        pure_model = NCF(
            user_num=num_users,
            item_num=num_movies,
            factor_num=8,
            num_layers=4,
            dropout=0.0,
            model="NeuMF-end",
        ).to(device)

        pos_weight = torch.tensor([num_ng], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(pure_model.parameters(), lr=0.001, weight_decay=1e-6)

        best_hr_val, best_ndcg_val, best_epoch_pure = 0.0, 0.0, 0
        best_state_pure = None
        best_val_loss = 10

        pure_history = {"epoch": [], "train_loss": [], "val_loss": [], "hr_val": [], "ndcg_val": []}

        start_total_time = time.time()

        for epoch in range(epochs_pure):
            epoch_start = time.time()

            pure_model.train()
            train_dataset.ng_sample()
            total_train_loss, n_train_batches = 0.0, 0

            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.to(device)
                preds = pure_model(user, item)
                loss = loss_fn(preds, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)

            pure_model.eval()
            total_val_loss, n_val_batches = 0.0, 0
            with torch.no_grad():
                for user, item, label in val_loader:
                    user, item, label = user.to(device), item.to(device), label.to(device)
                    preds = pure_model(user, item)
                    loss_val = loss_fn(preds, label)
                    total_val_loss += loss_val.item()
                    n_val_batches += 1

            avg_val_loss = total_val_loss / max(1, n_val_batches)
            hr_val, ndcg_val = evaluate_topk_pure(pure_model, val_loader, top_k, device)
            epoch_time = time.time() - epoch_start

            print(f"[PURE NCF] epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                  f"HR@{top_k}_val={hr_val:.4f}, NDCG@{top_k}_val={ndcg_val:.4f}, time={epoch_time:.2f}")

            pure_history["epoch"].append(epoch)
            pure_history["train_loss"].append(avg_train_loss)
            pure_history["val_loss"].append(avg_val_loss)
            pure_history["hr_val"].append(hr_val)
            pure_history["ndcg_val"].append(ndcg_val)

            if avg_val_loss < best_val_loss:
                best_val_loss, best_epoch_pure = avg_val_loss, epoch
                best_hr_val, best_ndcg_val = hr_val, ndcg_val
                best_state_pure = copy.deepcopy(pure_model.state_dict())

            if avg_val_loss > avg_train_loss:
                print(f"[PURE NCF] stopping: val_loss({avg_val_loss:.4f}) > train_loss({avg_train_loss:.4f})")
                break

        print(f"PURE NCF best epoch {best_epoch_pure}: HR@{top_k}_val={best_hr_val:.4f}, NDCG@{top_k}_val={best_ndcg_val:.4f}")

        if best_state_pure is not None:
            pure_model.load_state_dict(best_state_pure)

        hr_test_pure, ndcg_test_pure = evaluate_topk_pure(pure_model, test_loader, top_k, device)
        print(f"PURE NCF final on TEST: HR@{top_k}={hr_test_pure:.4f}, NDCG@{top_k}={ndcg_test_pure:.4f}")

        total_time = time.time() - start_total_time
        print(f"PURE NCF total_time={time.strftime('%H:%M:%S', time.gmtime(total_time))}")

        logs_dir = os.path.join(HERE, "logs_pure_ncf")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, f"pure_ncf_{n_run}_history.json"), "w") as f:
            json.dump(pure_history, f, indent=2)

        results["pure_ncf"] = {
            "best_epoch": best_epoch_pure,
            "best_val_hr": best_hr_val,
            "best_val_ndcg": best_ndcg_val,
            "test_hr": hr_test_pure,
            "test_ndcg": ndcg_test_pure,
            "n_epochs_ran": len(pure_history["epoch"]),
        }

    if run_gincf:
        print("\n=== Training GradientIsomapCF (IsomapNN + NeuMFOnManifold) ===")

        movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True)
        features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

        logs_folder = os.path.join(HERE, "logs_movielens_isomap_cf", str(n_run))

        gi_cf = GradientIsomapCF(
            train_feature=features_t,
            train_events=train_events,
            num_users=num_users,
            num_items=num_movies,
            user_pos_set=user_pos_set,
            ng_seed=42,
            latent_len=64,
            n_neighbors=10,
            epochs=gradisomap_epochs,
            cf_epochs=cf_epochs,
            final_cf_epochs=final_cf_epochs,
            batch_size=2048,
            lr_isomap=lr_isomap,
            lr_ncf=1e-3,
            factor_num=16,
            num_layers=3,
            dropout=0.0,
            model_type="NeuMF-end",
            logs_folder=logs_folder,
            device=str(device),
            stop_criteria_value=0.001,
            num_ng=num_ng,
            select_by=select_by,
            final_patience=final_patience,
            inner_patience=inner_patience,
        )

        isomap_model, ncf_manifold_model = gi_cf.train(val_loader=val_loader, top_k=top_k, device=device)

        hr_iso, ndcg_iso = evaluate_topk_isomap(ncf_manifold_model, isomap_model, test_loader, top_k, device)
        print(f"GradientIsomapCF final on TEST: HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")

        with open(os.path.join(logs_folder, "history.json"), "w", encoding="utf-8") as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        results["gincf"] = {
            "test_hr": hr_iso,
            "test_ndcg": ndcg_iso,
            "outer_history": gi_cf.history,
        }

    results_path = os.path.join(HERE, f"results_{n_run}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Save] Combined results -> {results_path}")

    return results


if __name__ == "__main__":
    main(
        max_users=300,
        max_movies=800,
        min_seq_len=2,
        num_ng=2,
        top_k=10,
        epochs_pure=100,
        gradisomap_epochs=10,
        cf_epochs=30,
        final_cf_epochs=30,
        run_ncf=True,
        run_gincf=True,
        n_run="verify01",
        seed=0,
    )
