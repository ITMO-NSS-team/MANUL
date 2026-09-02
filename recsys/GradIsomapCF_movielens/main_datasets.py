# main.py
import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import pandas as pd
import json
import time
import copy
import torch.nn.functional as F

from torch.utils.data import DataLoader

from GradientIsomapCF_log import GradientIsomapCF
from evaluation import evaluate_topk_isomap, evaluate_topk_pure
from NCF import NCF
from prepare_data import (
    prepare_sequences,
    prepare_sequences_amazon,
    subsample_users_items,
    train_val_test_split_next_item,
    build_movie_user_matrix,
    load_amazon_books,
)
from manifold_visualization import (
    plot_gi_losses, plot_pure_ncf_losses,
    plot_pure_ncf_metrics, plot_gi_convergence,
)
from new_datasets import NCFTrainDatasetFutureBlind, NCFTestDatasetSampled


# ─────────────────────────────────────────────
#  Загрузчики датасетов
# ─────────────────────────────────────────────

def load_movielens_1m_ratings(ml1m_dir: str) -> pd.DataFrame:
    ratings_path = os.path.join(ml1m_dir, "ratings.dat")
    print(f"Загружаем рейтинги из {ratings_path} ...")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    print(f"  Уникальных пользователей: {df['userId'].nunique():,}")
    print(f"  Уникальных фильмов:       {df['movieId'].nunique():,}")
    print(f"  Всего рейтингов:          {len(df):,}")
    return df


# ─────────────────────────────────────────────
#  Роутер: загрузка + prepare_sequences
#  по имени датасета
# ─────────────────────────────────────────────

DATASET_MOVIELENS = "movielens"
DATASET_AMAZON    = "amazon_books"


def load_and_prepare(dataset_name: str, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Единая точка входа для загрузки и первичной подготовки данных.

    Возвращает:
        df_mapped  — DataFrame с колонками user_idx, movie_idx, rating, timestamp
        user2seq   — {user_idx: [(item_idx, timestamp, rating), ...]}
    """
    if dataset_name == DATASET_MOVIELENS:
        raw_df    = load_movielens_1m_ratings(config["ml1m_dir"])
        df_mapped, user2seq = prepare_sequences(raw_df)

    elif dataset_name == DATASET_AMAZON:
        raw_df = load_amazon_books(
            #hf_reviews_name   = config.get("hf_name", "cogsci13/Amazon-Reviews-2023-Books-Review"),
            #hf_reviews_config = config.get("hf_config", "raw_review_Books"),
            #cache_path        = config.get("cache_path", "data/amazon_books_reviews.parquet"),
        )
        df_mapped, user2seq, _, _ = prepare_sequences_amazon(raw_df)

    else:
        raise ValueError(
            f"Неизвестный датасет: '{dataset_name}'. "
            f"Допустимые значения: '{DATASET_MOVIELENS}', '{DATASET_AMAZON}'"
        )

    return df_mapped, user2seq


# ─────────────────────────────────────────────
#  FocalLoss (без изменений)
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p_t           = (targets * torch.sigmoid(logits)
                         + (1 - targets) * (1 - torch.sigmoid(logits)))
        focal_weight  = (1 - p_t) ** self.gamma
        alpha_t       = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * ce_loss).mean()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main(
    dataset_name: str  = DATASET_MOVIELENS,   # <── ПЕРЕКЛЮЧАТЕЛЬ
    dataset_config: dict | None = None,        # специфичные пути / параметры
    max_users:   int   = 300,
    max_movies:  int   = 800,
    min_seq_len: int   = 5,
    num_ng:      int   = 3,
    top_k:       int   = 10,
    epochs_pure: int   = 10,
    gradisomap_epochs: int = 5,
    run_ncf:    bool   = True,
    run_gincf:  bool   = True,
    n_run:      int    = 68,
):
    # ── дефолтные пути по датасету ──────────────────────────────────
    if dataset_config is None:
        if dataset_name == DATASET_MOVIELENS:
            dataset_config = {"ml1m_dir": r"data\ml-1m"}
        elif dataset_name == DATASET_AMAZON:
            dataset_config = {
                "hf_name":    "cogsci13/Amazon-Reviews-2023-Books-Review",
                "hf_config":  "raw_review_Books",
                "cache_path": "data/amazon_books_reviews.parquet",
            }

    # ── логи-папки с именем датасета ────────────────────────────────
    logs_pure_dir   = f"logs_pure_ncf_{dataset_name}"
    logs_gi_dir     = f"logs_{dataset_name}_isomap_cf"

    # ── загрузка и подготовка ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Датасет: {dataset_name}")
    print(f"{'='*55}\n")

    df_mapped, user2seq = load_and_prepare(dataset_name, dataset_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nУстройство: {device}")

    # ── подвыборка ───────────────────────────────────────────────────
    print("\nПодвыборка пользователей и айтемов...")
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped,
        max_users   = max_users,
        max_movies  = max_movies,
        min_seq_len = min_seq_len,
    )

    user_pos_set = {
        u: set(m for (m, ts, r) in seq)
        for u, seq in user2seq_sub.items()
    }

    # ── сплит ────────────────────────────────────────────────────────
    print("\nTrain/Val/Test split (next-item)...")
    train_events, val_next, test_next = train_val_test_split_next_item(
        user2seq_sub, min_len=3
    )

    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))

    user_hist_val_set  = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    train_pairs = [(int(u), int(m)) for (u, m, r) in train_events]

    # ── датасеты и лоадеры ───────────────────────────────────────────
    train_dataset = NCFTrainDatasetFutureBlind(
        features_pos       = train_pairs,
        num_items          = num_movies,
        user_pos_train_set = user_pos_train_set,
        num_ng             = num_ng,
        seed               = 42,
    )
    val_dataset = NCFTestDatasetSampled(
        next_triples     = val_next,
        num_items        = num_movies,
        user_pos_all_set = user_hist_val_set,
        num_ng           = 99,
        seed             = 123,
    )
    test_dataset = NCFTestDatasetSampled(
        next_triples     = test_next,
        num_items        = num_movies,
        user_pos_all_set = user_hist_test_set,
        num_ng           = 99,
        seed             = 456,
    )

    train_loader = DataLoader(train_dataset, batch_size=256,  shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=100,  shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=100,  shuffle=False)

    # ════════════════════════════════════════════════════════════════
    #  Pure NCF
    # ════════════════════════════════════════════════════════════════
    if run_ncf:
        pure_model = NCF(
            user_num   = num_users,
            item_num   = num_movies,
            factor_num = 8,
            num_layers = 4,
            dropout    = 0.0,
            model      = "NeuMF-end",
        ).to(device)
        print("factor_num=8, num_layers=4")

        pos_weight = torch.tensor([num_ng], device=device, dtype=torch.float32)
        loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = optim.Adam(pure_model.parameters(), lr=0.001, weight_decay=1e-6)

        best_hr_val, best_ndcg_val, best_epoch_pure = 0.0, 0.0, 0
        best_state_pure  = None
        best_val_loss    = 10.0

        pure_history = {k: [] for k in
                        ["epoch", "train_loss", "val_loss", "hr_val", "ndcg_val"]}

        start_total_time = time.time()

        for epoch in range(epochs_pure):
            start_epoch_time = time.time()
            pure_model.train()
            train_dataset.ng_sample()
            total_train_loss, n_train_batches = 0.0, 0

            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.to(device)
                preds = pure_model(user, item)
                loss  = loss_fn(preds, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                n_train_batches  += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)

            pure_model.eval()
            total_val_loss, n_val_batches = 0.0, 0
            with torch.no_grad():
                for user, item, label in val_loader:
                    user, item, label = user.to(device), item.to(device), label.to(device)
                    loss_v             = loss_fn(pure_model(user, item), label)
                    total_val_loss    += loss_v.item()
                    n_val_batches     += 1

            avg_val_loss       = total_val_loss / max(1, n_val_batches)
            hr_val, ndcg_val   = evaluate_topk_pure(pure_model, val_loader, top_k, device)
            epoch_time         = time.time() - start_epoch_time

            print(f"[PURE NCF | {dataset_name}] epoch {epoch}: "
                  f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                  f"HR@{top_k}_val={hr_val:.4f}, NDCG@{top_k}_val={ndcg_val:.4f}, "
                  f"time={epoch_time:.2f}s")

            for k, v in zip(pure_history,
                            [epoch, avg_train_loss, avg_val_loss, hr_val, ndcg_val]):
                pure_history[k].append(v)

            if avg_val_loss < best_val_loss:
                best_val_loss, best_epoch_pure = avg_val_loss, epoch
                best_hr_val, best_ndcg_val     = hr_val, ndcg_val
                best_state_pure                = copy.deepcopy(pure_model.state_dict())

            if avg_val_loss > avg_train_loss:
                break

        print(f"PURE NCF best epoch {best_epoch_pure}: "
              f"HR@{top_k}_val={best_hr_val:.4f}, NDCG@{top_k}_val={best_ndcg_val:.4f}")

        if best_state_pure is not None:
            pure_model.load_state_dict(best_state_pure)

        hr_test_pure, ndcg_test_pure = evaluate_topk_pure(
            pure_model, test_loader, top_k, device
        )
        print(f"PURE NCF final on TEST: "
              f"HR@{top_k}={hr_test_pure:.4f}, NDCG@{top_k}={ndcg_test_pure:.4f}")

        total_time = time.time() - start_total_time
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

        images_dir = os.path.join(logs_pure_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        plot_pure_ncf_losses(pure_history, images_dir,
                             run_name=f"pure_ncf_{dataset_name}_{n_run}_losses")
        plot_pure_ncf_metrics(pure_history, images_dir,
                              run_name=f"pure_ncf_{dataset_name}_{n_run}_metrics")

    # ════════════════════════════════════════════════════════════════
    #  GradientIsomapCF
    # ════════════════════════════════════════════════════════════════
    if run_gincf:
        print(f"\n=== GradientIsomapCF | {dataset_name} ===")

        movie_user_mat = build_movie_user_matrix(
            train_events, num_movies, num_users, use_ratings=True
        )
        features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

        gi_cf = GradientIsomapCF(
            train_feature    = features_t,
            train_events     = train_events,
            num_users        = num_users,
            num_items        = num_movies,
            user_pos_set     = user_pos_set,
            ng_seed          = 42,
            latent_len       = 128,
            n_neighbors      = 10,
            epochs           = gradisomap_epochs,
            cf_epochs        = 100,
            final_cf_epochs  = 100,
            batch_size       = 2048,
            lr_isomap        = 3e-2,
            lr_ncf           = 1e-3, #5e-4,
            factor_num       = 32,
            num_layers       = 3,
            dropout          = 0.0,
            model_type       = "NeuMF-end",
            logs_folder      = f"{logs_gi_dir}/{n_run}",
            device           = str(device),
            stop_criteria_value = 0.001,
            num_ng           = num_ng,)

        isomap_model, ncf_manifold_model = gi_cf.train(
            val_loader = val_loader,
            top_k      = top_k,
            device     = device,
        )

        hr_iso, ndcg_iso = evaluate_topk_isomap(
            ncf_manifold_model, isomap_model, test_loader, top_k, device
        )
        print(f"GradientIsomapCF final TEST: "
              f"HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")

        images_dir = os.path.join(logs_gi_dir, f"{n_run}/images")
        plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")

        with open(f"{logs_gi_dir}/{n_run}/images/history.json", "w",
                  encoding="utf-8") as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        plot_gi_convergence(
            gi_cf.history, top_k=top_k,
            save_path=os.path.join(logs_gi_dir, f"{n_run}/images/metrics.png"),
        )


# ─────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── MovieLens ───────────────────────────────────────
    # main(
    #     dataset_name   = DATASET_MOVIELENS,
    #     dataset_config = {"ml1m_dir": r"data\ml-1m"},
    #     max_users      = 300,
    #     max_movies     = 800,
    #     min_seq_len    = 2,
    #     num_ng         = 2,
    #     top_k          = 10,
    #     epochs_pure    = 100,
    #     gradisomap_epochs = 30,
    #     run_ncf        = True,
    #     run_gincf      = False,
    #     n_run          = 68,
    # )

    # ── Amazon Books ─────────────────────────────────────────────────
    main(
        dataset_name   = DATASET_AMAZON,
        dataset_config = {
            "hf_name":    "cogsci13/Amazon-Reviews-2023-Books-Review",
            "hf_config":  "raw_review_Books",
            "cache_path": "data/amazon_books_reviews.parquet",
        },
        max_users      = 2000,
        max_movies     = 5000,
        min_seq_len    = 1,
        num_ng         = 2,
        top_k          = 10,
        epochs_pure    = 100,
        gradisomap_epochs = 30,
        run_ncf        = False,
        run_gincf      = True,
        n_run          = 717,
    )