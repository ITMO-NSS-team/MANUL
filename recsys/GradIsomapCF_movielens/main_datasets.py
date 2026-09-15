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

#from GradientIsomapCF_log import GradientIsomapCF
from GradientIsomapCF_log_best_final import GradientIsomapCF
from evaluation import evaluate_topk_isomap, evaluate_topk_pure
from NCF import NCF
from prepare_data import (
    prepare_sequences,
    prepare_sequences_amazon,
    subsample_users_items,
    train_val_test_split_next_item,
    build_movie_user_matrix,
    load_amazon_books,
    load_tecd_marketplace
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


# В evaluate_topk — разделите на repeat / new

def evaluate_topk_split(model, test_loader, top_k, device,
                        user_train_items: dict,
                        isomap_model=None):
    """
    Считает HR@K и NDCG@K отдельно для:
      - repeat items (тест-айтем уже был в train у пользователя)
      - new items    (тест-айтем НЕ был в train у пользователя)
    
    Поддерживает как обычный NCF, так и NeuMFOnManifold (с передачей isomap_model).
    """
    hr_repeat, hr_new = [], []
    ndcg_repeat, ndcg_new = [], []

    model.eval()
    if isomap_model is not None:
        isomap_model.eval()
        with torch.no_grad():
            item_Z = isomap_model().to(torch.float32)
    else:
        item_Z = None

    with torch.no_grad():
        for users, items, labels in test_loader:
            users = users.to(device)
            items = items.to(device)

            # Получаем предсказания модели
            if item_Z is not None:
                predictions = model(users, items, item_Z)
            else:
                predictions = model(users, items)

            predictions = predictions.view(-1)

            # В NCFTestDatasetSampled на каждого юзера приходится ровно 100 примеров (1 pos + 99 negs)
            group_size = 100
            for start_idx in range(0, len(predictions), group_size):
                u_group = users[start_idx:start_idx + group_size]
                i_group = items[start_idx:start_idx + group_size]
                l_group = labels[start_idx:start_idx + group_size]
                p_group = predictions[start_idx:start_idx + group_size]

                u_int = u_group[0].item()

                # Находим индекс позитивного (целевого) элемента (где label == 1.0)
                pos_indices = (l_group == 1.0).nonzero(as_tuple=True)[0]
                if len(pos_indices) == 0:
                    continue
                
                target_idx = pos_indices[0].item()
                target_item = i_group[target_idx].item()
                target_score = p_group[target_idx].item()

                # Считаем ранг целевого элемента (сколько негативов получили скор выше)
                rank = (p_group > target_score).sum().item()

                if rank < top_k:
                    hit = 1.0
                    ndcg = 1.0 / np.log2(rank + 2)
                else:
                    hit = 0.0
                    ndcg = 0.0

                # Проверяем, был ли целевой элемент в обучающей выборке данного пользователя
                is_repeat = target_item in user_train_items.get(u_int, set())

                if is_repeat:
                    hr_repeat.append(hit)
                    ndcg_repeat.append(ndcg)
                else:
                    hr_new.append(hit)
                    ndcg_new.append(ndcg)

    return {
        "HR@K_all":      np.mean(hr_repeat + hr_new) if (hr_repeat or hr_new) else 0.0,
        "NDCG@K_all":    np.mean(ndcg_repeat + ndcg_new) if (ndcg_repeat or ndcg_new) else 0.0,
        "HR@K_repeat":   np.mean(hr_repeat) if hr_repeat else 0.0,
        "NDCG@K_repeat": np.mean(ndcg_repeat) if ndcg_repeat else 0.0,
        "HR@K_new":      np.mean(hr_new) if hr_new else 0.0,
        "NDCG@K_new":    np.mean(ndcg_new) if ndcg_new else 0.0,
        "n_repeat":      len(hr_repeat),
        "n_new":         len(hr_new),
    }

# ─────────────────────────────────────────────
#  Роутер: загрузка + prepare_sequences
#  по имени датасета
# ─────────────────────────────────────────────

DATASET_MOVIELENS = "movielens"
DATASET_AMAZON    = "amazon_books"
DATASET_TECD = "tecd_marketplace"


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

    elif dataset_name == DATASET_TECD:
        raw_df = load_tecd_marketplace(
            subset_path = config.get(
                "subset_path",
                "data/tecd/tecd_marketplace_subset.parquet"),
            use_positive_only = config.get("use_positive_only", True),
        )
        df_mapped, user2seq, _, _ = prepare_sequences_amazon(raw_df)
        # prepare_sequences_amazon подходит — там тоже строковые ID

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
            factor_num = 32,
            num_layers = 4,
            dropout    = 0.0,
            model      = "NeuMF-end",
        ).to(device)
        print("factor_num=32, num_layers=4")

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
            user_pos_set     = user_pos_train_set,
            ng_seed          = 42,
            latent_len       = 128,
            n_neighbors      = 20,
            epochs           = gradisomap_epochs,
            cf_epochs        = 100,
            final_cf_epochs  = 100,
            batch_size       = 2048,
            lr_isomap        = 3e-2,
            lr_ncf           = 1e-3,
            factor_num       = 32,
            num_layers       = 3,
            dropout          = 0.0,
            model_type       = "NeuMF-end",
            logs_folder      = f"{logs_gi_dir}/{n_run}",
            device           = str(device),
            stop_criteria_value = 0.001,
            num_ng           = num_ng,)

                # ── обучение: получаем оба чекпоинта сразу ──
        (isomap_best, ncf_best), (isomap_last, ncf_last) = gi_cf.train(
            val_loader = val_loader,
            top_k      = top_k,
            device     = device,
        )

        # ── Test metrics: BEST ──
        hr_best, ndcg_best = evaluate_topk_isomap(
            ncf_best, isomap_best, test_loader, top_k, device
        )
        print(f"[FINAL/BEST] TEST: HR@{top_k}={hr_best:.4f}, NDCG@{top_k}={ndcg_best:.4f}")

        # ── Test metrics: LAST ──
        hr_last, ndcg_last = evaluate_topk_isomap(
            ncf_last, isomap_last, test_loader, top_k, device
        )
        print(f"[FINAL/LAST] TEST: HR@{top_k}={hr_last:.4f}, NDCG@{top_k}={ndcg_last:.4f}")

        # ── Графики (не зависят от return value — используют gi_cf.history) ──
        images_dir = os.path.join(logs_gi_dir, f"{n_run}/images")
        os.makedirs(images_dir, exist_ok=True)

        plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")

        with open(os.path.join(images_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        plot_gi_convergence(
            gi_cf.history, top_k=top_k,
            save_path=os.path.join(images_dir, "metrics.png"),
        )

        # ── Repeat / New: BEST ──
        print("\n--- [BEST] Оценка с разделением на repeat/new items ---")
        split_best = evaluate_topk_split(
            model            = ncf_best,
            test_loader      = test_loader,
            top_k            = top_k,
            device           = device,
            user_train_items = user_pos_train_set,
            isomap_model     = isomap_best,
        )
        print(f"  HR@{top_k} all:      {split_best['HR@K_all']:.4f}")
        print(f"  NDCG@{top_k} all:    {split_best['NDCG@K_all']:.4f}")
        print(f"  HR@{top_k} repeat:   {split_best['HR@K_repeat']:.4f}  (n={split_best['n_repeat']})")
        print(f"  NDCG@{top_k} repeat: {split_best['NDCG@K_repeat']:.4f}")
        print(f"  HR@{top_k} new:      {split_best['HR@K_new']:.4f}  (n={split_best['n_new']})")
        print(f"  NDCG@{top_k} new:    {split_best['NDCG@K_new']:.4f}")

        # ── Repeat / New: LAST ──
        print("\n--- [LAST] Оценка с разделением на repeat/new items ---")
        split_last = evaluate_topk_split(
            model            = ncf_last,
            test_loader      = test_loader,
            top_k            = top_k,
            device           = device,
            user_train_items = user_pos_train_set,
            isomap_model     = isomap_last,
        )
        print(f"  HR@{top_k} all:      {split_last['HR@K_all']:.4f}")
        print(f"  NDCG@{top_k} all:    {split_last['NDCG@K_all']:.4f}")
        print(f"  HR@{top_k} repeat:   {split_last['HR@K_repeat']:.4f}  (n={split_last['n_repeat']})")
        print(f"  NDCG@{top_k} repeat: {split_last['NDCG@K_repeat']:.4f}")
        print(f"  HR@{top_k} new:      {split_last['HR@K_new']:.4f}  (n={split_last['n_new']})")
        print(f"  NDCG@{top_k} new:    {split_last['NDCG@K_new']:.4f}")

        # ── сохранить итоговые test-метрики рядом с графиками ──
        test_summary = {
            "BEST": {"HR": hr_best, "NDCG": ndcg_best,
                     "repeat_new": split_best},
            "LAST": {"HR": hr_last, "NDCG": ndcg_last,
                     "repeat_new": split_last},
        }
        with open(os.path.join(images_dir, "test_summary.json"), "w") as f:
            json.dump(test_summary, f, indent=4, default=float)

        #isomap_model, ncf_manifold_model = gi_cf.train(
        #    val_loader = val_loader,
        #    top_k      = top_k,
        #    device     = device,
        #)
#
        #hr_iso, ndcg_iso = evaluate_topk_isomap(
        #    ncf_manifold_model, isomap_model, test_loader, top_k, device
        #)
        #print(f"GradientIsomapCF final TEST: "
        #      f"HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")

        #
        #
        #images_dir = os.path.join(logs_gi_dir, f"{n_run}/images")
        #plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")
#
        #with open(f"{logs_gi_dir}/{n_run}/images/history.json", "w",
        #          encoding="utf-8") as f:
        #    json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)
#
        #plot_gi_convergence(
        #    gi_cf.history, top_k=top_k,
        #    save_path=os.path.join(logs_gi_dir, f"{n_run}/images/metrics.png"),
        #)
#
        #        # === ДОПОЛНИТЕЛЬНАЯ ОЦЕНКА С РАЗДЕЛЕНИЕМ НА REPEAT/NEW ===
        #print("\n--- Оценка с разделением на repeat/new items ---")
#
        ## Передаем user_pos_train_set (он уже содержит ТОЛЬКО train-взаимодействия!)
        #split_results = evaluate_topk_split(
        #    model=ncf_manifold_model,
        #    test_loader=test_loader,
        #    top_k=top_k,
        #    device=device,
        #    user_train_items=user_pos_train_set,  # <--- ИСПРАВЛЕНО ЗДЕСЬ
        #    isomap_model=isomap_model,            # <--- ИСПРАВЛЕНО ЗДЕСЬ
        #)
#
        ## Выводим подробный отчет
        #print(f"  HR@{top_k} all:        {split_results['HR@K_all']:.4f}")
        #print(f"  NDCG@{top_k} all:      {split_results['NDCG@K_all']:.4f}")
        #print(f"  HR@{top_k} repeat:     {split_results['HR@K_repeat']:.4f} (n={split_results['n_repeat']})")
        #print(f"  NDCG@{top_k} repeat:   {split_results['NDCG@K_repeat']:.4f}")
        #print(f"  HR@{top_k} new:        {split_results['HR@K_new']:.4f} (n={split_results['n_new']})")
        #print(f"  NDCG@{top_k} new:      {split_results['NDCG@K_new']:.4f}")
#
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
    #main(
    #    dataset_name   = DATASET_AMAZON,
    #    dataset_config = {
    #        "hf_name":    "cogsci13/Amazon-Reviews-2023-Books-Review",
    #        "hf_config":  "raw_review_Books",
    #        "cache_path": "data/amazon_books_reviews.parquet",
    #    },
    #    max_users      = 2000,
    #    max_movies     = 5000,
    #    min_seq_len    = 1,
    #    num_ng         = 2,
    #    top_k          = 10,
    #    epochs_pure    = 100,
    #    gradisomap_epochs = 30,
    #    run_ncf        = False,
    #    run_gincf      = True,
    #    n_run          = 717,
    #)

    # ── T-ECD ─────────────────────────────────────────────────
    main(
                dataset_name   = DATASET_TECD,
                dataset_config = {
                    "subset_path": "recsys/GradIsomapCF_movielens/data/tecd/tecd_marketplace_subset.parquet",
                    "use_positive_only": True,   # как MovieLens — только позитив
                },
                max_users      = 500,
                max_movies     = 1500,
                min_seq_len    = 4,
                num_ng         = 2,
                top_k          = 20,
                epochs_pure    = 100,
                gradisomap_epochs = 30,
                run_ncf        = False,
                run_gincf      = True,
                n_run          = 816,
            )
    
    