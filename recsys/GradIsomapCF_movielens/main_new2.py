import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import DataLoader
import json
import time
import copy
import torch.nn.functional as F

from GradientIsomapCF_log import GradientIsomapCF

from evaluation import evaluate_topk_isomap, evaluate_topk_pure
from NCF import NCF
from prepare_data import prepare_sequences, subsample_users_items, train_val_test_split_next_item, build_movie_user_matrix
from manifold_visualization import plot_gi_losses, plot_pure_ncf_losses, plot_pure_ncf_metrics,  plot_gi_convergence
from new_datasets import NCFTrainDatasetFutureBlind, NCFTestDatasetSampled


def load_movielens_1m_ratings(ml1m_dir):
    ratings_path = os.path.join(ml1m_dir, "ratings.dat")
    print(f"Загружаем рейтинги из {ratings_path} ...")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"]
    )
    print("Unique_users:", df['userId'].nunique(), 'Unique_items:', df['movieId'].nunique())
    print(f"Загружено {len(df)} рейтингов")
    return df


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        p_t = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


def main(
        max_users=300,
        max_movies=800,
        min_seq_len=5,
        num_ng=3,
        top_k=10,
        epochs_pure=10,
        gradisomap_epochs=5,
        run_ncf=True,
        run_gincf=True,
        n_run=68,
        ):

    ml1m_dir = r"data\ml-1m"
    ratings_df = load_movielens_1m_ratings(ml1m_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")

    print("\nПодготовка последовательностей...")
    df_mapped, user2seq = prepare_sequences(ratings_df)

    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped,
        max_users=max_users,
        max_movies=max_movies,
        min_seq_len=min_seq_len,
    )

    user_pos_set = {
        u: set(m for (m, ts, r) in seq)
        for u, seq in user2seq_sub.items()
    }

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
        seed=42
    )

    val_dataset = NCFTestDatasetSampled(
        next_triples=val_next,
        num_items=num_movies,
        user_pos_all_set=user_hist_val_set,
        num_ng=99,
        seed=123
    )

    test_dataset = NCFTestDatasetSampled(
        next_triples=test_next,
        num_items=num_movies,
        user_pos_all_set=user_hist_test_set,
        num_ng=99,
        seed=456
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    if run_ncf:
        pure_model = NCF(
            user_num=num_users,
            item_num=num_movies,
            factor_num=8,
            num_layers=4,
            dropout=0.0,
            model='NeuMF-end'
        ).to(device)
        print("factor_num=8, num_layers=4")

        #loss_fn = nn.BCEWithLogitsLoss()
        pos_weight = torch.tensor([num_ng], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #loss_fn = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=torch.tensor([num_ng], device=device))

        optimizer = optim.Adam(pure_model.parameters(), lr=0.001, weight_decay=1e-6)

        best_hr_val, best_ndcg_val, best_epoch_pure = 0.0, 0.0, 0
        best_state_pure = None
        best_val_loss = 10

        pure_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'hr_val': [],
            'ndcg_val': []
        }

        start_total_time = time.time()

        for epoch in range(epochs_pure):

            start_epoch_time = time.time()

            pure_model.train()
            train_dataset.ng_sample()
            total_train_loss = 0.0
            n_train_batches = 0

            for user, item, label in train_loader:
                user = user.to(device)
                item = item.to(device)
                label = label.to(device)

                preds = pure_model(user, item)
                loss = loss_fn(preds, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)

            pure_model.eval()
            total_val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for user, item, label in val_loader:
                    user = user.to(device)
                    item = item.to(device)
                    label = label.to(device)

                    preds = pure_model(user, item)
                    loss_val = loss_fn(preds, label)
                    total_val_loss += loss_val.item()
                    n_val_batches += 1

            avg_val_loss = total_val_loss / max(1, n_val_batches)

            hr_val, ndcg_val = evaluate_topk_pure(pure_model, val_loader, top_k, device)

            epoch_time = time.time() - start_epoch_time

            print(f"[PURE NCF] epoch {epoch}: "
                  f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                  f"HR@{top_k}_val={hr_val:.4f}, NDCG@{top_k}_val={ndcg_val:.4f}, time={epoch_time:.2f}")

            pure_history['epoch'].append(epoch)
            pure_history['train_loss'].append(avg_train_loss)
            pure_history['val_loss'].append(avg_val_loss)
            pure_history['hr_val'].append(hr_val)
            pure_history['ndcg_val'].append(ndcg_val)

            if avg_val_loss < best_val_loss:
                best_val_loss, best_epoch_pure = avg_val_loss, epoch
                best_hr_val, best_ndcg_val = hr_val, ndcg_val
                best_state_pure = copy.deepcopy(pure_model.state_dict())

            if avg_val_loss > avg_train_loss:
                break

        print(f"PURE NCF best epoch {best_epoch_pure}: "
              f"HR@{top_k}_val={best_hr_val:.4f}, NDCG@{top_k}_val={best_ndcg_val:.4f}")

        if best_state_pure is not None:
            pure_model.load_state_dict(best_state_pure)

        hr_test_pure, ndcg_test_pure = evaluate_topk_pure(pure_model, test_loader, top_k, device)
        print(f"PURE NCF final on TEST: HR@{top_k}={hr_test_pure:.4f}, NDCG@{top_k}={ndcg_test_pure:.4f}")

        total_time = time.time() - start_total_time
        print(f"Total time={total_time}")
        print(f"Total_time={time.strftime('%H:%M:%S', time.gmtime(total_time))}")

        images_dir = os.path.join("logs_pure_ncf", "images/new_val")
        os.makedirs(images_dir, exist_ok=True)
        plot_pure_ncf_losses(pure_history, images_dir, run_name=f"pure_ncf_new_{n_run}_focal_losses")
        plot_pure_ncf_metrics(pure_history, images_dir, run_name=f"pure_ncf_new_{n_run}_focal_metrics")

    if run_gincf:
        print("\n=== Обучение GradientIsomapCF (IsomapNN + NeuMFOnManifold) ===")

        movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True)
        features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

        gi_cf = GradientIsomapCF(
            train_feature=features_t,
            train_events=train_events,
            num_users=num_users,
            num_items=num_movies,
            user_pos_set=user_pos_set,
            ng_seed=42,
            latent_len=64,  # размерность manifold Z
            n_neighbors=10,
            epochs=gradisomap_epochs,  # outer_epochs
            cf_epochs=100,  # внутренних эпох CF на фиксированном Z
            final_cf_epochs=100,  # количество эпох для финальной модели
            batch_size=2048,
            lr_isomap=5e-2,  # 1e-4
            lr_ncf=1e-3, #1e-3,  # 1e-3
            factor_num=16,
            num_layers=3,
            dropout=0.0,  # 0.0
            model_type='NeuMF-end',
            logs_folder=f"logs_movielens_isomap_cf/{n_run}",
            device=str(device),
            stop_criteria_value=0.001,
            num_ng=num_ng
        )

        isomap_model, ncf_manifold_model = gi_cf.train(
            val_loader=val_loader,
            top_k=top_k,
            device=device
        )

        hr_iso, ndcg_iso = evaluate_topk_isomap(ncf_manifold_model, isomap_model, test_loader, top_k, device)
        print(f"GradientIsomapCF final on TEST: HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")
        print(f"GI + NeuMFOnM HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")

        images_dir = os.path.join("logs_movielens_isomap_cf", f"{n_run}/images")
        plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")

        with open(f"logs_movielens_isomap_cf/{n_run}/images/history.json", 'w', encoding='utf-8') as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        images_dir_png = os.path.join("logs_movielens_isomap_cf", f"{n_run}/images/metrics.png")
        plot_gi_convergence(gi_cf.history, top_k=top_k, save_path=images_dir_png)


if __name__ == "__main__":

    """
    есть выбор режима - что запускать, только классический NCF или GradientIsomapNCF (можно вместе)
    run_ncf - запускаем NCF
    run_gincf - запускаем GradientIsomapNCF
    epochs_pure - максимальное кол-во эпох для чистого NCF (параметры настраиваются при определении модели стр 135-141)
    gradisomap_epochs - число эпох на внешнем цикле (параметры внутреннего и внешнего цикла настраиваются при определении модели выше)
    
    n_run - номер эксперимента - он будет либо в названии файлов-графиков, либо в названии папки 
    """
    main(
        max_users=300,
        max_movies=800,
        min_seq_len=2,
        num_ng=2,
        top_k=10,
        epochs_pure=100,
        gradisomap_epochs=30,
        run_ncf=True,
        run_gincf=False,
        n_run=68
        )
