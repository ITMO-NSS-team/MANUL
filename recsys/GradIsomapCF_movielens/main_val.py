import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import DataLoader
import json

#from GradientIsomapCF_val import GradientIsomapCF
#from GradientIsomapCF_alternating import GradientIsomapCF
#from GradientIsomapCF_correct import GradientIsomapCF
#from GradientIsomapCF_reinit import GradientIsomapCF

#from GradientIsomapCF_reinit_new import GradientIsomapCF
from GradientIsomapCF_normalization import GradientIsomapCF

#from GradientIsomapCF_reinit_val import GradientIsomapCF
from evaluation import evaluate_topk_isomap, evaluate_topk_pure
from NCF_datasets import NCFTestDataset, NCFTrainDataset
from NCF import NCF
from prepare_data import prepare_sequences, subsample_users_items, train_val_test_split_next_item, build_movie_user_matrix
from manifold_visualization import plot_gi_losses, plot_pure_ncf_losses, plot_gi_convergence, plot_isomap_deltas  # plot_manifold_isomap, plot_gi_convergence, plot_movie_pca, plot_cf_inner_losses
from isomap_analysis import analyze_distance_matrices


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
    print(f"Загружено {len(df)} рейтингов")
    return df


def main(
        max_users=300,
        max_movies=800,
        min_seq_len=5,
        num_ng=3,
        top_k=10,
        epochs_pure=10,
        gradisomap_epochs=5,
        run_ncf=True,
        run_gincf=True
        ):

    ml1m_dir = r"data\ml-1m"
    ratings_df = load_movielens_1m_ratings(ml1m_dir)

    print("\nПодготовка последовательностей...")
    df_mapped, user2seq = prepare_sequences(ratings_df)

    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped,
        max_users=max_users,
        max_movies=max_movies,
        min_seq_len=min_seq_len,
    )

    print("\nTrain/Val/Test split (next-item)...")
    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)
    print(f"Train событий: {len(train_events)}, Val пользователей: {len(val_next)}, Test пользователей: {len(test_next)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")

    print("\n=== Обучение pure NeuMF-end (без Isomap) ===")

    train_pairs = []
    train_mat = sp.dok_matrix((num_users, num_movies), dtype=np.float32)
    for (u, m, r) in train_events:
        u = int(u)
        m = int(m)
        train_pairs.append((u, m))
        train_mat[u, m] = 1.0

    train_dataset = NCFTrainDataset(train_pairs, num_movies, train_mat, num_ng=num_ng)
    val_dataset = NCFTestDataset(val_next,  num_movies, train_mat, num_ng=99)
    test_dataset = NCFTestDataset(test_next, num_movies, train_mat, num_ng=99)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    if run_ncf:
        pure_model = NCF(
            user_num=num_users,
            item_num=num_movies,
            factor_num=32,
            num_layers=3,
            dropout=0.1,
            model='NeuMF-end'
        ).to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(pure_model.parameters(), lr=0.001)

        best_hr_val, best_ndcg_val, best_epoch_pure = 0.0, 0.0, 0
        best_state_pure = None

        # история для графика
        pure_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(epochs_pure):
            # ----- TRAIN -----
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

            # ----- VAL BCE-LOSS -----
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

            # ----- VAL HR/NDCG -----
            hr_val, ndcg_val = evaluate_topk_pure(pure_model, val_loader, top_k, device)

            print(f"[PURE NCF] epoch {epoch}: "
                  f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                  f"HR@{top_k}_val={hr_val:.4f}, NDCG@{top_k}_val={ndcg_val:.4f}")

            # логируем историю
            pure_history['epoch'].append(epoch)
            pure_history['train_loss'].append(avg_train_loss)
            pure_history['val_loss'].append(avg_val_loss)

            # выбор лучшей эпохи по HR@K (как было)
            if hr_val > best_hr_val:
                best_hr_val, best_ndcg_val, best_epoch_pure = hr_val, ndcg_val, epoch
                best_state_pure = pure_model.state_dict()

        print(f"PURE NCF best epoch {best_epoch_pure}: "
              f"HR@{top_k}_val={best_hr_val:.4f}, NDCG@{top_k}_val={best_ndcg_val:.4f}")

        # восстанавливаем лучшую по HR модель
        if best_state_pure is not None:
            pure_model.load_state_dict(best_state_pure)

        # финальная оценка на тесте
        hr_test_pure, ndcg_test_pure = evaluate_topk_pure(pure_model, test_loader, top_k, device)
        print(f"PURE NCF final on TEST: HR@{top_k}={hr_test_pure:.4f}, NDCG@{top_k}={ndcg_test_pure:.4f}")

        # строим и сохраняем график train/val-loss
        images_dir = os.path.join("logs_pure_ncf", "images")  # можно подставить свой путь/имя run'а
        os.makedirs(images_dir, exist_ok=True)
        plot_pure_ncf_losses(pure_history, images_dir, run_name="pure_ncf")

    if run_gincf:
        print("\n=== Обучение GradientIsomapCF (IsomapNN + NeuMFOnManifold) ===")

        movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True)
        features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

        gi_cf = GradientIsomapCF(
            train_feature=features_t,
            train_events=train_events,
            num_users=num_users,
            num_items=num_movies,
            latent_len=128,  # размерность manifold Z
            n_neighbors=10,
            epochs=gradisomap_epochs,  # outer_epochs
            cf_epochs=100,  # внутренних эпох CF на фиксированном Z
            final_cf_epochs=100,  # количество эпох для финальной модели
            batch_size=2048,
            lr_isomap=1e-1,  # 1e-4
            lr_ncf=5e-4, #1e-3,  # 1e-3
            factor_num=32,
            num_layers=3,
            dropout=0.0,  # 0.0
            model_type='NeuMF-end',
            logs_folder="logs_movielens_isomap_cf",
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

        images_dir = os.path.join("logs_movielens_isomap_cf", "run48/images")
        plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")

        with open("logs_movielens_isomap_cf/run48/images/history.json", 'w', encoding='utf-8') as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        with open("logs_movielens_isomap_cf/run48/images/dist_history.json", 'w', encoding='utf-8') as f:
            json.dump(gi_cf.dist_history, f, ensure_ascii=False, indent=4)

        images_dir = os.path.join("logs_movielens_isomap_cf", "run48/images")
        plot_isomap_deltas(gi_cf.dist_history, images_dir, run_name="ginmf")

        images_dir_png = os.path.join("logs_movielens_isomap_cf", "run48/images/metrics.png")
        plot_gi_convergence(gi_cf.history, top_k=top_k, save_path=images_dir_png)

        logs = gi_cf.logs_folder

        D_paths = {}
        for i in range(gradisomap_epochs):
            D_paths[f"epoch{i}"] = os.path.join(logs, f"D_epoch{i}.npy")

        images_dir = os.path.join(logs, "run48/images/D_analysis")
        analyze_distance_matrices(D_paths, k=10, images_dir=images_dir, run_name="ginmf")


        # 4. Визуализация manifold'а фильмов (Isomap)
        #plot_manifold_isomap(isomap_model, train_events, num_movies, color_by='rating')
        #plot_manifold_isomap(isomap_model, train_events, num_movies, color_by='popularity')

        # 5. Baseline: PCA по исходной матрице фильм×пользователь
        #plot_movie_pca(movie_user_mat, train_events, color_by='rating')
        #plot_movie_pca(movie_user_mat, train_events, color_by='popularity')

        #images_dir = os.path.join("logs_movielens_isomap_cf", 'run_14', "images")
        #plot_cf_inner_losses(gi_cf.cf_history, images_dir=images_dir, run_name='run_14')


if __name__ == "__main__":

    main(
        max_users=300,
        max_movies=800,
        min_seq_len=5,
        num_ng=2,
        top_k=10,
        epochs_pure=30,
        gradisomap_epochs=30,
        run_ncf=False,
        run_gincf=True
        )
