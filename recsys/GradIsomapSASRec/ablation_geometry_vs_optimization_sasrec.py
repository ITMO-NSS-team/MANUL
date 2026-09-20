"""
SASRec-версия ablation_geometry_vs_optimization.py.

Отвечает на тот же вопрос: помогает ли outer-loop оптимизация D_input,
или финальный SASRec переобучается так же хорошо на неоптимизированной
начальной геометрии?

Сравнивает три снимка геометрии для заданной конфигурации:
  - "pure_init": D_input_init.npy → один forward IsomapNN без обучения
    (0 outer шагов — чистая евклидова cdist геометрия)
  - "epoch0": Z из matrices_epoch0.npz (один outer шаг)
  - "epochN": Z из последнего matrices_epoch*.npz (полная сходимость)

Переиспользует сохранённые Z-снимки из уже завершённого sweep-запуска
GradientIsomapSASRec — только шаг "обучить SASRec-голову на фиксированном Z"
выполняется заново, аналогично ablation_geometry_vs_optimization.py.

Дополнительно содержит:
  - build_data_sasrec(): подготовка данных для последовательного обучения
  - train_and_eval_sasrec_on_fixed_Z(): SASRecOnManifold на фиксированном Z
  - train_and_eval_sasrec_euclidean_baseline(): чистый SASRec из source.py
"""
import argparse
import copy
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))                       # .../MANUL/recsys/GradIsomapSASRec
MANUL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # .../MANUL
NSS_LAB_DIR = os.path.dirname(MANUL_DIR)                                # .../ (папка, содержащая MANUL)
GINCF_DIR = os.path.join(MANUL_DIR, "recsys", "GradIsomapCF_movielens")  # оригинальная папка с prepare_data/run_experiment и data

sys.path.insert(0, NSS_LAB_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from Adam.Isomap import IsomapNN
from SASRecOnManifold_best import SASRecOnManifold, pad_item_Z
from sasrec_manifold_sampler import SASRecManifoldTrainSampler, SASRecManifoldTestDataset
from evaluation_sasrec_manifold import evaluate_topk_sasrec_isomap
from recsys.GradIsomapCF_movielens.prepare_data import (
    prepare_sequences, subsample_users_items,
    train_val_test_split_next_item, build_movie_user_matrix,
)
from run_experiment_sasrec import load_movielens_ratings, load_amazon_ratings

# source.py SASRec — чистая архитектура без манифолда
from Hyperbolic_SASRec.source import SASRec
from Hyperbolic_SASRec.sampler import batch_sequence_sampler


class _IdentityIsomap(nn.Module):
    """
    Оборачивает предвычисленный Z так, чтобы evaluate_topk_sasrec_isomap
    (который вызывает isomap_model()) работал без изменений.
    Зеркало _IdentityIsomap из ablation_geometry_vs_optimization.py.
    """
    def __init__(self, Z: torch.Tensor):
        super().__init__()
        self.Z = Z

    def forward(self):
        return self.Z

    def eval(self):
        return self


# ──────────────────────────────────────────────────────────────────
#  build_data_sasrec
# ──────────────────────────────────────────────────────────────────

def build_data_sasrec(
    max_users, max_movies, min_seq_len, num_ng,
    dataset_dir_name, device, tmp_logs_folder,
    dataset_type="movielens",
    amazon_category="Beauty_and_Personal_Care",
    maxlen=50,
    ng_seed=42,
):
    """
    Зеркало build_data() из ablation_geometry_vs_optimization.py,
    адаптированное для последовательного обучения SASRec.

    Ключевые отличия от build_data():
      - Возвращает user_train (dict {uid: [item...]}) вместо inter_loader
      - Возвращает SASRecManifoldTestDataset для val/test вместо NCFTestDatasetSampled
      - Строит train_sampler (SASRecManifoldTrainSampler) как аналог inter_loader
      - Возвращает train_ranking_loader — диагностический загрузчик,
        позволяет строить train HR@10 рядом с val HR@10 (те же 1-vs-99)

    Отрицательная выборка: user_pos_set = train+val+test (как у научного
    руководителя в build_data для NCF — намеренно воспроизводит поведение
    GradientIsomapSASRec.__init__._build_implicit_interactions).
    """
    dataset_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)
    if dataset_type == "movielens":
        ratings_df = load_movielens_ratings(dataset_dir)
    elif dataset_type == "amazon":
        ratings_df = load_amazon_ratings(dataset_dir, amazon_category)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    df_mapped, user2seq = prepare_sequences(ratings_df)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies,
        min_seq_len=min_seq_len,
    )

    # Полное множество (train+val+test) — для исключения из негативов
    user_pos_set = {
        u: set(m for (m, ts, r) in seq)
        for u, seq in user2seq_sub.items()
    }

    train_events, val_next, test_next = train_val_test_split_next_item(
        user2seq_sub, min_len=3
    )

    # Train-only множества для exclusion sets val/test
    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))

    user_hist_val_set = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    # Последовательности (только train)
    user_train = defaultdict(list)
    for (u, m, r) in sorted(train_events, key=lambda x: (x[0], x[1])):
        user_train[int(u)].append(int(m))
    user_train = dict(user_train)

    # Val / Test датасеты
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

    # Диагностический train ranking loader
    # Один представительный (user, target_item) на пользователя из train,
    # оцениваемый по тому же 1-vs-99 протоколу что val/test.
    # Позволяет строить train HR@10 рядом с val HR@10.
    train_last_per_user = {}
    for (u, m, r) in train_events:
        train_last_per_user[int(u)] = int(m)
    train_ranking_triples = [(u, 0, m) for u, m in train_last_per_user.items()]
    train_ranking_dataset = SASRecManifoldTestDataset(
        user_train=user_train,
        next_triples=train_ranking_triples,
        num_items=num_movies,
        user_pos_all_set=user_pos_train_set,  # исключаем только train
        maxlen=maxlen, num_ng=99, seed=789,
    )

    # Признаковая матрица для D_input
    movie_user_mat = build_movie_user_matrix(
        train_events, num_movies, num_users, use_ratings=True
    )
    features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

    # Train sampler — зеркало gi_cf.inter_loader из build_data()
    # user_pos_set передаётся как в GradientIsomapSASRec (полное множество)
    train_sampler = SASRecManifoldTrainSampler(
        user_train=user_train,
        n_items=num_movies,
        maxlen=maxlen,
        user_pos_train_set=user_pos_set,   
        seed=ng_seed,
    )

    n_users_with_seq = len([u for u, s in user_train.items() if len(s) >= 2])
    # batch_size=256 — тот же дефолт что в GradientIsomapSASRec
    n_batches_per_epoch = max(1, n_users_with_seq // 256)

    with torch.no_grad():
        D_input_init = torch.cdist(features_t, features_t)
        D_input_init = D_input_init / D_input_init.max()

    return {
        "num_users": num_users,
        "num_movies": num_movies,
        "user_train": user_train,
        "user_pos_set": user_pos_set,
        "train_sampler": train_sampler,
        "n_batches_per_epoch": n_batches_per_epoch,
        "val_loader": DataLoader(val_dataset, batch_size=100, shuffle=False),
        "test_loader": DataLoader(test_dataset, batch_size=100, shuffle=False),
        "train_ranking_loader": DataLoader(train_ranking_dataset, batch_size=100, shuffle=False),
        "D_input_init": D_input_init,
        "maxlen": maxlen,
    }


# ──────────────────────────────────────────────────────────────────
#  train_and_eval_sasrec_on_fixed_Z
# ──────────────────────────────────────────────────────────────────

def train_and_eval_sasrec_on_fixed_Z(
    item_Z_raw: torch.Tensor,
    data: dict,
    device,
    sasrec_config: dict,
    latent_dim: int,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 3,
    seed: int = 0,
    verbose: bool = False,
    return_history: bool = False,
    select_by: str = "hr",
    freeze_item_projection: bool = False,
    batch_size: int = 256,
):
    """
    Зеркало train_and_eval_ncf_on_fixed_Z() для SASRec.

    Обучает SASRecOnManifold на фиксированном Z (outer loop НЕ запускается).
    Исправление deepcopy применено.

    select_by="hr" (рекомендуется): выбор чекпоинта по val HR@10.
    select_by="loss": по val BCE loss.

    Отличие от train_and_eval_ncf_on_fixed_Z:
      - Вход: последовательности (seq, pos, neg) вместо (user, item, label)
      - Модель: SASRecOnManifold вместо NeuMFOnManifold
      - Loss: только на непадинговых позициях (где pos != pad_token)
    """
    torch.manual_seed(seed)
    num_movies = data["num_movies"]
    maxlen = data.get("maxlen", sasrec_config.get("maxlen", 50))
    pad_token = num_movies

    item_Z_raw = item_Z_raw.to(device)
    item_Z = pad_item_Z(item_Z_raw)

    sasrec = SASRecOnManifold(
        config=sasrec_config,
        item_num=num_movies,
        latent_dim=latent_dim,
        freeze_item_projection=freeze_item_projection,
        item_projection_init_data=item_Z_raw if freeze_item_projection else None,
    ).to(device)

    trainable_params = [p for p in sasrec.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_hr = -float("inf")
    best_state = None
    no_improve = 0

    train_sampler = data["train_sampler"]
    n_batches = data["n_batches_per_epoch"]
    train_ranking_loader = data.get("train_ranking_loader")
    history = {"train_loss": [], "val_loss": [], "val_hr": [], "train_hr": []}

    # IterableDataset → бесконечный итератор
    train_loader = DataLoader(train_sampler, batch_size=batch_size, num_workers=0)
    batch_iter = iter(train_loader)

    for ep in range(epochs):
        sasrec.train()
        total_loss = 0.0

        for _ in range(n_batches):
            seq, pos, neg = next(batch_iter)
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)

            pos_logits, neg_logits = sasrec(seq, pos, neg, item_Z)
            indices = torch.where(pos != pad_token)
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)
            batch_loss = loss_fn(pos_logits[indices], pos_labels[indices])
            batch_loss += loss_fn(neg_logits[indices], neg_labels[indices])

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_train = total_loss / n_batches
        history["train_loss"].append(avg_train)

        # ── Val ──
        sasrec.eval()
        total_vl, n_vl = 0.0, 0
        hits = []
        with torch.no_grad():
            for v_seq, v_cands, v_labels in data["val_loader"]:
                v_seq = v_seq.to(device)
                v_cands = v_cands.to(device)
                v_labels = v_labels.to(device)
                v_logits = sasrec.predict_candidates(v_seq, v_cands, item_Z)
                total_vl += loss_fn(v_logits, v_labels).item()
                n_vl += 1
                # Инлайн HR@10 — позитив всегда на индексе 0
                # (NCFTestDatasetSampled / SASRecManifoldTestDataset: target первый)
                _, topk_idx = torch.topk(v_logits.squeeze(0)
                                         if v_logits.shape[0] == 1 else v_logits, 10, dim=-1)
                for b in range(v_logits.shape[0]):
                    hits.append(1.0 if 0 in topk_idx[b].tolist() else 0.0)

        avg_vl = total_vl / max(1, n_vl)
        avg_hr = float(np.mean(hits)) if hits else 0.0
        history["val_loss"].append(avg_vl)
        history["val_hr"].append(avg_hr)

        # Диагностический train HR@10
        avg_train_hr = None
        if train_ranking_loader is not None:
            train_hits = []
            with torch.no_grad():
                for tr_seq, tr_cands, tr_labels in train_ranking_loader:
                    tr_seq = tr_seq.to(device)
                    tr_cands = tr_cands.to(device)
                    tr_logits = sasrec.predict_candidates(tr_seq, tr_cands, item_Z)
                    for b in range(tr_logits.shape[0]):
                        _, topk_idx = torch.topk(tr_logits[b], 10)
                        train_hits.append(1.0 if 0 in topk_idx.tolist() else 0.0)
            avg_train_hr = float(np.mean(train_hits)) if train_hits else 0.0
            history["train_hr"].append(avg_train_hr)

        if verbose:
            train_hr_str = (f" train_hr@10={avg_train_hr:.4f}"
                            if avg_train_hr is not None else "")
            print(f"  ep {ep+1}/{epochs} train={avg_train:.4f} val={avg_vl:.4f} "
                  f"val_hr@10={avg_hr:.4f}{train_hr_str}", flush=True)

        # Выбор чекпоинта — deepcopy 
        improved = (avg_hr > best_val_hr) if select_by == "hr" \
            else (avg_vl < best_val_loss)
        if improved:
            best_val_loss = avg_vl
            best_val_hr = avg_hr
            best_state = copy.deepcopy(sasrec.state_dict())  # deepcopy — fix
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"  early stop at ep {ep+1}", flush=True)
            break

    if best_state is not None:
        sasrec.load_state_dict(best_state)

    # Evaluate on test
    identity = _IdentityIsomap(item_Z_raw)
    hr, ndcg = evaluate_topk_sasrec_isomap(
        sasrec, identity, data["test_loader"], top_k=10, device=device
    )

    if return_history:
        return hr, ndcg, best_val_loss, history
    return hr, ndcg, best_val_loss


# ──────────────────────────────────────────────────────────────────
#  train_and_eval_sasrec_euclidean_baseline
# ──────────────────────────────────────────────────────────────────

def train_and_eval_sasrec_euclidean_baseline(
    data: dict,
    device,
    sasrec_config: dict,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 3,
    seed: int = 0,
    verbose: bool = False,
    return_history: bool = False,
    return_item_embeddings: bool = False,
    select_by: str = "hr",
    batch_size: int = 256,
):
    """
    Чистый SASRec из source.py — без манифолда, с обучаемым item_emb
    (nn.Embedding). Геометрия пространства айтемов находится через
    gradient descent, а не задаётся через Isomap.

    Зеркало train_and_eval_euclidean_baseline() для SASRec.

    Обучается через те же train_sampler/val_loader/test_loader что и
    train_and_eval_sasrec_on_fixed_Z — единственное отличие между
    тремя руками сравнения — это источник item представлений.

    return_item_embeddings=True: возвращает веса item_emb.weight
    (обученные евклидовы координаты айтемов) для вычисления матрицы
    расстояний и записи в full_hyperbolicity_table.
    """
    torch.manual_seed(seed)
    num_movies = data["num_movies"]
    pad_token = num_movies

    sasrec = SASRec(config=sasrec_config, item_num=num_movies).to(device)
    optimizer = optim.Adam(sasrec.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_hr = -float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_hr": []}

    train_sampler = data["train_sampler"]
    n_batches = data["n_batches_per_epoch"]
    train_loader = DataLoader(train_sampler, batch_size=batch_size, num_workers=0)
    batch_iter = iter(train_loader)

    for ep in range(epochs):
        sasrec.train()
        total_loss = 0.0

        for _ in range(n_batches):
            seq, pos, neg = next(batch_iter)
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # source.py forward: принимает (seq, pos, neg) — всё без item_Z
            pos_logits, neg_logits = sasrec(seq, pos, neg)
            indices = torch.where(pos != pad_token)
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)
            batch_loss = loss_fn(pos_logits[indices], pos_labels[indices])
            batch_loss += loss_fn(neg_logits[indices], neg_labels[indices])

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_train = total_loss / n_batches
        history["train_loss"].append(avg_train)

        # ── Val ──
        sasrec.eval()
        total_vl, n_vl = 0.0, 0
        hits = []
        maxlen = sasrec_config.get("maxlen", 50)

        with torch.no_grad():
            for v_seq, v_cands, v_labels in data["val_loader"]:
                v_seq = v_seq.to(device)
                v_cands = v_cands.to(device)
                v_labels = v_labels.to(device)

                # source.py predict_candidates: нет готового метода,
                # поэтому используем log2feats + точечное произведение
                log_feats = sasrec.log2feats(v_seq)        # [B, L, H]
                final_feat = log_feats[:, -1, :]           # [B, H]
                cand_embs = sasrec.item_emb(v_cands)       # [B, C, H]
                v_logits = (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1)  # [B, C]

                total_vl += loss_fn(v_logits, v_labels).item()
                n_vl += 1
                for b in range(v_logits.shape[0]):
                    _, topk_idx = torch.topk(v_logits[b], 10)
                    hits.append(1.0 if 0 in topk_idx.tolist() else 0.0)

        avg_vl = total_vl / max(1, n_vl)
        avg_hr = float(np.mean(hits)) if hits else 0.0
        history["val_loss"].append(avg_vl)
        history["val_hr"].append(avg_hr)

        if verbose:
            print(f"  ep {ep+1}/{epochs} train={avg_train:.4f} val={avg_vl:.4f} "
                  f"val_hr@10={avg_hr:.4f}", flush=True)

        improved = (avg_hr > best_val_hr) if select_by == "hr" \
            else (avg_vl < best_val_loss)
        if improved:
            best_val_loss = avg_vl
            best_val_hr = avg_hr
            best_state = copy.deepcopy(sasrec.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"  early stop at ep {ep+1}", flush=True)
            break

    if best_state is not None:
        sasrec.load_state_dict(best_state)

    # Test evaluation через _predict_candidates
    sasrec.eval()
    HR, NDCG = [], []
    with torch.no_grad():
        for t_seq, t_cands, t_labels in data["test_loader"]:
            t_seq = t_seq.to(device)
            t_cands = t_cands.to(device)
            t_labels = t_labels.to(device)

            log_feats = sasrec.log2feats(t_seq)
            final_feat = log_feats[:, -1, :]
            cand_embs = sasrec.item_emb(t_cands)
            t_logits = (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1)

            for b in range(t_logits.shape[0]):
                b_logits = t_logits[b]
                b_labels = t_labels[b]
                pos_indices = (b_labels == 1.0).nonzero(as_tuple=True)[0]
                if len(pos_indices) == 0:
                    continue
                pos_score = b_logits[pos_indices[0]].item()
                rank = (b_logits > pos_score).sum().item()
                if rank < 10:
                    HR.append(1.0)
                    NDCG.append(1.0 / np.log2(rank + 2))
                else:
                    HR.append(0.0)
                    NDCG.append(0.0)

    hr = float(np.mean(HR)) if HR else 0.0
    ndcg = float(np.mean(NDCG)) if NDCG else 0.0

    item_emb = None
    if return_item_embeddings:
        with torch.no_grad():
            # source.py: item_emb включает pad строку (num_items+1 строк)
            # берём только реальные айтемы [0:num_movies]
            item_emb = sasrec.item_emb.weight[:num_movies].detach().cpu()

    if return_item_embeddings and return_history:
        return hr, ndcg, best_val_loss, item_emb, history
    if return_item_embeddings:
        return hr, ndcg, best_val_loss, item_emb
    if return_history:
        return hr, ndcg, best_val_loss, history
    return hr, ndcg, best_val_loss


# ──────────────────────────────────────────────────────────────────
#  main — ablation
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True,
                        help="Папка с matrices_epoch*.npz от GradientIsomapSASRec")
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="ml-1m")
    parser.add_argument("--dataset_type", default="movielens",
                        choices=["movielens", "amazon"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--select_by", default="hr", choices=["loss", "hr"])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
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

    tmp_logs_folder = os.path.join(HERE, "ablation_sasrec_tmp_logs")
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

    # ── Загружаем Z-снимки ──
    epoch_files = glob.glob(os.path.join(args.logs_folder, "matrices_epoch*.npz"))

    def epoch_num(p):
        m = re.search(r"matrices_epoch(\d+)\.npz", os.path.basename(p))
        return int(m.group(1)) if m else -1

    epoch_files = sorted(epoch_files, key=epoch_num)
    last_epoch_file = epoch_files[-1]
    last_epoch_n = epoch_num(last_epoch_file)

    # pure_init: D_input_init → IsomapNN без обучения (0 outer шагов)
    D_init = np.load(os.path.join(args.logs_folder, "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(
        weights_initial_assumption=D_init_t,
        n_components=args.latent_dim,
        n_neighbors=10,
    ).to(device)
    with torch.no_grad():
        Z_pure_init = pure_init_model().to(torch.float32).detach()

    Z_epoch0 = torch.tensor(np.load(epoch_files[0])["Z"], dtype=torch.float32)
    Z_epochN = torch.tensor(np.load(last_epoch_file)["Z"], dtype=torch.float32)

    # ── Ablation: три геометрии ──
    results = {}
    for name, Z in [
        ("pure_init (euclidean, 0 outer steps)", Z_pure_init),
        ("epoch0 (1 outer step)", Z_epoch0),
        (f"epoch{last_epoch_n} (converged)", Z_epochN),
    ]:
        print(f"\n--- SASRec on fixed Z: {name} ---", flush=True)
        hr, ndcg, val_loss = train_and_eval_sasrec_on_fixed_Z(
            item_Z_raw=Z, data=data, device=device,
            sasrec_config=sasrec_config, latent_dim=args.latent_dim,
            select_by=args.select_by, patience=args.patience,
            epochs=args.epochs, batch_size=args.batch_size,
        )
        results[name] = (hr, ndcg, val_loss)
        print(f"{name}: test HR@10={hr:.4f} NDCG@10={ndcg:.4f} "
              f"best_val_loss={val_loss:.4f}", flush=True)

    print("\n=== ABLATION SUMMARY (SASRec) ===")
    for name, (hr, ndcg, val_loss) in results.items():
        print(f"{name:40s}  HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    main()
