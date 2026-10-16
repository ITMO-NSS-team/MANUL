import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import json
import time
import copy
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_manul = types.ModuleType("MANUL")
_manul.__path__ = [_repo_root]
sys.modules.setdefault("MANUL", _manul)

from evaluation import evaluate_topk_isomap
from NCF import NCF
from prepare_data import (
    prepare_sequences,
    subsample_users_items,
    train_val_test_split_next_item,
    build_movie_user_matrix,
    thin_train_events,
    corater_stats,
    popularity_baseline,
)

from manifold_visualization import plot_gi_losses, plot_pure_ncf_losses, plot_gi_convergence
from new_datasets import NCFTrainDatasetFutureBlind, NCFTestDatasetSampled
from fast_common import evaluate_topk_vec, eval_tensors, val_loss_vec
from determinism import seed_everything
from run_logger import RunLogger

DATASET_DIRS = {
    "ml-1m": "ml-1m",
    "ml-10m": "ml-10m",
    "amazon": "amazon_beauty",
}


def load_movielens_ratings(movielens_dir):
    ratings_path = os.path.join(movielens_dir, "ratings.dat")
    if not os.path.isfile(ratings_path):
        alt = os.path.join(os.path.dirname(movielens_dir.rstrip("/\\")), "ml-10M100K", "ratings.dat")
        if os.path.basename(movielens_dir.rstrip("/\\")) == "ml-10m" and os.path.isfile(alt):
            ratings_path = alt
        else:
            raise FileNotFoundError(f"Не найден {ratings_path}")
    print(f"Загружаем рейтинги из {ratings_path} ...")
    df = pd.read_csv(
        ratings_path, sep=":", header=None, engine="c", usecols=[0, 2, 4, 6],
        names=["userId", "movieId", "rating", "timestamp"],
    )
    print("Unique_users:", df['userId'].nunique(), 'Unique_items:', df['movieId'].nunique())
    print(f"Загружено {len(df)} рейтингов")
    return df


def _load_amazon_json(path):
    print(f"Загружаем рейтинги из {path} (классический 5-core JSON) ...")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append((
                rec["reviewerID"],
                rec["asin"],
                float(rec["overall"]),
                int(rec["unixReviewTime"]),
            ))
    df = pd.DataFrame(records, columns=["userId", "movieId", "rating", "timestamp"])
    print("Unique_users:", df['userId'].nunique(), 'Unique_items:', df['movieId'].nunique())
    print(f"Загружено {len(df)} рейтингов")
    return df


def _load_amazon_csv(path):
    print(f"Загружаем рейтинги из {path} ...")
    df = pd.read_csv(path)
    cols = {c.strip(): c for c in df.columns}

    if "user_id" in cols and "parent_asin" in cols:
        df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
    elif "userId" in cols and "itemId" in cols:
        df = df.rename(columns={"userId": "userId", "itemId": "movieId"})
    elif "user_id" in cols and "item_id" in cols:
        df = df.rename(columns={"user_id": "userId", "item_id": "movieId"})
    else:
        raise ValueError(
            f"Не удалось распознать формат Amazon CSV {path!r}: заголовок "
            f"{list(df.columns)}. Ожидаются столбцы user_id+parent_asin "
            f"(Reviews 2023) или userId+itemId (ratings-only 2014)."
        )
    df = df[["userId", "movieId", "rating", "timestamp"]]
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    print("Unique_users:", df['userId'].nunique(), 'Unique_items:', df['movieId'].nunique())
    print(f"Загружено {len(df)} рейтингов")
    return df


def load_amazon_ratings(amazon_dir, category):
    candidates = [
        os.path.join(amazon_dir, f"{category}.csv"),
        os.path.join(amazon_dir, f"{category}.csv.gz"),
        os.path.join(amazon_dir, f"{category}_5.json"),
        os.path.join(amazon_dir, f"reviews_{category}_5.json"),
        os.path.join(amazon_dir, f"ratings_{category}.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            if path.endswith(".json"):
                return _load_amazon_json(path)
            return _load_amazon_csv(path)
    raise FileNotFoundError(
        f"В {amazon_dir!r} не найден ни один Amazon-файл для категории {category!r}. "
        f"Искались: {', '.join(os.path.basename(p) for p in candidates)}. "
        f"Положите файл (например, Reviews 2023 5-core CSV "
        f"{category}.csv со столбцами user_id,parent_asin,rating,timestamp) "
        f"в эту папку."
    )


def load_ratings_df(dataset, data_root, amazon_category):
    if dataset in ("ml-1m", "ml-10m"):
        return load_movielens_ratings(os.path.join(data_root, DATASET_DIRS[dataset]))
    if dataset == "amazon":
        return load_amazon_ratings(os.path.join(data_root, DATASET_DIRS["amazon"]), amazon_category)
    raise ValueError(f"Неизвестный dataset: {dataset!r} (ожидаются ml-1m, ml-10m, amazon)")


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


def train_pure_ncf(
        pure_model,
        train_dataset,
        num_ng,
        top_k,
        device,
        val_tensors,
        epochs,
        batch_size=256,
        lr=0.001,
        weight_decay=1e-6,
        history=None,
        select_by="loss",
        patience=10,
        summary=None,
        use_pos_weight=True,
        resample_negatives=True,
):
    """
    Прежнее правило "val_loss > train_loss ---> стоп" убрано: оно сравнивает величины с разным
    соотношением классов (train 1:num_ng, val 1:99) и у GI-стадии аналога не имеет.
    """
    # use_pos_weight=False ---> обычный BCE (как у Euclid-базлайна в my_verification/current)
    pos_weight = torch.tensor([num_ng], device=device, dtype=torch.float32) if use_pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(pure_model.parameters(), lr=lr, weight_decay=weight_decay)

    vu, vi, vl = val_tensors
    score_fn = lambda u, i: pure_model(u, i)

    best_val_loss = float("inf")
    best_val_hr = -float("inf")
    no_improve = 0
    best_epoch = 0
    best_hr_val, best_ndcg_val = 0.0, 0.0
    best_state = None
    epoch_times = []
    stopped_early = False

    start_total_time = time.time()
    for epoch in range(epochs):
        start_epoch_time = time.time()

        if resample_negatives:
            train_dataset.ng_sample()
        users_all = train_dataset.users_fill.to(device)
        items_all = train_dataset.items_fill.to(device)
        labels_all = train_dataset.labels_fill.to(device)
        n = users_all.numel()

        pure_model.train()
        perm = torch.randperm(n, device=device)
        total_train_loss = 0.0
        n_train_batches = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            preds = pure_model(users_all[idx], items_all[idx])
            loss = loss_fn(preds, labels_all[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_train_batches += 1
        avg_train_loss = total_train_loss / max(1, n_train_batches)

        pure_model.eval()
        avg_val_loss = val_loss_vec(score_fn, vu, vi, vl, device, pos_weight=pos_weight)
        hr_val, ndcg_val = evaluate_topk_vec(score_fn, vu, vi, vl, top_k, device)

        epoch_time = time.time() - start_epoch_time
        epoch_times.append(epoch_time)

        print(f"[PURE NCF] epoch {epoch}: "
              f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"HR@{top_k}_val={hr_val:.4f}, NDCG@{top_k}_val={ndcg_val:.4f}, time={epoch_time:.2f}")

        if history is not None:
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['hr_val'].append(hr_val)
            history['ndcg_val'].append(ndcg_val)

        improved = (hr_val > best_val_hr) if select_by == "hr" else (avg_val_loss < best_val_loss)
        if improved:
            best_val_loss = avg_val_loss
            best_val_hr = hr_val
            best_epoch = epoch
            best_hr_val, best_ndcg_val = hr_val, ndcg_val
            best_state = copy.deepcopy(pure_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[PURE NCF] early stop at epoch {epoch} (no improvement for {patience} epochs)")
            stopped_early = True
            break

    if best_state is not None:
        pure_model.load_state_dict(best_state)

    total_time = time.time() - start_total_time
    print(f"PURE NCF best epoch {best_epoch}: "
          f"HR@{top_k}_val={best_hr_val:.4f}, NDCG@{top_k}_val={best_ndcg_val:.4f}")
    print(f"Total time={total_time}")
    print(f"Total_time={time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    if summary is not None:
        summary.update(dict(
            best_epoch=best_epoch, best_val_loss=best_val_loss,
            best_val_hr=best_hr_val, best_val_ndcg=best_ndcg_val,
            epochs_run=len(epoch_times), epochs_max=epochs, stopped_early=stopped_early,
            select_by=select_by, patience=patience, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, total_time_s=total_time, epoch_time_s=epoch_times,
        ))
    return pure_model


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
        seed=42,
        dataset="ml-1m",
        data_root="data",
        amazon_category="Beauty_and_Personal_Care",
        trainer="log",
        neg_exclude="train",
        train_keep_frac=1.0,   #  (эксперимент по разреженности)
        thin_seed=None,
        latent_len=64,
        n_neighbors=10,
        lr_isomap=5e-2,
        lr_ncf=1e-3,
        factor_num=16,
        num_layers=3,
        ncf_weight_decay=1e-6, # weight decay NCF-части GI (только --trainer fast)
        pure_factor_num=None,  # голова pure NCF; None ---> как у GI
        pure_num_layers=None,
        pure_lr=None,          # None ---> lr_ncf
        pure_weight_decay=None,  # None ---> ncf_weight_decay
        cf_epochs=100,
        final_cf_epochs=100,
        batch_size=2048,
        outer_batch_size=1048576,
        inner_patience=40,
        outer_patience=300,
        select_by="loss",
        final_patience=10,
        ncf_patience=None,
        partial_warm_start=False,
        warm_start_inner=False,
        use_item_projection=True,
        use_procrustes_align=True,
        freeze_item_projection=False,
        use_init_assumption=False,
        dist_type="pos",
        save_matrices=False,
        dropout=0.0, # dropout в MLP-башне: и у pure NCF, и у GI
        use_pos_weight=True,
        dedup=True, # False ---> не удалять повторные (user,item) (как в current; на Amazon это утечка таргета)
        legacy_tie_break=False,
        use_ratings=False,
        pure_neg_like_gi=False,
        pure_resample=True, # False ---> pure NCF учится на фиксированных негативах
        gi_resample=False, # True ---> GI-тренер 'log' пересэмплирует негативы каждую эпоху (fast-тренер делает это всегда)
        check_knn_connectivity=False,  # True ---> напечатать число компонент связности kNN-графа начального D
        np_head=None,  # None - выкл. 'bary_cos'|'bary_euclid'|'item_knn' ---> DiffIsomap + непараметрическая голова, БЕЗ NCF
        np_temp=10.0, # фиксированная температура логита: logit = temp*score + bias
        np_bias=0.0, # фиксированный сдвиг логита
        np_knn_k=20, # число соседей в item_knn
        resample_negatives=None,       # None ---> как раньше у каждой «руки»; True/False ---> принудительно для ВСЕХ: pure NCF и GI (любая голова)
        ):

    _cfg = dict(locals())
    for _k, _dv in dict(dropout=0.0, use_pos_weight=True, dedup=True, legacy_tie_break=False, use_ratings=False,
                        pure_neg_like_gi=False, pure_resample=True, gi_resample=False,
                        check_knn_connectivity=False, np_head=None, np_temp=10.0, np_bias=0.0,
                        np_knn_k=20, resample_negatives=None).items():
        if _cfg.get(_k) == _dv:
            _cfg.pop(_k, None)
    run_log = RunLogger(os.path.join("logs_movielens_isomap_cf", str(n_run), "run_log.json"))
    run_log.start(config=_cfg)

    seed_everything(seed)

    if trainer not in ("log", "fast"):
        raise ValueError(f"trainer должен быть 'log' или 'fast', получено {trainer!r}")
    if np_head is not None:
        from nonparam_baselines import HEADS as _NP_HEADS
        if np_head not in _NP_HEADS:
            raise ValueError(f"--np-head должен быть одним из {_NP_HEADS}, получено {np_head!r}")
        if run_gincf:
            print("[np-head] включён режим 'геометрия + непараметрическая голова': NCF-часть GI не обучается "
                  "(--run-gincf не запускает обычный GI); pure NCF (--run-ncf) обучается как обычно")
            run_gincf = False
    if trainer == "fast" and not use_pos_weight and np_head is None:
        raise ValueError("--no-pos-weight не поддерживается для --trainer fast (pos_weight зашит внутри "
                         "GradientIsomapCF_fast.py); для воспроизведения current используйте --trainer log")

    _rs = resample_negatives
    eff_resample = dict(
        pure_ncf=pure_resample if _rs is None else bool(_rs),
        gi_log=gi_resample if _rs is None else bool(_rs),
        gi_fast=True if _rs is None else bool(_rs),
        np_head=True if _rs is None else bool(_rs),
    )
    _arm = "np_head" if np_head is not None else ("gi_fast" if trainer == "fast" else "gi_log")
    print(f"[protocol] пересэмплирование негативов каждую эпоху: pure NCF={eff_resample['pure_ncf']}, "
          f"GI-рука ({_arm})={eff_resample[_arm]}; рейтинги в признаках GI: {use_ratings}")
    run_log.set("protocol", dict(resample_negatives=eff_resample, resample_flag=resample_negatives,
                                 use_ratings=use_ratings, active_gi_arm=_arm))

    ratings_df = load_ratings_df(dataset, data_root, amazon_category)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    print(f"Seed зафиксирован: {seed} (детерминизм от прогона к прогону)")
    print(f"Датасет: {dataset}, trainer: {trainer}")

    import MANUL.Adam.Isomap as _isomap_module
    _isomap_module.device = str(device)

    print("\nПодготовка последовательностей...")
    df_mapped, user2seq = prepare_sequences(ratings_df, build_user2seq=False, dedup=dedup)

    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped,
        max_users=max_users,
        max_movies=max_movies,
        min_seq_len=min_seq_len,
        legacy_tie_break=legacy_tie_break,
    )

    user_pos_set = {
        u: set(m for (m, ts, r) in seq)
        for u, seq in user2seq_sub.items()
    }

    print("\nTrain/Val/Test split (next-item)...")
    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)

    # Множества для ОЦЕНКИ строятся по ПОЛНОМУ train (до прореживания): кандидаты val/test и исключаемые
    # из них айтемы не должны зависеть от train_keep_frac, иначе метрики на разной плотности несравнимы.
    user_pos_train_set_full = {u: set() for u in range(num_users)}
    for (u, m) in train_events[:, :2].tolist():
        user_pos_train_set_full[int(u)].add(int(m))

    user_hist_val_set = user_pos_train_set_full

    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set_full.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    train_events_full = train_events
    if train_keep_frac < 1.0:
        _thin_seed = seed if thin_seed is None else thin_seed
        train_events = thin_train_events(train_events_full, train_keep_frac, seed=_thin_seed)
        print(f"[thin] train_keep_frac={train_keep_frac}, thin_seed={_thin_seed}: "
              f"train-событий {len(train_events_full)} ---> {len(train_events)} "
              f"(val/test не затронуты, кандидаты оценки те же)")
    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m) in train_events[:, :2].tolist():
        user_pos_train_set[int(u)].add(int(m))

    train_pairs = train_events[:, :2].astype(np.int64)

    if neg_exclude == "train":
        neg_set = user_pos_train_set # протокол He et al.: негативы не знают о val/test
    elif neg_exclude == "full":
        neg_set = user_pos_set #утечка будущих позитивов в обучение GI
    else:
        raise ValueError(f"neg_exclude must be 'train' or 'full', got {neg_exclude!r}")
    print(f"neg_exclude = {neg_exclude} (множество, по которому исключаются тренировочные негативы GI)")

    # по умолчанию pure NCF исключает негативы по train (как раньше). pure_neg_like_gi=True -> по тому же
    # множеству neg_set, что и GI: тогда оба арма видят ОДНИ И ТЕ ЖЕ негативы (при pure_resample=False).
    train_dataset = NCFTrainDatasetFutureBlind(
        features_pos=train_pairs,
        num_items=num_movies,
        user_pos_train_set=(neg_set if pure_neg_like_gi else user_pos_train_set),
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

    # Тензоры оценки (T x (1+99)) для векторизованных метрик pure NCF и fast-тренера.
    val_tensors = eval_tensors(val_dataset, device)
    test_tensors = eval_tensors(test_dataset, device)

    # --trainer log принимает те же тензоры (T, 1+num_ng); PAD-слоты в них игнорируются.
    val_loader = val_tensors
    test_loader = test_tensors

    print(f"\n num_users={num_users}, num_movies={num_movies}, "
          f"train_pairs={len(train_pairs)}, val_users={len(val_next)}, test_users={len(test_next)}")
    run_log.set("data", dict(
        dataset=dataset, ratings_total=int(len(ratings_df)),
        num_users=int(num_users), num_items=int(num_movies),
        train_positives=int(len(train_pairs)), val_users=int(len(val_next)), test_users=int(len(test_next)),
        train_density=float(len(train_pairs)) / max(1, int(num_users) * int(num_movies)),
        train_keep_frac=float(train_keep_frac), thin_seed=(seed if thin_seed is None else thin_seed),
        train_positives_full=int(len(train_events_full)),
        train_density_full=float(len(train_events_full)) / max(1, int(num_users) * int(num_movies)),
        corater_stats=corater_stats(train_events, int(num_movies), int(num_users)),
        batch_size=int(batch_size), interactions_per_epoch=int(len(train_pairs) * (1 + num_ng)),
        steps_per_epoch=-(-int(len(train_pairs) * (1 + num_ng)) // int(batch_size)),
        neg_exclude=neg_exclude, num_ng_train=int(num_ng), eval_candidates_per_user=100,
        val_pad_slots=int((val_dataset.labels_fill < 0).sum()), test_pad_slots=int((test_dataset.labels_fill < 0).sum()),
    ))

    print(f"[steps] interactions/epoch = {len(train_pairs)} pos x (1+{num_ng}) = {len(train_pairs) * (1 + num_ng)}, "
          f"batch_size={batch_size} -> {-(-len(train_pairs) * (1 + num_ng) // batch_size)} gradient steps per epoch")

    # Контрольные уровни на тех же кандидатах (и на том же, возможно прореженном, train): случайный и "по популярности".
    # Нужны, чтобы сравнивать выигрыш НАД популярностью, а не сырой HR (задачи разной сложности несравнимы по сырому HR).
    baselines = {}
    for _split, _t in (("val", val_tensors), ("test", test_tensors)):
        baselines[_split] = popularity_baseline(_t[1].cpu().numpy(), _t[2].cpu().numpy(),
                                                train_events, int(num_movies), top_k)
    run_log.set("baselines", baselines)
    _b = baselines["test"]
    print(f"[baseline] test HR@{top_k}: random={_b['random_hr']:.4f}, popularity={_b['popularity_hr']:.4f} "
          f"(границы по ничьим: {_b['popularity_hr_pessimistic']:.4f} .. {_b['popularity_hr_optimistic']:.4f})")

    if run_ncf:
        _pf = pure_factor_num if pure_factor_num is not None else factor_num
        _pl = pure_num_layers if pure_num_layers is not None else num_layers
        _plr = pure_lr if pure_lr is not None else lr_ncf
        _pwd = pure_weight_decay if pure_weight_decay is not None else ncf_weight_decay
        pure_model = NCF(
            user_num=num_users,
            item_num=num_movies,
            factor_num=_pf,
            num_layers=_pl,
            dropout=dropout,
            model='NeuMF-end'
        ).to(device)
        print(f"[pure NCF] factor_num={_pf}, num_layers={_pl}, lr={_plr}, weight_decay={_pwd}, "
              f"trainable params={sum(p.numel() for p in pure_model.parameters())}")

        pure_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'hr_val': [],
            'ndcg_val': []
        }

        pure_summary = {}
        train_pure_ncf(
            pure_model,
            train_dataset,
            num_ng=num_ng,
            top_k=top_k,
            device=device,
            val_tensors=val_tensors,
            epochs=epochs_pure,
            batch_size=batch_size,
            lr=_plr,
            weight_decay=_pwd,
            history=pure_history,
            select_by=select_by,
            patience=ncf_patience if ncf_patience is not None else final_patience,
            summary=pure_summary,
            use_pos_weight=use_pos_weight,
            resample_negatives=eff_resample["pure_ncf"],
        )

        score_fn = lambda u, i: pure_model(u, i)
        hr_test_pure, ndcg_test_pure = evaluate_topk_vec(
            score_fn, *test_tensors, top_k, device)
        print(f"PURE NCF final on TEST: HR@{top_k}={hr_test_pure:.4f}, NDCG@{top_k}={ndcg_test_pure:.4f}")
        run_log.set("pure_ncf", dict(
            model=dict(factor_num=_pf, num_layers=_pl, dropout=dropout, model='NeuMF-end',
                       params=int(sum(p.numel() for p in pure_model.parameters()))),
            val=dict(top_k=top_k, hr=pure_summary.get("best_val_hr"), ndcg=pure_summary.get("best_val_ndcg")),
            summary=pure_summary, history=pure_history,
            test=dict(top_k=top_k, hr=hr_test_pure, ndcg=ndcg_test_pure),
        ))

        images_dir = os.path.join("logs_pure_ncf", "images/new_val")
        os.makedirs(images_dir, exist_ok=True)
        plot_pure_ncf_losses(pure_history, images_dir, run_name=f"pure_ncf_new_{n_run}_focal_losses")

    if np_head is not None:
        print(f"\n=== DiffIsomap + непараметрическая голова '{np_head}' (NCF не обучается) ===")
        seed_everything(seed)
        from geometry_baseline_trainer import GeometryBaselineTrainer
        _mum = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=use_ratings)
        geo_trainer = GeometryBaselineTrainer(
            train_feature=torch.tensor(_mum, dtype=torch.float32, device=device),
            train_events=train_events, num_users=num_users, num_items=num_movies,
            user_pos_set=neg_set, ng_seed=42 + seed, latent_len=latent_len, n_neighbors=n_neighbors,
            epochs=gradisomap_epochs, lr_isomap=lr_isomap, num_ng=num_ng,
            head=np_head, temp=np_temp, bias=np_bias, knn_k=np_knn_k, use_pos_weight=use_pos_weight,
            patience=outer_patience, row_chunk=outer_batch_size, device=str(device), seed=seed,
            resample_negatives=eff_resample["np_head"])
        _geo_model, _geo_Z, geo_res = geo_trainer.train(
            val_tensors, test_tensors, top_k=top_k, use_init_assumption=use_init_assumption, dist_type=dist_type)
        _ti, _tg = geo_res["init_geometry"]["test"], geo_res["trained_geometry"]["test"]
        print(f"[geom-baseline:{np_head}] TEST HR@{top_k}: начальная геометрия={_ti['hr']:.4f}, "
              f"обученная (эпоха {geo_res['best_epoch']})={_tg['hr']:.4f}, popularity={baselines['test']['popularity_hr']:.4f}")
        run_log.set("geometry_baseline", dict(top_k=top_k, **geo_res, history=geo_trainer.history))

        from geometry_baseline_trainer import plot_geometry_baseline
        _gdir = os.path.join("logs_movielens_isomap_cf", f"{n_run}", "images")
        os.makedirs(_gdir, exist_ok=True)
        with open(os.path.join(_gdir, "geometry_history.json"), "w", encoding="utf-8") as _f:
            json.dump(geo_trainer.history, _f, ensure_ascii=False, indent=4)
        plot_geometry_baseline(geo_trainer.history, geo_res, os.path.join(_gdir, "geometry_baseline.png"), top_k=top_k)
        print(f"[geom-baseline:{np_head}] график: {os.path.join(_gdir, 'geometry_baseline.png')}; "
              f"история: {os.path.join(_gdir, 'geometry_history.json')}")

    if run_gincf:
        print("\n=== Обучение GradientIsomapCF (IsomapNN + NeuMFOnManifold) ===")
        seed_everything(seed)
        t_gi_start = time.time()

        movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=use_ratings)
        features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

        logs_folder = f"logs_movielens_isomap_cf/{n_run}"

        if check_knn_connectivity:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            with torch.no_grad():
                _D = torch.cdist(features_t, features_t)
                _D = _D / _D.max().clamp(min=1e-12)
                _nn = torch.topk(_D, n_neighbors, dim=1, largest=False).indices.cpu().numpy()
            _rows = np.repeat(np.arange(_nn.shape[0]), _nn.shape[1])
            _g = coo_matrix((np.ones(_rows.size), (_rows, _nn.reshape(-1))), shape=(_nn.shape[0],) * 2)
            _ncomp, _lab = connected_components(_g, directed=False)
            print(f"[check] kNN-граф начального D (k={n_neighbors}): компонент связности = {_ncomp}, "
                  f"размеры: {sorted(np.bincount(_lab).tolist(), reverse=True)[:8]}")

        if trainer == "fast":
            from GradientIsomapCF_fast import GradientIsomapCF as GradientIsomapCFFast

            gi_cf = GradientIsomapCFFast(
                train_feature=features_t,
                train_events=train_events,
                num_users=num_users,
                num_items=num_movies,
                user_pos_set=neg_set,
                ng_seed=42,
                latent_len=latent_len,
                n_neighbors=n_neighbors,
                epochs=gradisomap_epochs,
                cf_epochs=cf_epochs,
                final_cf_epochs=final_cf_epochs,
                batch_size=batch_size,
                outer_batch_size=outer_batch_size,
                lr_isomap=lr_isomap,
                lr_ncf=lr_ncf,
                factor_num=factor_num,
                num_layers=num_layers,
                dropout=0.2, #dropout
                model_type='NeuMF-end',
                logs_folder=logs_folder,
                device=str(device),
                stop_criteria_value=0.001,
                num_ng=num_ng,
                partial_warm_start=partial_warm_start,
                warm_start_inner=warm_start_inner,
                inner_patience=inner_patience,
                outer_patience=outer_patience,
                use_item_projection=use_item_projection,
                use_procrustes_align=use_procrustes_align,
                freeze_item_projection=freeze_item_projection,
                select_by=select_by,
                final_patience=final_patience,
                save_matrices=save_matrices,
                ncf_weight_decay=ncf_weight_decay,
                resample_negatives=eff_resample["gi_fast"],
            )
            isomap_model, ncf_manifold_model, item_Z_final = gi_cf.train(
                val_tensors=val_tensors,
                top_k=top_k,
                device=device,
                use_init_assumption=use_init_assumption,
                dist_type=dist_type,
            )

            score_fn_iso = lambda u, i: ncf_manifold_model(u, i, item_Z_final)
            hr_iso, ndcg_iso = evaluate_topk_vec(score_fn_iso, *test_tensors, top_k, device)
            hr_val_iso, ndcg_val_iso = evaluate_topk_vec(score_fn_iso, *val_tensors, top_k, device)
        else:
            from GradientIsomapCF_log import GradientIsomapCF
            gi_cf = GradientIsomapCF(
                train_feature=features_t,
                train_events=train_events,
                num_users=num_users,
                num_items=num_movies,
                user_pos_set=neg_set,
                ng_seed=42,
                latent_len=latent_len,  # размерность manifold Z
                n_neighbors=n_neighbors,
                epochs=gradisomap_epochs,  # outer_epochs
                cf_epochs=cf_epochs,  # внутренних эпох CF на фиксированном Z
                final_cf_epochs=final_cf_epochs,  # количество эпох для финальной модели
                batch_size=batch_size,
                lr_isomap=lr_isomap,
                lr_ncf=lr_ncf,
                factor_num=factor_num,
                num_layers=num_layers,
                dropout=dropout,
                model_type='NeuMF-end',
                logs_folder=logs_folder,
                device=str(device),
                stop_criteria_value=0.001,
                num_ng=num_ng,
                # раньше эти параметры в 'log'-тренер НЕ передавались (select_by='loss', inner_patience=5,
                # freeze_item_projection=False, независимо от CLI) - а у pure NCF select_by из CLI применялся:
                select_by=select_by,
                final_patience=final_patience,
                inner_patience=inner_patience,
                warm_start_inner=warm_start_inner,
                freeze_item_projection=freeze_item_projection,
                resample_negatives=eff_resample["gi_log"],
            )

            # pos_weight=num_ng: тот же взвешенный BCE, что у pure NCF (use_pos_weight=False ---> plain BCE у обоих)
            pos_weight = (torch.tensor([num_ng], device=device, dtype=torch.float32)
                          if use_pos_weight else None)
            isomap_model, ncf_manifold_model = gi_cf.train(
                val_loader=val_loader,
                top_k=top_k,
                device=device,
                pos_weight=pos_weight,
            )

            hr_iso, ndcg_iso = evaluate_topk_isomap(ncf_manifold_model, isomap_model, test_loader, top_k, device)

        print(f"GradientIsomapCF final on TEST: HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")
        print(f"GI + NeuMFOnM HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}")
        run_log.set("gi", dict(
            trainer=trainer, logs_folder=logs_folder,
            total_time_s=time.time() - t_gi_start,
            test=dict(top_k=top_k, hr=hr_iso, ndcg=ndcg_iso),
            val=(dict(top_k=top_k, hr=hr_val_iso, ndcg=ndcg_val_iso) if trainer == "fast" else None),
            model=dict(factor_num=factor_num, num_layers=num_layers, latent_len=latent_len, n_neighbors=n_neighbors,
                       params_ncf_part=int(sum(p.numel() for p in ncf_manifold_model.parameters())) if trainer == "fast" else None,
                       params_distance_matrix=int(num_movies * (num_movies - 1) // 2)),
            steps_per_epoch=getattr(gi_cf, "steps_per_epoch", None),
            spectrum=getattr(gi_cf, "spectrum_history", None),
            spectrum_full=getattr(gi_cf, "spectrum_full", None),
            history=gi_cf.history,
            cf_history=gi_cf.cf_history,
        ))

        images_dir = os.path.join("logs_movielens_isomap_cf", f"{n_run}/images")
        plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")

        os.makedirs(os.path.join("logs_movielens_isomap_cf", f"{n_run}", "images"), exist_ok=True)
        with open(f"logs_movielens_isomap_cf/{n_run}/images/history.json", 'w', encoding='utf-8') as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        images_dir_png = os.path.join("logs_movielens_isomap_cf", f"{n_run}/images/metrics.png")
        plot_gi_convergence(gi_cf.history, top_k=top_k, save_path=images_dir_png)

    summ = dict(top_k=top_k)
    if run_ncf:
        summ["pure_ncf_test"] = dict(hr=hr_test_pure, ndcg=ndcg_test_pure)
    if run_gincf:
        summ["gi_test"] = dict(hr=hr_iso, ndcg=ndcg_iso)
    if run_ncf and run_gincf:
        summ["gi_minus_pure"] = dict(hr=hr_iso - hr_test_pure, ndcg=ndcg_iso - ndcg_test_pure)
    summ["popularity_hr_test"] = baselines["test"]["popularity_hr"]
    summ["random_hr_test"] = baselines["test"]["random_hr"]
    if run_ncf:
        summ["pure_ncf_gain_over_popularity"] = hr_test_pure - baselines["test"]["popularity_hr"]
    if run_gincf:
        summ["gi_gain_over_popularity"] = hr_iso - baselines["test"]["popularity_hr"]
    if np_head is not None:
        _g = geo_res["trained_geometry"]["test"]
        summ["geometry_baseline_test"] = dict(head=np_head, hr=_g["hr"], ndcg=_g["ndcg"],
                                              init_geometry_hr=geo_res["init_geometry"]["test"]["hr"])
        summ["geometry_baseline_gain_over_popularity"] = _g["hr"] - baselines["test"]["popularity_hr"]
        if run_ncf:
            summ["geometry_baseline_minus_pure"] = dict(hr=_g["hr"] - hr_test_pure, ndcg=_g["ndcg"] - ndcg_test_pure)
    run_log.set("summary", summ, save=False)
    run_log.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GradIsomapCF (DiffIsomap + NCF): ml-1m / ml-10m / Amazon, "
                    "тренер log (current) или fast (патч 2), детерминизм по --seed")
    parser.add_argument("--dataset", choices=["ml-1m", "ml-10m", "amazon"], default="ml-1m",
                        help="датасет; файлы данных кладутся в data/<имя из README>")
    parser.add_argument("--amazon-category", default="Beauty_and_Personal_Care",
                        help="категория Amazon (имя файла без расширения: "
                             "<category>.csv для Reviews 2023 или <category>_5.json "
                             "для классического 5-core)")
    parser.add_argument("--data-root", default="data",
                        help="корневая папка с данными (по умолчанию ./data)")
    parser.add_argument("--trainer", choices=["log", "fast"], default="fast",
                        help="log = нетронутый GradientIsomapCF_log"
                             "fast = bilevel-тренер из ветки fast")
    parser.add_argument("--train-keep-frac", type=float, default=1.0,
                        help="доля train-взаимодействий каждого пользователя, которую оставить (0<f<=1); "
                             "val/test и кандидаты оценки не меняются. Для экспериментов по разреженности")
    parser.add_argument("--thin-seed", type=int, default=None,
                        help="seed прореживания (по умолчанию = --seed, т.е. каждый сид даёт свою "
                             "реализацию прореженной матрицы, одинаковую у pure NCF и GI)")
    parser.add_argument("--neg-exclude", choices=["train", "full"], default="train",
                        help="какое множество исключать при сэмплировании ТРЕНИРОВОЧНЫХ негативов GI: "
                             "train = только train-позитивы (чистый протокол, по умолчанию); "
                             "full = вся история включая val/test (утечка)")
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--max-movies", type=int, default=800)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--num-ng", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs-pure", type=int, default=100)
    parser.add_argument("--gradisomap-epochs", type=int, default=30)
    parser.add_argument("--run-ncf", dest="run_ncf", action="store_true", default=True,
                        help="обучать pure NCF базлайн (по умолчанию включён; "
                             "отключается флагом --no-ncf)")
    parser.add_argument("--no-ncf", dest="run_ncf", action="store_false",
                        help="не обучать pure NCF базлайн")
    parser.add_argument("--run-gincf", dest="run_gincf", action="store_true", default=False,
                        help="обучать GradientIsomapCF (по умолчанию выключен)")
    parser.add_argument("--n-run", type=int, default=68)
    parser.add_argument("--seed", type=int, default=42,
                        help="глобальный сид детерминизма")
    parser.add_argument("--latent-len", type=int, default=64)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--lr-isomap", type=float, default=5e-2)
    parser.add_argument("--lr-ncf", type=float, default=1e-3)
    parser.add_argument("--factor-num", type=int, default=16, help="голова NeuMF у GI (и у pure NCF, если не задан --pure-factor-num)")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--wd-ncf", type=float, default=1e-6, help="weight decay NCF-части GI (fast) и pure NCF, если не задан --pure-wd")
    parser.add_argument("--pure-factor-num", type=int, default=None)
    parser.add_argument("--pure-num-layers", type=int, default=None)
    parser.add_argument("--pure-lr", type=float, default=None)
    parser.add_argument("--pure-wd", type=float, default=None)
    parser.add_argument("--cf-epochs", type=int, default=100)
    parser.add_argument("--final-cf-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="батч и для pure NCF, и для GI (внутренние/финальная стадии)")
    # специфичное для --trainer fast
    parser.add_argument("--outer-batch-size", type=int, default=1048576,
                        help="размер чанка внешнего градиента (только fast)")
    parser.add_argument("--inner-patience", type=int, default=40)
    parser.add_argument("--outer-patience", type=int, default=300)
    parser.add_argument("--select-by", choices=["loss", "hr"], default="loss",
                        help="критерий отбора эпохи/состояния - ОДИН для pure NCF и GI")
    parser.add_argument("--final-patience", type=int, default=10,
                        help="терпение финальной стадии GI; и pure NCF, если не задан --ncf-patience")
    parser.add_argument("--ncf-patience", type=int, default=None)
    parser.add_argument("--partial-warm-start", action="store_true", default=False)
    parser.add_argument("--warm-start-inner", action="store_true", default=False)
    parser.add_argument("--no-item-projection", dest="use_item_projection",
                        action="store_false", default=True,
                        help="вместо обучаемых проекций брать координаты Z напрямую "
                             "(требует --latent-len >= max(factor_num, mlp_user_dim))")
    parser.add_argument("--no-procrustes", dest="use_procrustes_align",
                        action="store_false", default=True)
    parser.add_argument("--freeze-item-projection", action="store_true", default=False,
                        help="frozen PCA-режим item-проекций (из current; "
                             "требует --latent-len == factor_num*2**(num_layers-1))")
    parser.add_argument("--use-init-assumption", action="store_true", default=False)
    parser.add_argument("--dist-type", choices=["pos", "uniform", "normal", "exp"],
                        default="pos", help="схема случайной матрицы D (только fast)")
    parser.add_argument("--save-matrices", action="store_true", default=False,
                        help="сохранять матрицы каждой внешней эпохи (только fast)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout в MLP-башне - и у pure NCF, и у GI")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false", default=True,
                        help="обычный BCE вместо pos_weight=num_ng (pure NCF и --trainer log)")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", default=True,
                        help="НЕ удалять повторные (user,item) - поведение current (на Amazon: таргет может лежать в train)")
    parser.add_argument("--legacy-tie-break", action="store_true", default=False,
                        help="старый отбор top-k пользователей/айтемов при равных counts")
    parser.add_argument("--use-ratings", action=argparse.BooleanOptionalAction, default=False,
                        help="--use-ratings: признаки айтема = средний рейтинг; --no-use-ratings / без флага - "
                             "счётчики. Влияет на ВСЕ GI-модели (NCF-голова и --np-head: из этой матрицы строится начальное D); "
                             "у pure NCF рейтингов нет (он учится на 0/1)")
    parser.add_argument("--resample-negatives", action=argparse.BooleanOptionalAction, default=None,
                        help="единый переключатель пересэмплирования негативов каждую эпоху для pure NCF И для GI (любая голова: "
                             "NCF 'fast'/'log' или --np-head). Не задан - поведение по умолчанию каждой ветки (pure: да, GI fast: да, "
                             "GI log: нет, --np-head: да); --no-resample-negatives - у всех фиксированный "
                             "набор негативов (один сэмпл при старте); --resample-negatives - у всех свежие каждую эпоху")
    parser.add_argument("--pure-neg-like-gi", action="store_true", default=True,
                        help="негативы pure NCF исключать по тому же множеству, что у GI (--neg-exclude)")
    parser.add_argument("--no-pure-resample", dest="pure_resample", action="store_false", default=False,
                        help="pure NCF на ФИКСИРОВАННЫХ негативах (как Euclid)")
    parser.add_argument("--gi-resample", action="store_true", default=False,
                        help="--trainer log: пересэмплировать негативы каждую эпоху (как pure NCF по умолчанию)")
    parser.add_argument("--check-knn-connectivity", action="store_true", default=False,
                        help="напечатать число компонент связности kNN-графа начального D (Isomap-патч fast)")
    parser.add_argument("--np-head", choices=["bary_cos", "bary_euclid", "item_knn"], default=None,
                        help="DiffIsomap + непараметрическая голова вместо NCF: n эпох (--gradisomap-epochs), "
                             "на каждой Z=MDS(D) -> скор -> BCE -> шаг по D (--lr-isomap); NCF не обучается; "
                             "выбор лучшей эпохи по val HR. Результат - секция geometry_baseline в run_log.json")
    parser.add_argument("--np-temp", type=float, default=10.0,
                        help="фиксированная температура логита у --np-head (не обучается)")
    parser.add_argument("--np-bias", type=float, default=0.0,
                        help="фиксированный сдвиг логита у --np-head (не обучается)")
    parser.add_argument("--np-knn-k", type=int, default=20,
                        help="число ближайших соседей в item_knn (--np-head item_knn)")
    args = parser.parse_args()

    main(
        max_users=args.max_users,
        max_movies=args.max_movies,
        min_seq_len=args.min_seq_len,
        num_ng=args.num_ng,
        top_k=args.top_k,
        epochs_pure=args.epochs_pure,
        gradisomap_epochs=args.gradisomap_epochs,
        run_ncf=args.run_ncf,
        run_gincf=args.run_gincf,
        n_run=args.n_run,
        seed=args.seed,
        dataset=args.dataset,
        data_root=args.data_root,
        amazon_category=args.amazon_category,
        trainer=args.trainer,
        neg_exclude=args.neg_exclude,
        train_keep_frac=args.train_keep_frac,
        thin_seed=args.thin_seed,
        latent_len=args.latent_len,
        n_neighbors=args.n_neighbors,
        lr_isomap=args.lr_isomap,
        lr_ncf=args.lr_ncf,
        factor_num=args.factor_num,
        num_layers=args.num_layers,
        ncf_weight_decay=args.wd_ncf,
        pure_factor_num=args.pure_factor_num,
        pure_num_layers=args.pure_num_layers,
        pure_lr=args.pure_lr,
        pure_weight_decay=args.pure_wd,
        cf_epochs=args.cf_epochs,
        final_cf_epochs=args.final_cf_epochs,
        batch_size=args.batch_size,
        outer_batch_size=args.outer_batch_size,
        inner_patience=args.inner_patience,
        outer_patience=args.outer_patience,
        select_by=args.select_by,
        final_patience=args.final_patience,
        ncf_patience=args.ncf_patience,
        partial_warm_start=args.partial_warm_start,
        warm_start_inner=args.warm_start_inner,
        use_item_projection=args.use_item_projection,
        use_procrustes_align=args.use_procrustes_align,
        freeze_item_projection=args.freeze_item_projection,
        use_init_assumption=args.use_init_assumption,
        dist_type=args.dist_type,
        save_matrices=args.save_matrices,
        dropout=args.dropout,
        use_pos_weight=args.use_pos_weight,
        dedup=args.dedup,
        legacy_tie_break=args.legacy_tie_break,
        use_ratings=args.use_ratings,
        pure_neg_like_gi=args.pure_neg_like_gi,
        pure_resample=args.pure_resample,
        gi_resample=args.gi_resample,
        resample_negatives=args.resample_negatives,
        check_knn_connectivity=args.check_knn_connectivity,
        np_head=args.np_head,
        np_temp=args.np_temp,
        np_bias=args.np_bias,
        np_knn_k=args.np_knn_k,
    )