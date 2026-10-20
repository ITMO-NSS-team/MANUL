import os
import sys
import copy
import json
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, GINCF_DIR)

from prepare_data import prepare_sequences, subsample_users_items, train_val_test_split_next_item

from geoopt import PoincareBallExact
from run_classic_sasrec2 import SASRec 


# ---------------------------------------------------------------------
# HypSASRec
# ---------------------------------------------------------------------
class HypSASRec(SASRec):
    def __init__(self, config, item_num):
        super().__init__(config, item_num)

        self.geom = config["geom"]
        self.c = torch.tensor(float(config["c"]), dtype=torch.float32)

        if self.geom == "ball":
            self.manifold = PoincareBallExact(c=float(config["c"]))
            self.scaler = nn.Parameter(
                torch.zeros(self.item_emb.num_embeddings, dtype=torch.float32),
                requires_grad=bool(config["bias"]),
            )
        else:
            raise NotImplementedError

    def forward(self, log_seqs):
        log_feats = self.log2feats(log_seqs)            # [B, L, H]
        hyp_feats = self.manifold.expmap0(log_feats)    # [B, L, H] in ball
        logits = self.poincare_hyperplane(hyp_feats)    # [B, L, N]
        return logits

    def poincare_hyperplane(self, hyp_feats):
        weight = self.item_emb.weight                  # [N, H]
        w_norm = weight.norm(dim=-1)                   # [N]
        w_unit = weight / w_norm.unsqueeze(-1)         # [N, H]
        return unidirectional_poincare_mlr(hyp_feats, w_norm, w_unit, self.scaler, self.c)


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    rc = c.sqrt()
    drcr = 2.0 * rc * r

    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)
    denom_safe = torch.clamp_min(1.0 - cx2, 1e-15)

    return 2 * z_norm / rc * torch.arcsinh(
        (2.0 * torch.matmul(rcx, z_unit.T) * drcr.cosh() - (1.0 + cx2) * drcr.sinh())
        / denom_safe
    )


# ---------------------------------------------------------------------
# Загрузка данных: Amazon / MovieLens → общий формат
# ---------------------------------------------------------------------
def load_movielens_ratings(movielens_dir: str) -> pd.DataFrame:
    """
    Поддержка MovieLens:
    - ratings.dat (ml-1m/ml-10m классический формат UserID::MovieID::Rating::Timestamp)
    - или ratings.csv (на случай других сборок)
    """
    dat_path = os.path.join(movielens_dir, "ratings.dat")
    csv_path = os.path.join(movielens_dir, "ratings.csv")

    if os.path.exists(dat_path):
        print(f"Loading MovieLens ratings from {dat_path} ...")
        df = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
        )
    elif os.path.exists(csv_path):
        print(f"Loading MovieLens ratings from {csv_path} ...")
        df = pd.read_csv(csv_path)
        # пытаемся привести к стандартным именам
        rename = {}
        if "user_id" in df.columns: rename["user_id"] = "userId"
        if "movie_id" in df.columns: rename["movie_id"] = "movieId"
        if "itemId" in df.columns: rename["itemId"] = "movieId"
        df = df.rename(columns=rename)
        df = df[["userId", "movieId", "rating", "timestamp"]]
    else:
        raise FileNotFoundError(f"MovieLens ratings not found in {movielens_dir}. "
                                f"Expected ratings.dat or ratings.csv")

    print("Unique_users:", df["userId"].nunique(), "Unique_items:", df["movieId"].nunique())
    print(f"Loaded {len(df)} ratings")
    return df


def load_amazon_ratings(dataset_dir: str, category: str) -> pd.DataFrame:
    ratings_path = os.path.join(dataset_dir, f"{category}.csv")
    print(f"Loading Amazon ratings from {ratings_path} ...")
    df = pd.read_csv(ratings_path)

    # приводим к стандарту prepare_sequences
    df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
    df = df[["userId", "movieId", "rating", "timestamp"]]
    return df


def load_data_generic(
    dataset_type: str,
    dataset_dir_name: str,
    amazon_category: str,
    max_users: int,
    max_movies: int,
    min_seq_len: int = 5,
):
    """
    Возвращает ровно то же, что твой load_amazon_data:
    train_seqs, val_targets, test_targets, num_users, num_items, user_pos_set
    """
    dataset_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)

    if dataset_type == "amazon":
        df = load_amazon_ratings(dataset_dir, amazon_category)
    elif dataset_type == "movielens":
        df = load_movielens_ratings(dataset_dir)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    df_mapped, user2seq = prepare_sequences(df)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies, min_seq_len=min_seq_len
    )

    print("\nTrain/Val/Test split...")
    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)

    user_train = defaultdict(list)
    for (u, m, r) in train_events:
        user_train[int(u)].append(int(m))
    train_seqs = dict(user_train)

    val_targets = {int(u): int(target) for (u, _, target) in val_next}
    test_targets = {int(u): int(target) for (u, _, target) in test_next}

    user_pos_set = {u: set(seq) for u, seq in train_seqs.items()}
    for u, target in val_targets.items():
        user_pos_set[u].add(target)

    print(f"Prepared: users={num_users} items={num_movies} "
          f"train_users={len(train_seqs)} val_users={len(val_targets)} test_users={len(test_targets)}")

    return train_seqs, val_targets, test_targets, num_users, num_movies, user_pos_set


# ---------------------------------------------------------------------
# Datasets / samplers 
# ---------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(self, user_seqs, num_items, maxlen):
        self.user_seqs = {u: seq for u, seq in user_seqs.items() if len(seq) >= 2}
        self.users = list(self.user_seqs.keys())
        self.num_items = num_items
        self.maxlen = maxlen
        self.pad_token = num_items

    def __len__(self):
        return len(self.users) * 10

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        u = self.users[idx % len(self.users)]
        seq_items = self.user_seqs[u]

        seq = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        pos = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        neg = np.full(self.maxlen, self.pad_token, dtype=np.int32)

        nxt = seq_items[-1]
        idx_pos = self.maxlen - 1
        seen_set = set(seq_items)

        for item in reversed(seq_items[:-1]):
            seq[idx_pos] = item
            pos[idx_pos] = nxt

            neg_item = rng.randint(self.num_items)
            while neg_item in seen_set:
                neg_item = rng.randint(self.num_items)
            neg[idx_pos] = neg_item

            nxt = item
            idx_pos -= 1
            if idx_pos == -1:
                break

        return torch.tensor(seq), torch.tensor(pos), torch.tensor(neg)


class TestDataset(Dataset):
    def __init__(self, user_seqs, targets, num_items, maxlen, user_pos_set, num_neg=99):
        self.data = []
        self.pad_token = num_items
        rng = np.random.default_rng(42)

        for u, target_item in targets.items():
            if u not in user_seqs:
                continue
            seq = user_seqs[u]
            if len(seq) >= maxlen:
                padded_seq = seq[-maxlen:]
            else:
                padded_seq = [self.pad_token] * (maxlen - len(seq)) + seq

            candidates = [target_item]
            pos_set = user_pos_set.get(u, set())

            attempts = 0
            while len(candidates) < num_neg + 1 and attempts < 2000:
                neg_item = int(rng.integers(0, num_items))
                if neg_item not in pos_set and neg_item != target_item:
                    candidates.append(neg_item)
                attempts += 1

            labels = [1.0] + [0.0] * (len(candidates) - 1)
            self.data.append((padded_seq, candidates, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, cands, labs = self.data[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(cands, dtype=torch.long),
            torch.tensor(labs, dtype=torch.float32),
        )


# ---------------------------------------------------------------------
# Eval 
# ---------------------------------------------------------------------
def hyp_candidate_logits(model: HypSASRec, seq: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    log_feats = model.log2feats(seq)
    hyp_last = model.manifold.expmap0(log_feats[:, -1, :])
    logits_all = model.poincare_hyperplane(hyp_last)
    return logits_all.gather(1, cands)


def evaluate_loss_hyp(model, loader, device, criterion_bce):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for seq, cands, labels in loader:
            seq = seq.to(device)
            cands = cands.to(device)
            labels = labels.to(device)
            logits = hyp_candidate_logits(model, seq, cands)
            loss = criterion_bce(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate_model_hyp(model, loader, device, top_k=10):
    model.eval()
    HR, NDCG = [], []
    with torch.no_grad():
        for seq, cands, labels in loader:
            seq = seq.to(device)
            cands = cands.to(device)

            logits = hyp_candidate_logits(model, seq, cands)
            pos_score = logits[:, 0]
            rank = (logits > pos_score.unsqueeze(1)).sum(dim=1)

            hit = (rank < top_k).float()
            HR.extend(hit.cpu().numpy().tolist())

            ndcg = torch.where(
                rank < top_k,
                1.0 / torch.log2(rank.float() + 2.0),
                torch.zeros_like(rank, dtype=torch.float32),
            )
            NDCG.extend(ndcg.cpu().numpy().tolist())

    return float(np.mean(HR)), float(np.mean(NDCG))


# ---------------------------------------------------------------------
# MAIN: обучение HypSASRec (Amazon / MovieLens)
# ---------------------------------------------------------------------
def run_one(
    dataset_type: str,
    dataset_dir_name: str,
    amazon_category: str,
    history_path: str,
    save_dir: str,
    max_users: int,
    max_movies: int,
    seed: int = 0,
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CONFIG = {
        "hidden_units": 64,
        "maxlen": 50,
        "num_blocks": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 50,
        "patience": 5,
        "geom": "ball",
        "c": 1.0,
        "bias": True,
    }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_seqs, val_targets, test_targets, num_users, num_items, user_pos_set = load_data_generic(
        dataset_type=dataset_type,
        dataset_dir_name=dataset_dir_name,
        amazon_category=amazon_category,
        max_users=max_users,
        max_movies=max_movies,
        min_seq_len=5,
    )

    train_dataset = TrainDataset(train_seqs, num_items, CONFIG["maxlen"])
    val_dataset = TestDataset(train_seqs, val_targets, num_items, CONFIG["maxlen"], user_pos_set)

    test_seqs_for_eval = {u: seq + [val_targets[u]] for u, seq in train_seqs.items() if u in val_targets}
    test_dataset = TestDataset(test_seqs_for_eval, test_targets, num_items, CONFIG["maxlen"], user_pos_set)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = HypSASRec(CONFIG, num_items).to(DEVICE)
    model.c = model.c.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.98))

    ce = nn.CrossEntropyLoss(ignore_index=model.pad_token).to(DEVICE)
    bce = nn.BCEWithLogitsLoss().to(DEVICE)

    print(f"\n=== Train HypSASRec | dataset={dataset_type} dir={dataset_dir_name} users={max_users} items={max_movies} seed={seed} ===")
    print(f"device={DEVICE}")

    best_val_hr = -1.0
    best_model_state = None
    no_improve = 0

    history = {
        "dataset_type": dataset_type,
        "dataset_dir_name": dataset_dir_name,
        "amazon_category": amazon_category,
        "config": CONFIG,
        "device": DEVICE,
        "num_users": int(num_users),
        "num_items": int(num_items),
        "epochs": [],
        "best_val_hr": None,
        "best_epoch": None,
        "test_hr": None,
        "test_ndcg": None,
        "seed": int(seed),
    }

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for seq, pos, neg in train_loader:
            seq = seq.to(DEVICE)
            pos = pos.to(DEVICE)

            logits = model(seq)              # [B, L, N]
            logits = logits.permute(0, 2, 1) # [B, N, L]
            loss = ce(logits, pos.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        train_loss = total_loss / max(1, num_batches)

        val_hr, val_ndcg = evaluate_model_hyp(model, val_loader, DEVICE, top_k=10)
        val_loss = evaluate_loss_hyp(model, val_loader, DEVICE, bce)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train CE Loss: {train_loss:.4f} | "
            f"Val BCE Loss: {val_loss:.4f} | "
            f"Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f}"
        )

        history["epochs"].append({
            "epoch": epoch + 1,
            "train_ce_loss": float(train_loss),
            "val_bce_loss": float(val_loss),
            "val_hr@10": float(val_hr),
            "val_ndcg@10": float(val_ndcg),
        })

        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\nLoading best checkpoint and evaluating on test...")
    model.load_state_dict(best_model_state)
    test_hr, test_ndcg = evaluate_model_hyp(model, test_loader, DEVICE, top_k=10)

    print("\n=== HypSASRec RESULTS ===")
    print(f"Best Val HR@10:  {best_val_hr:.4f}")
    print(f"Test HR@10:      {test_hr:.4f}")
    print(f"Test NDCG@10:    {test_ndcg:.4f}")

    history["best_val_hr"] = float(best_val_hr)
    history["test_hr"] = float(test_hr)
    history["test_ndcg"] = float(test_ndcg)
    if history["epochs"]:
        best_ep = max(history["epochs"], key=lambda e: e["val_hr@10"])
        history["best_epoch"] = best_ep["epoch"]

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # save npz geometry
    os.makedirs(save_dir, exist_ok=True)
    item_emb = model.item_emb.weight[:-1].detach().cpu().numpy().astype(np.float32)
    D = np.linalg.norm(item_emb[:, None, :] - item_emb[None, :, :], axis=-1).astype(np.float64)

    out_path = os.path.join(
        save_dir,
        f"hyp_sasrec_geometry_{dataset_type}_{max_users}x{max_movies}_seed{seed}.npz"
    )
    np.savez(out_path, Z=item_emb, D=D, best_val_hr=float(best_val_hr), test_hr=float(test_hr), test_ndcg=float(test_ndcg))
    print(f"[Save] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="movielens", choices=["amazon", "movielens"])
    parser.add_argument("--dataset_dir_name", default="ml-1m",#"amazon_beauty",
                        help="Folder name under GradIsomapCF_movielens/data/")
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care",
                        help="Used only for dataset_type=amazon")
    parser.add_argument("--max_users", type=int, default=6000)
    parser.add_argument("--max_movies", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history_path", 
                        default="Z:/Borisova/MANUL/recsys/SASRec/hyp_sasrec_movielens/history_hyp_sasrec_6000x3000_seed0.json")
    parser.add_argument("--save_dir",
                        default="Z:/Borisova/MANUL/recsys/SASRec/hyp_sasrec_movielens/")
    args = parser.parse_args()

    run_one(
        dataset_type=args.dataset_type,
        dataset_dir_name=args.dataset_dir_name,
        amazon_category=args.amazon_category,
        history_path=args.history_path,
        save_dir=args.save_dir,
        max_users=args.max_users,
        max_movies=args.max_movies,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()