import argparse
import copy
import glob
import os
import re
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from geoopt import PoincareBallExact
from HyperSASRec import HypSASRec

HERE = os.path.dirname(os.path.abspath(__file__))
MANUL_DIR = os.path.dirname(os.path.dirname(HERE))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, MANUL_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from Adam.Isomap import IsomapNN
from SASRecOnManifold_best2 import SASRecOnManifold, pad_item_Z
from sasrec_manifold_sampler import SASRecManifoldTrainSampler, SASRecManifoldTestDataset
from evaluation_sasrec_manifold import evaluate_topk_sasrec_isomap
from prepare_data import (
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
    build_movie_user_matrix,
)
from run_experiment_sasrec import load_movielens_ratings, load_amazon_ratings, _build_user_seq_train

def build_data_sasrec(max_users, max_movies, min_seq_len, dataset_dir_name, device, tmp_logs_folder,
                      dataset_type="movielens", amazon_category="Beauty_and_Personal_Care"):
    """Подготовка данных для SASRec (идентично run_experiment.py для SASRec)."""
    dataset_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)
    if dataset_type == "movielens":
        ratings_df = load_movielens_ratings(dataset_dir)
    elif dataset_type == "amazon":
        ratings_df = load_amazon_ratings(dataset_dir, amazon_category)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    df_mapped, user2seq = prepare_sequences(ratings_df)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies, min_seq_len=min_seq_len)

    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)

    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))
    
    user_train = _build_user_seq_train(train_events)

    user_hist_val_set = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    maxlen = 50
    val_dataset = SASRecManifoldTestDataset(
        user_train=user_train, next_triples=val_next, num_items=num_movies,
        user_pos_all_set=user_hist_val_set, maxlen=maxlen, num_ng=99, seed=123)
    test_dataset = SASRecManifoldTestDataset(
        user_train=user_train, next_triples=test_next, num_items=num_movies,
        user_pos_all_set=user_hist_test_set, maxlen=maxlen, num_ng=99, seed=456)

    train_sampler = SASRecManifoldTrainSampler(
        user_train=user_train, n_items=num_movies, maxlen=maxlen,
        user_pos_train_set=user_pos_train_set, seed=42)
    
    train_loader = DataLoader(train_sampler, batch_size=256, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True)
    features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

    with torch.no_grad():
        D_input_init = torch.cdist(features_t, features_t)
        D_input_init = D_input_init / D_input_init.max()

    return {
        "num_users": num_users, "num_movies": num_movies,
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
        "D_input_init": D_input_init, "maxlen": maxlen
    }

def poincare_mlr_pairwise(x, z_norm, z_unit, r, c):
    """
    x:      [B, L, H] hyp points (inside ball)
    z_norm: [B, L]
    z_unit: [B, L, H]
    r:      [B, L]
    c: scalar tensor on same device
    returns logits [B, L]
    """
    rc = torch.sqrt(c)
    drcr = 2.0 * rc * r
    rcx = rc * x
    cx2 = (rcx ** 2).sum(dim=-1)  # [B, L]
    denom_safe = torch.clamp_min(1.0 - cx2, 1e-15)  # [B, L]

    dot = (rcx * z_unit).sum(dim=-1)  # [B, L]
    num = 2.0 * dot * torch.cosh(drcr) - (1.0 + cx2) * torch.sinh(drcr)
    out = 2.0 * z_norm / rc * torch.asinh(num / denom_safe)
    return out


def poincare_mlr_candidates(x, w, r, c):
    """
    Candidate scoring.
    x: [B, H] hyp points
    w: [B, C, H] item weight vectors for candidates
    r: [B, C] candidate biases
    returns logits [B, C]
    """
    eps = 1e-12
    z_norm = torch.clamp_min(w.norm(dim=-1), eps)         # [B, C]
    z_unit = w / z_norm.unsqueeze(-1)                     # [B, C, H]

    rc = torch.sqrt(c)
    drcr = 2.0 * rc * r                                  # [B, C]
    rcx = rc * x                                          # [B, H]
    cx2 = (rcx ** 2).sum(dim=-1, keepdim=True)            # [B, 1]
    denom_safe = torch.clamp_min(1.0 - cx2, 1e-15)        # [B, 1]

    dot = (rcx.unsqueeze(1) * z_unit).sum(dim=-1)         # [B, C]
    num = 2.0 * dot * torch.cosh(drcr) - (1.0 + cx2) * torch.sinh(drcr)
    out = 2.0 * z_norm / rc * torch.asinh(num / denom_safe)
    return out

def eval_hypsasrec_topk(model, loader, device, k=10):
    model.eval()
    HR, NDCG = [], []
    with torch.no_grad():
        for seq, cands, labels in loader:
            seq = seq.to(device)          # [B, L]
            cands = cands.to(device)      # [B, C]

            log_feats = model.log2feats(seq)              # [B, L, H] (euclid)
            hyp_last = model.manifold.expmap0(log_feats[:, -1, :])  # [B, H] (ball)

            w = model.item_emb.weight[cands]              # [B, C, H]
            r = model.scaler[cands] if hasattr(model, "scaler") else torch.zeros_like(cands, dtype=torch.float32)
            r = r.to(device).to(torch.float32)

            logits = poincare_mlr_candidates(hyp_last, w, r, model.c)  # [B, C]

            pos_score = logits[:, 0]
            rank = (logits > pos_score.unsqueeze(1)).sum(dim=1)  # [B]

            hit = (rank < k).to(torch.float32)
            HR.extend(hit.cpu().numpy().tolist())

            ndcg = torch.where(
                rank < k,
                1.0 / torch.log2(rank.to(torch.float32) + 2.0),
                torch.zeros_like(rank, dtype=torch.float32),
            )
            NDCG.extend(ndcg.cpu().numpy().tolist())

    return float(np.mean(HR)) if HR else 0.0, float(np.mean(NDCG)) if NDCG else 0.0

def eval_hypsasrec_val_loss(model, loader, device, loss_fn):
    """
    Validation loss on sampled candidates (same contract as your val_loader):
      batch: (seq, cands, labels)
    labels: [B, C] float (1 for pos at index 0, 0 for negatives)
    """
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for seq, cands, labels in loader:
            seq = seq.to(device)            # [B, L]
            cands = cands.to(device)        # [B, C]
            labels = labels.to(device)      # [B, C] float32

            log_feats = model.log2feats(seq)                         # [B, L, H]
            hyp_last = model.manifold.expmap0(log_feats[:, -1, :])   # [B, H]

            w = model.item_emb.weight[cands]                         # [B, C, H]
            if hasattr(model, "scaler"):
                r = model.scaler[cands].to(torch.float32)            # [B, C]
            else:
                r = torch.zeros_like(cands, dtype=torch.float32, device=device)

            logits = poincare_mlr_candidates(hyp_last, w, r, model.c)  # [B, C]
            loss = loss_fn(logits, labels)

            total += float(loss.item())
            n_batches += 1

    return total / max(1, n_batches)

def train_and_eval_sasrec_on_fixed_Z_old(item_Z, data, device, latent_dim, sasrec_config,
                                     lr=1e-3, epochs=50, patience=8, seed=0, verbose=False):
    """Учим SASRec на ЗАМОРОЖЕННОЙ геометрии Z."""
    torch.manual_seed(seed)
    item_Z = item_Z.to(device)
    item_Z_padded = pad_item_Z(item_Z)

    sasrec = SASRecOnManifold(
        config=sasrec_config, item_num=data["num_movies"], latent_dim=latent_dim,
        freeze_item_projection=True, item_projection_init_data=item_Z
    ).to(device)
    
    optimizer = optim.Adam(sasrec.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.BCEWithLogitsLoss()
    pad_token = data["num_movies"]

    best_val_hr = -float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        sasrec.train()
        total_loss, n_batches = 0.0, 0
        for seq, pos, neg in data["train_loader"]:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = sasrec(seq, pos, neg, item_Z_padded)
            indices = torch.where(pos != pad_token)
            
            loss = loss_fn(pos_logits[indices], torch.ones_like(pos_logits[indices]))
            loss += loss_fn(neg_logits[indices], torch.zeros_like(neg_logits[indices]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= 100: break # Ограничим batches для скорости абляции

        sasrec.eval()
        hr, ndcg = evaluate_topk_sasrec_isomap(sasrec, _IdentityIsomap(item_Z), data["val_loader"], top_k=10, device=device)
        
        if verbose:
            print(f"  ep {ep+1}/{epochs} val_hr@10={hr:.4f}", flush=True)

        if hr > best_val_hr:
            best_val_hr = hr
            best_state = copy.deepcopy(sasrec.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            if verbose: print(f"  early stop at ep {ep+1}", flush=True)
            break

    if best_state is not None:
        sasrec.load_state_dict(best_state)

    hr, ndcg = evaluate_topk_sasrec_isomap(sasrec, _IdentityIsomap(item_Z), data["test_loader"], top_k=10, device=device)
    return hr, ndcg


def train_and_eval_hypsasrec_on_fixed_Z(
    item_Z, data, device, latent_dim, sasrec_config,
    lr=1e-3, epochs=50, patience=8, seed=0,
    c=1.0, freeze_items=True, freeze_scaler=True,
    verbose=False
):
    """
    item_Z: [n_items, latent_dim] float tensor
    """
    torch.manual_seed(seed)
    n_items = data["num_movies"]
    pad_token = n_items

    # --- Build HypSASRec ---
    # Важно: hidden_units должно совпадать с dim Z (или надо добавлять проекцию)
    config = dict(sasrec_config)
    config["hidden_units"] = latent_dim
    config["geom"] = "ball"
    config["c"] = float(c)
    config["bias"] = True  # scaler exists; можно True, но мы его заморозим при желании

    model = HypSASRec(config, item_num=n_items).to(device)

    # ВАЖНО: перенести c на device (у тебя в классе это не buffer)
    model.c = torch.tensor(float(c), device=device)

    # --- Init item_emb from Z (+ pad row) ---
    item_Z = item_Z.to(device).to(torch.float32)
    with torch.no_grad():
        model.item_emb.weight[:n_items].copy_(item_Z)
        model.item_emb.weight[pad_token].zero_()

    if freeze_items:
        model.item_emb.weight.requires_grad_(False)
    if freeze_scaler and hasattr(model, "scaler"):
        model.scaler.requires_grad_(False)

    # optimizer only for trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.BCEWithLogitsLoss()
    val_loss_fn = nn.BCEWithLogitsLoss()

    best_val_hr = -1e9
    best_state = None
    no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_hr": [],
        "val_ndcg": [],
        "best_val_hr": -1e9,     # заполним в конце
        "best_epoch": -1,
        "c": float(c),
        "freeze_items": bool(freeze_items),
        "freeze_scaler": bool(freeze_scaler),
        "lr": float(lr),
        "epochs": int(epochs),
        "patience": int(patience),
        "seed": int(seed),
        "latent_dim": int(latent_dim),
    }

    for ep in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for seq, pos, neg in data["train_loader"]:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            log_feats = model.log2feats(seq)              # [B, L, H] euclid
            hyp_feats = model.manifold.expmap0(log_feats) # [B, L, H] ball

            # pos logits
            w_pos = model.item_emb.weight[pos]            # [B, L, H]
            z_norm = torch.clamp_min(w_pos.norm(dim=-1), 1e-12)
            z_unit = w_pos / z_norm.unsqueeze(-1)
            r_pos = model.scaler[pos].to(torch.float32) if hasattr(model, "scaler") else torch.zeros_like(z_norm)
            pos_logits = poincare_mlr_pairwise(hyp_feats, z_norm, z_unit, r_pos, model.c)

            # neg logits
            w_neg = model.item_emb.weight[neg]
            z_norm_n = torch.clamp_min(w_neg.norm(dim=-1), 1e-12)
            z_unit_n = w_neg / z_norm_n.unsqueeze(-1)
            r_neg = model.scaler[neg].to(torch.float32) if hasattr(model, "scaler") else torch.zeros_like(z_norm_n)
            neg_logits = poincare_mlr_pairwise(hyp_feats, z_norm_n, z_unit_n, r_neg, model.c)

            idx = torch.where(pos != pad_token)
            loss = loss_fn(pos_logits[idx], torch.ones_like(pos_logits[idx]))
            loss = loss + loss_fn(neg_logits[idx], torch.zeros_like(neg_logits[idx]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            if n_batches >= 100:
                break

        #train_loss = total_loss / max(1, n_batches)
#
        ## val
        #val_hr, val_ndcg = eval_hypsasrec_topk(model, data["val_loader"], device, k=10)
#
        #if verbose:
        #    print(f"  ep {ep+1}/{epochs} train_loss={train_loss:.4f} val_hr@10={val_hr:.4f} val_ndcg@10={val_ndcg:.4f}", flush=True)
#
        train_loss = total_loss / max(1, n_batches)

        # --- Validation: loss + HR/NDCG ---
        avg_val_loss = eval_hypsasrec_val_loss(model, data["val_loader"], device, val_loss_fn)
        val_hr, val_ndcg = eval_hypsasrec_topk(model, data["val_loader"], device, k=10)

        history["epoch"].append(int(ep))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_hr"].append(float(val_hr))
        history["val_ndcg"].append(float(val_ndcg))
        # -------------------

        if verbose:
            print(
                f"  ep {ep+1}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={avg_val_loss:.4f} "
                f"val_hr@10={val_hr:.4f} val_ndcg@10={val_ndcg:.4f}",
                flush=True
            )

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            history["best_val_hr"] = float(val_hr)
            history["best_epoch"] = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at ep {ep+1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_hr, test_ndcg = eval_hypsasrec_topk(model, data["test_loader"], device, k=10)
    return test_hr, test_ndcg, history


def train_and_eval_hypsasrec_on_fixed_Z_nolog(
    item_Z, data, device, latent_dim, sasrec_config,
    lr=1e-3, epochs=50, patience=8, seed=0,
    c=1.0, freeze_items=True, freeze_scaler=True,
    verbose=False
):
    """
    item_Z: [n_items, latent_dim] float tensor
    """
    torch.manual_seed(seed)
    n_items = data["num_movies"]
    pad_token = n_items

    # --- Build HypSASRec ---
    # Важно: hidden_units должно совпадать с dim Z (или надо добавлять проекцию)
    config = dict(sasrec_config)
    config["hidden_units"] = latent_dim
    config["geom"] = "ball"
    config["c"] = float(c)
    config["bias"] = True  # scaler exists; можно True, но мы его заморозим при желании

    model = HypSASRec(config, item_num=n_items).to(device)

    # ВАЖНО: перенести c на device (у тебя в классе это не buffer)
    model.c = torch.tensor(float(c), device=device)

    # --- Init item_emb from Z (+ pad row) ---
    item_Z = item_Z.to(device).to(torch.float32)
    with torch.no_grad():
        model.item_emb.weight[:n_items].copy_(item_Z)
        model.item_emb.weight[pad_token].zero_()

    if freeze_items:
        model.item_emb.weight.requires_grad_(False)
    if freeze_scaler and hasattr(model, "scaler"):
        model.scaler.requires_grad_(False)

    # optimizer only for trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.BCEWithLogitsLoss()
    val_loss_fn = nn.BCEWithLogitsLoss()

    best_val_hr = -1e9
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for seq, pos, neg in data["train_loader"]:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            log_feats = model.log2feats(seq)              # [B, L, H] euclid
            hyp_feats = model.manifold.expmap0(log_feats) # [B, L, H] ball

            # pos logits
            w_pos = model.item_emb.weight[pos]            # [B, L, H]
            z_norm = torch.clamp_min(w_pos.norm(dim=-1), 1e-12)
            z_unit = w_pos / z_norm.unsqueeze(-1)
            r_pos = model.scaler[pos].to(torch.float32) if hasattr(model, "scaler") else torch.zeros_like(z_norm)
            pos_logits = poincare_mlr_pairwise(hyp_feats, z_norm, z_unit, r_pos, model.c)

            # neg logits
            w_neg = model.item_emb.weight[neg]
            z_norm_n = torch.clamp_min(w_neg.norm(dim=-1), 1e-12)
            z_unit_n = w_neg / z_norm_n.unsqueeze(-1)
            r_neg = model.scaler[neg].to(torch.float32) if hasattr(model, "scaler") else torch.zeros_like(z_norm_n)
            neg_logits = poincare_mlr_pairwise(hyp_feats, z_norm_n, z_unit_n, r_neg, model.c)

            idx = torch.where(pos != pad_token)
            loss = loss_fn(pos_logits[idx], torch.ones_like(pos_logits[idx]))
            loss = loss + loss_fn(neg_logits[idx], torch.zeros_like(neg_logits[idx]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            if n_batches >= 100:
                break

        #train_loss = total_loss / max(1, n_batches)
#
        ## val
        #val_hr, val_ndcg = eval_hypsasrec_topk(model, data["val_loader"], device, k=10)
#
        #if verbose:
        #    print(f"  ep {ep+1}/{epochs} train_loss={train_loss:.4f} val_hr@10={val_hr:.4f} val_ndcg@10={val_ndcg:.4f}", flush=True)
#
        train_loss = total_loss / max(1, n_batches)

        # --- Validation: loss + HR/NDCG ---
        avg_val_loss = eval_hypsasrec_val_loss(model, data["val_loader"], device, val_loss_fn)
        val_hr, val_ndcg = eval_hypsasrec_topk(model, data["val_loader"], device, k=10)

        if verbose:
            print(
                f"  ep {ep+1}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={avg_val_loss:.4f} "
                f"val_hr@10={val_hr:.4f} val_ndcg@10={val_ndcg:.4f}",
                flush=True
            )

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at ep {ep+1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_hr, test_ndcg = eval_hypsasrec_topk(model, data["test_loader"], device, k=10)
    return test_hr, test_ndcg


class _IdentityIsomap(nn.Module):
    def __init__(self, Z):
        super().__init__()
        self.Z = Z
    def forward(self):
        return self.Z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True, help="Папка с логами основного запуска GradientIsomapSASRec")
    parser.add_argument("--max_users", type=int, default=3000)
    parser.add_argument("--max_movies", type=int, default=2500)
    parser.add_argument("--dataset_dir_name", default="amazon_beauty")
    parser.add_argument("--dataset_type", default="amazon", choices=["movielens", "amazon"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--best_epoch", type=int, default=300)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--geom_c", default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sasrec_config = {'hidden_units': 64, 'maxlen': 50, 'num_blocks': 2, 'num_heads': 1, 'dropout_rate': 0.2}

    tmp_logs = os.path.join(HERE, "ablation_tmp_logs")

    data = build_data_sasrec(args.max_users, args.max_movies, 5, args.dataset_dir_name, device, tmp_logs, args.dataset_type, args.amazon_category)

    # 1. Загружаем геометрии из сохраненных файлов основного эксперимента
    D_init = np.load(os.path.join(args.logs_folder, "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(D_init_t, n_components=args.latent_dim, n_neighbors=10).to(device)
    with torch.no_grad():
        Z_pure_init = pure_init_model().to(torch.float32).detach()

    epoch_files = glob.glob(os.path.join(args.logs_folder, "matrices_epoch*.npz"))
    epoch_files.sort(key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    
    Z_epoch0 = torch.tensor(np.load(epoch_files[0])["Z"], dtype=torch.float32)
    #Z_epochN = torch.tensor(np.load(epoch_files[args.best_epoch])["Z"], dtype=torch.float32)

    npz_path = os.path.join(args.logs_folder, f"matrices_epoch{args.best_epoch}.npz")
    Z_epochN = torch.tensor(np.load(npz_path)["Z"], dtype=torch.float32)
    #Z_epochN = torch.tensor(np.load(f"matrices_epoch{args.best_epoch}.npz")["Z"], dtype=torch.float32)
    
    #results = {}
    #for name, Z in [("pure_init (Euclidean)", Z_pure_init), ("epoch0 (1 step)", Z_epoch0), ("epochN (converged)", Z_epochN)]:
    #    print(f"\n--- training SASRec on {name} ---", flush=True)
    #    hr, ndcg = train_and_eval_sasrec_on_fixed_Z(Z, data, device, args.latent_dim, sasrec_config, verbose=True)
    #    results[name] = (hr, ndcg)
    #    print(f"{name}: test HR@10={hr:.4f} NDCG@10={ndcg:.4f}", flush=True)


        # --- куда сохраняем результаты абляции ---
    #save_dir = os.path.join(HERE, "ablation", "ablation_exp_2_best")
    #save_dir = os.path.join("Z:/Borisova/MANUL/recsys/SASRec/", "ablation", "ablation_exp_3000x2500_seed0_best")
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    def _slug(s: str) -> str:
        # безопасное имя файла
        return (s.lower()
                 .replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace("/", "_"))

    results = {}
    for name, Z in [
        ("pure_init (Euclidean)", Z_pure_init),
        ("epoch0 (1 step)", Z_epoch0),
        ("epochN (converged)", Z_epochN),
    ]:
        print(f"\n--- training HypSASRec on {name} ---", flush=True)
        hr, ndcg, hist = train_and_eval_hypsasrec_on_fixed_Z(
            Z, data, device, args.latent_dim, sasrec_config,
            lr=args.lr if hasattr(args, "lr") else 1e-3,
            epochs=args.epochs if hasattr(args, "epochs") else 50,
            patience=args.patience if hasattr(args, "patience") else 8,
            verbose=True,
            c=args.geom_c,
            freeze_items=True,
            freeze_scaler=True,
        )
        results[name] = (hr, ndcg)
        print(f"{name}: test HR@10={hr:.4f} NDCG@10={ndcg:.4f}", flush=True)

        # --- SAVE NPZ: Z + D(Z) + metrics ---
        Z_np = Z.detach().cpu().numpy().astype(np.float32)
        D_np = np.linalg.norm(Z_np[:, None, :] - Z_np[None, :, :], axis=-1).astype(np.float64)

        out_path = os.path.join(save_dir, f"{_slug(name)}.npz")
        np.savez(out_path, Z=Z_np, D=D_np, hr=float(hr), ndcg=float(ndcg))
        print(f"[Save] {out_path}", flush=True)

        hist["name"] = name
        hist["c"] = 5.0
        hist["save_npz"] = os.path.basename(out_path)
        hist_path = os.path.join(save_dir, f"{_slug(name)}_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)
        print(f"[Save] {hist_path}", flush=True)

    print("\n=== ABLATION SUMMARY ===")
    for name, (hr, ndcg) in results.items():
        print(f"{name:30s}  HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")

if __name__ == "__main__":
    main()
