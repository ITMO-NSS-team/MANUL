"""
Ablation for SASRec: does the outer-loop optimization of D_input actually 
improve downstream SASRec quality, or does the initial un-optimized geometry 
do about as well?
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

HERE = os.path.dirname(os.path.abspath(__file__))
MANUL_DIR = os.path.dirname(os.path.dirname(HERE))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, MANUL_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from Adam.Isomap import IsomapNN
from SASRecOnManifold_best import SASRecOnManifold, pad_item_Z
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

def train_and_eval_sasrec_on_fixed_Z(item_Z, data, device, latent_dim, sasrec_config,
                                     lr=1e-3, epochs=50, patience=5, seed=0, verbose=False):
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

class _IdentityIsomap(nn.Module):
    def __init__(self, Z):
        super().__init__()
        self.Z = Z
    def forward(self):
        return self.Z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True, help="Папка с логами основного запуска GradientIsomapSASRec")
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="amazon_beauty")
    parser.add_argument("--dataset_type", default="amazon", choices=["movielens", "amazon"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--latent_dim", type=int, default=64)
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
    Z_epochN = torch.tensor(np.load(epoch_files[-1])["Z"], dtype=torch.float32)

    results = {}
    for name, Z in [("pure_init (Euclidean)", Z_pure_init), ("epoch0 (1 step)", Z_epoch0), ("epochN (converged)", Z_epochN)]:
        print(f"\n--- training SASRec on {name} ---", flush=True)
        hr, ndcg = train_and_eval_sasrec_on_fixed_Z(Z, data, device, args.latent_dim, sasrec_config, verbose=True)
        results[name] = (hr, ndcg)
        print(f"{name}: test HR@10={hr:.4f} NDCG@10={ndcg:.4f}", flush=True)

    print("\n=== ABLATION SUMMARY ===")
    for name, (hr, ndcg) in results.items():
        print(f"{name:30s}  HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")

if __name__ == "__main__":
    main()
