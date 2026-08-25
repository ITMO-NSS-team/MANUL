"""
Ablation: does the outer-loop optimization of D_input actually improve
downstream quality, or does "final NCF" retraining do about as well on the
un-optimized initial geometry? Motivated by a real bug found during review
(GradientIsomapCF_log.py's "final NCF" step used a broken best-checkpoint
save - a bare state_dict() reference, not a deepcopy - so every previously
reported test_hr/test_ndcg number reflected a noisy late-training NCF state,
not the genuinely best one; see the commit fixing it). Reuses the saved
Z snapshots (matrices_epoch*.npz) from an already-completed sweep run rather
than re-running the expensive 30-outer-epoch bilevel optimization - only the
downstream "train a NeuMFOnManifold head on a fixed Z, evaluate" step is
redone, now with the checkpoint bug fixed.

Compares three geometry snapshots for a given config:
  - "pure_init": D_input_init.npy passed through one untrained IsomapNN
    forward pass (zero outer steps - the raw Euclidean-cdist geometry)
  - "epoch0": Z from matrices_epoch0.npz (one outer step taken)
  - "epochN": Z from the last saved matrices_epoch*.npz (fully converged)
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

from Adam.Isomap import IsomapNN
from NCF import NCF
from NeuMFOnManifold import NeuMFOnManifold
from evaluation import evaluate_topk_isomap, evaluate_topk_pure
from prepare_data import (
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
    build_movie_user_matrix,
)
from new_datasets import NCFTestDatasetSampled
from run_experiment import load_movielens_ratings
from GradientIsomapCF_log import GradientIsomapCF


def build_data(max_users, max_movies, min_seq_len, num_ng, dataset_dir_name, device, tmp_logs_folder):
    """Mirrors run_experiment.main()'s data pipeline exactly, including
    reusing GradientIsomapCF's own internal negative-sampling for training
    interactions (NOT new_datasets.NCFTrainDatasetFutureBlind - an earlier
    version of this script used that by mistake, which excludes negatives
    by TRAIN-ONLY history; GradientIsomapCF's own _build_implicit_interactions
    excludes by FULL train+val+test history via self.user_pos_set, a
    materially different negative-sampling scheme. Constructing a real
    GradientIsomapCF instance (cheap - building interactions happens in
    __init__, no training triggered) and reusing its .inter_loader/
    .users_all/.items_all/.labels_all is the only way to guarantee this
    ablation trains on exactly the same data the real pipeline's "final NCF"
    step does)."""
    movielens_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)
    ratings_df = load_movielens_ratings(movielens_dir)
    df_mapped, user2seq = prepare_sequences(ratings_df)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies, min_seq_len=min_seq_len)

    user_pos_set = {u: set(m for (m, ts, r) in seq) for u, seq in user2seq_sub.items()}

    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)

    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))
    user_hist_val_set = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    val_dataset = NCFTestDatasetSampled(
        next_triples=val_next, num_items=num_movies,
        user_pos_all_set=user_hist_val_set, num_ng=99, seed=123)
    test_dataset = NCFTestDatasetSampled(
        next_triples=test_next, num_items=num_movies,
        user_pos_all_set=user_hist_test_set, num_ng=99, seed=456)

    movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True)
    features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

    # Constructed only to reuse its interaction-building - .train() is never
    # called, so no GPU training happens here.
    gi_cf = GradientIsomapCF(
        train_feature=features_t, train_events=train_events,
        num_users=num_users, num_items=num_movies, user_pos_set=user_pos_set,
        ng_seed=42, latent_len=64, n_neighbors=10, batch_size=2048,
        logs_folder=tmp_logs_folder, device=str(device), num_ng=num_ng,
    )

    # Same D_input initialization GradientIsomapCF.train(use_init_assumption=
    # True) computes internally (torch.cdist on rating-profile features,
    # normalized by max) - exposed here so other scripts (e.g. the Poincare
    # baseline) can fit alternative geometries against the identical target,
    # for a fair "same starting information, different geometry" comparison.
    with torch.no_grad():
        D_input_init = torch.cdist(features_t, features_t)
        D_input_init = D_input_init / D_input_init.max()

    return {
        "num_users": num_users, "num_movies": num_movies,
        "inter_loader": gi_cf.inter_loader,
        "val_loader": DataLoader(val_dataset, batch_size=100, shuffle=False),
        "test_loader": DataLoader(test_dataset, batch_size=100, shuffle=False),
        "D_input_init": D_input_init,
    }


def train_and_eval_ncf_on_fixed_Z(item_Z, data, device, latent_dim, factor_num=16, num_layers=3,
                                  lr=1e-3, epochs=30, patience=3, seed=0, verbose=False,
                                  return_history=False):
    """Faithfully mirrors GradientIsomapCF.train()'s "final NCF" block
    (GradientIsomapCF_log.py), with the checkpoint bug fixed (deepcopy)."""
    torch.manual_seed(seed)
    item_Z = item_Z.to(device)

    ncf = NeuMFOnManifold(
        user_num=data["num_users"], latent_dim=latent_dim, factor_num=factor_num,
        num_layers=num_layers, dropout=0.0, model_type="NeuMF-end",
    ).to(device)
    optimizer = optim.AdamW(ncf.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(epochs):
        ncf.train()
        total_train_loss, n_train_batches = 0.0, 0
        for batch_users, batch_items, batch_labels in data["inter_loader"]:
            batch_users, batch_items, batch_labels = (
                batch_users.to(device), batch_items.to(device), batch_labels.to(device))
            preds = ncf(batch_users, batch_items, item_Z)
            loss = loss_fn(preds, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_train_batches += 1
        avg_train_loss = total_train_loss / max(1, n_train_batches)

        ncf.eval()
        total_val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for val_users, val_items, val_labels in data["val_loader"]:
                val_users, val_items, val_labels = (
                    val_users.to(device), val_items.to(device), val_labels.to(device))
                preds_val = ncf(val_users, val_items, item_Z)
                total_val_loss += loss_fn(preds_val, val_labels).item()
                n_val_batches += 1
        avg_val_loss = total_val_loss / max(1, n_val_batches)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        if verbose:
            print(f"  ep {ep+1}/{epochs} train={avg_train_loss:.4f} val={avg_val_loss:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(ncf.state_dict())  # the actual fix
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            if verbose:
                print(f"  early stop at ep {ep+1}", flush=True)
            break

    if best_state is not None:
        ncf.load_state_dict(best_state)

    hr, ndcg = evaluate_topk_isomap(ncf, _IdentityIsomap(item_Z), data["test_loader"], top_k=10, device=device)
    if return_history:
        return hr, ndcg, best_val_loss, history
    return hr, ndcg, best_val_loss


def train_and_eval_euclidean_baseline(data, device, factor_num=16, num_layers=3, lr=1e-3,
                                      epochs=30, patience=3, seed=0, return_item_embeddings=False):
    """Plain NeuMF (learnable embedding tables, geometry-free) trained
    through the exact same inter_loader/val_loader/test_loader as the
    manifold-based arms (train_and_eval_ncf_on_fixed_Z, poincare_baseline.py)
    - same negative-sampling scheme, same batch composition, same
    optimizer/epochs/early-stopping - so the only thing that differs across
    all three comparison arms is the item representation itself."""
    torch.manual_seed(seed)

    ncf = NCF(
        user_num=data["num_users"], item_num=data["num_movies"], factor_num=factor_num,
        num_layers=num_layers, dropout=0.0, model="NeuMF-end",
    ).to(device)
    optimizer = optim.AdamW(ncf.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        ncf.train()
        for batch_users, batch_items, batch_labels in data["inter_loader"]:
            batch_users, batch_items, batch_labels = (
                batch_users.to(device), batch_items.to(device), batch_labels.to(device))
            preds = ncf(batch_users, batch_items)
            loss = loss_fn(preds, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ncf.eval()
        total_val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for val_users, val_items, val_labels in data["val_loader"]:
                val_users, val_items, val_labels = (
                    val_users.to(device), val_items.to(device), val_labels.to(device))
                preds_val = ncf(val_users, val_items)
                total_val_loss += loss_fn(preds_val, val_labels).item()
                n_val_batches += 1
        avg_val_loss = total_val_loss / max(1, n_val_batches)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(ncf.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        ncf.load_state_dict(best_state)

    hr, ndcg = evaluate_topk_pure(ncf, data["test_loader"], top_k=10, device=device)
    if return_item_embeddings:
        with torch.no_grad():
            item_emb = torch.cat(
                [ncf.embed_item_GMF.weight, ncf.embed_item_MLP.weight], dim=-1
            ).detach().cpu()
        return hr, ndcg, best_val_loss, item_emb
    return hr, ndcg, best_val_loss


class _IdentityIsomap(nn.Module):
    """Wraps a precomputed Z so evaluate_topk_isomap (which calls
    isomap_model()) can be reused unmodified."""
    def __init__(self, Z):
        super().__init__()
        self.Z = Z

    def forward(self):
        return self.Z

    def eval(self):
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True)
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="ml-1m")
    parser.add_argument("--latent_dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    tmp_logs_folder = os.path.join(HERE, "ablation_tmp_logs")
    data = build_data(args.max_users, args.max_movies, min_seq_len=2, num_ng=2,
                      dataset_dir_name=args.dataset_dir_name, device=device,
                      tmp_logs_folder=tmp_logs_folder)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    epoch_files = glob.glob(os.path.join(args.logs_folder, "matrices_epoch*.npz"))
    def epoch_num(p):
        m = re.search(r"matrices_epoch(\d+)\.npz", os.path.basename(p))
        return int(m.group(1)) if m else -1
    epoch_files = sorted(epoch_files, key=epoch_num)
    last_epoch_file = epoch_files[-1]
    last_epoch_n = epoch_num(last_epoch_file)

    D_init = np.load(os.path.join(args.logs_folder, "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(weights_initial_assumption=D_init_t, n_components=args.latent_dim,
                               n_neighbors=10).to(device)
    with torch.no_grad():
        Z_pure_init = pure_init_model().to(torch.float32).detach()

    Z_epoch0 = torch.tensor(np.load(epoch_files[0])["Z"], dtype=torch.float32)
    Z_epochN = torch.tensor(np.load(last_epoch_file)["Z"], dtype=torch.float32)

    results = {}
    for name, Z in [("pure_init (0 outer steps)", Z_pure_init),
                    ("epoch0 (1 outer step)", Z_epoch0),
                    (f"epoch{last_epoch_n} (converged)", Z_epochN)]:
        print(f"\n--- training final NCF on {name} ---", flush=True)
        hr, ndcg, val_loss = train_and_eval_ncf_on_fixed_Z(Z, data, device, args.latent_dim)
        results[name] = (hr, ndcg, val_loss)
        print(f"{name}: test HR@10={hr:.4f} NDCG@10={ndcg:.4f} best_val_loss={val_loss:.4f}", flush=True)

    print("\n=== ABLATION SUMMARY ===")
    for name, (hr, ndcg, val_loss) in results.items():
        print(f"{name:30s}  HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    main()
