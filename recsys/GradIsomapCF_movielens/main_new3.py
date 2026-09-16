import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
import json
import time
import copy

from GradientIsomapCF_fast import GradientIsomapCF

from NCF import NCF
from prepare_data import (
    prepare_sequences,
    subsample_users_items,
    train_val_test_split_next_item,
    build_movie_user_matrix
)
from manifold_visualization import plot_gi_losses, plot_pure_ncf_losses, plot_gi_convergence
from new_datasets2 import NCFTestDatasetSampled
from fast_common import (
    build_forbidden_tensor,
    sample_negatives_device,
    evaluate_topk_vec,
    val_loss_vec, eval_tensors
)


def load_movielens_1m_ratings(ml1m_dir):
    ratings_path = os.path.join(ml1m_dir, "ratings.dat")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"]
    )
    print("Unique_users:", df['userId'].nunique(), 'Unique_items:', df['movieId'].nunique())
    return df


def train_pure_ncf(
        pure_model,
        train_pairs,
        user_pos_set_for_negs,
        num_users,
        num_movies,
        num_ng,
        top_k,
        device,
        val_tensors,
        epochs,
        patience,
        lr=5e-4,
        weight_decay=1e-5,
        batch_size=16384,
        select_by='hr',
        history=None,
        use_pos_weight=True
):
    users_pos = torch.tensor([p[0] for p in train_pairs], dtype=torch.long, device=device)
    items_pos = torch.tensor([p[1] for p in train_pairs], dtype=torch.long, device=device)
    n_pos = users_pos.numel()

    forbidden = build_forbidden_tensor(user_pos_set_for_negs, num_users, num_movies, device)

    pos_weight = torch.tensor([num_ng], device=device, dtype=torch.float32) if use_pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(pure_model.parameters(), lr=lr, weight_decay=weight_decay)

    vu, vi, vl = val_tensors

    best_hr_val, best_ndcg_val, best_val_loss, best_epoch = 0.0, 0.0, float("inf"), 0
    best_state = None
    patience_counter = 0
    score_fn = lambda u, i: pure_model(u, i)

    start_total = time.time()
    for epoch in range(epochs):
        t0 = time.time()

        users_rep = users_pos.repeat_interleave(num_ng)
        negs = sample_negatives_device(users_rep, num_movies, forbidden)
        users_all = torch.cat([users_pos, users_rep])
        items_all = torch.cat([items_pos, negs])
        labels_all = torch.cat([
            torch.ones(n_pos, dtype=torch.float32, device=device),
            torch.zeros(users_rep.numel(), dtype=torch.float32, device=device),
        ])
        n = users_all.numel()

        pure_model.train()
        perm = torch.randperm(n, device=device)
        total_loss_sum, n_batches = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            preds = pure_model(users_all[idx], items_all[idx])
            loss = loss_fn(preds, labels_all[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_sum += loss.item()
            n_batches += 1
        avg_train_loss = total_loss_sum / max(1, n_batches)

        pure_model.eval()
        avg_val_loss = val_loss_vec(score_fn, vu, vi, vl, device)
        hr_val, ndcg_val = evaluate_topk_vec(score_fn, vu, vi, vl, top_k, device)

        epoch_time = time.time() - t0
        print(f"[PURE NCF] epoch {epoch}: "
              f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"HR@{top_k}_val={hr_val:.4f}, NDCG@{top_k}_val={ndcg_val:.4f}, "
              f"time={epoch_time:.2f}s")

        if history is not None:
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['hr_val'].append(hr_val)
            history['ndcg_val'].append(ndcg_val)

        improved = (hr_val > best_hr_val) if select_by == 'hr' else (avg_val_loss < best_val_loss)
        if improved:
            best_hr_val, best_ndcg_val = hr_val, ndcg_val
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(pure_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[PURE NCF] Early stop at epoch {epoch}: no {select_by}-improvement for {patience} epochs\n")
                break

    if best_state is not None:
        pure_model.load_state_dict(best_state)

    total_time = time.time() - start_total
    print(f"\nPURE NCF best epoch {best_epoch}: HR@{top_k}_val={best_hr_val:.4f}, NDCG@{top_k}_val={best_ndcg_val:.4f}")
    print(f"Total time={total_time:.1f}s ({time.strftime('%H:%M:%S', time.gmtime(total_time))})\n")
    return pure_model


def main(
        max_users=300,
        max_movies=800,
        min_seq_len=5,
        num_ng=2,
        top_k=10,
        epochs_pure=100,
        gradisomap_epochs=5,
        run_ncf=True,
        factor_num=32,
        num_layers=3,
        run_gincf=True,
        n_run=68,
        patience_pure=15,
        partial_warm_start=False,
        batch_size=16384,
        outer_batch_size=1048576,
        select_by='loss',
        neg_exclude='train',
        lr_isomap=1e-3,
        lr_ncf=1e-3,
        cf_epochs=50,
        final_cf_epochs=25,
        save_matrices=False,
        n_neighbors=10,
        latent_len=32,
        use_init_assumption=False,
        warm_start_inner=False,
        outer_patience=12,
        use_item_projection=True,
        dist_type='pos',
        use_procrustes_align=True,
        ncf_weight_decay=1e-5,
        ncf_use_pos_weight=True
):

    torch.manual_seed(42)

    ml1m_dir = os.path.join("data", "ml-1m")
    ratings_df = load_movielens_1m_ratings(ml1m_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

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
    user_pos_train_set = {u: set() for u in range(num_users)}
    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)
    for (u, m, r) in train_events:
        user_pos_train_set[int(u)].add(int(m))

    user_hist_val_set = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    if neg_exclude == 'full':
        neg_set = user_pos_set
    elif neg_exclude == 'train':
        neg_set = user_pos_train_set    # He et al. NCF paper protocol
    else:
        raise ValueError(f"neg_exclude must be 'full' or 'train', got {neg_exclude!r}")
    print(f"num_ng = {num_ng}, neg_exclude = {neg_exclude}, batch_size = {batch_size}")

    val_dataset = NCFTestDatasetSampled(
        next_triples=val_next,
        num_items=num_movies,
        user_pos_all_set=user_hist_val_set,
        num_ng=99,
        seed=123,
    )
    test_dataset = NCFTestDatasetSampled(
        next_triples=test_next,
        num_items=num_movies,
        user_pos_all_set=user_hist_test_set,
        num_ng=99,
        seed=456,
    )
    val_tensors = eval_tensors(val_dataset, device)
    test_tensors = eval_tensors(test_dataset, device)

    train_pairs = [(int(u), int(m)) for (u, m, r) in train_events]

    if run_ncf:
        pure_model = NCF(
            user_num=num_users,
            item_num=num_movies,
            factor_num=factor_num,
            num_layers=num_layers,
            dropout=0.0,
            model='NeuMF-end',
        ).to(device)
        print(f"factor_num={factor_num}, num_layers={num_layers}")

        pure_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'hr_val': [], 'ndcg_val': []}

        train_pure_ncf(
            pure_model,
            train_pairs,
            neg_set,
            num_users=num_users,
            num_movies=num_movies,
            num_ng=num_ng,
            top_k=top_k,
            device=device,
            val_tensors=val_tensors,
            epochs=epochs_pure,
            patience=patience_pure,
            lr=lr_ncf,
            weight_decay=ncf_weight_decay,
            batch_size=batch_size,
            select_by=select_by,
            history=pure_history,
            use_pos_weight=ncf_use_pos_weight,
        )

        score_fn = lambda u, i: pure_model(u, i)
        hr_test_pure, ndcg_test_pure = evaluate_topk_vec(
            score_fn, *test_tensors, top_k, device)
        print(f"\nPURE NCF final on TEST: HR@{top_k}={hr_test_pure:.4f}, NDCG@{top_k}={ndcg_test_pure:.4f}\n")

        images_dir = os.path.join("logs_pure_ncf", "images/new_val")
        os.makedirs(images_dir, exist_ok=True)
        plot_pure_ncf_losses(pure_history, images_dir, run_name=f"pure_ncf_new_{n_run}")

    if run_gincf:
        print("\nGradientIsomapCF (IsomapNN + NeuMFOnManifold)")

        movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=False)
        features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)

        gi_cf = GradientIsomapCF(
            train_feature=features_t,
            train_events=train_events,
            num_users=num_users,
            num_items=num_movies,
            user_pos_set=neg_set, # same exclusion set as Pure NCF
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
            factor_num=32,
            num_layers=3,
            dropout=0.0,
            model_type='NeuMF-end',
            logs_folder=f"logs_movielens_isomap_cf/{n_run}",
            device=str(device),
            stop_criteria_value=0.001,
            num_ng=num_ng,
            partial_warm_start=partial_warm_start,
            warm_start_inner=warm_start_inner,
            outer_patience=outer_patience,
            use_item_projection=use_item_projection,
            use_procrustes_align=use_procrustes_align,
            select_by=select_by,
            save_matrices=save_matrices,
        )

        isomap_model, ncf_manifold_model, item_Z_final = gi_cf.train(
            val_tensors=val_tensors,
            top_k=top_k,
            device=device,
            use_init_assumption=use_init_assumption,
            dist_type=dist_type
        )

        score_fn_iso = lambda u, i: ncf_manifold_model(u, i, item_Z_final)
        hr_iso, ndcg_iso = evaluate_topk_vec(score_fn_iso, *test_tensors, top_k, device)
        print(f"\nGradientIsomapCF final on TEST: HR@{top_k}={hr_iso:.4f}, NDCG@{top_k}={ndcg_iso:.4f}\n")

        images_dir = os.path.join("logs_movielens_isomap_cf", f"{n_run}/images")
        os.makedirs(images_dir, exist_ok=True)
        plot_gi_losses(gi_cf.history, gi_cf.cf_history, images_dir, run_name="ginmf")

        with open(os.path.join(images_dir, "history.json"), 'w', encoding='utf-8') as f:
            json.dump(gi_cf.history, f, ensure_ascii=False, indent=4)

        plot_gi_convergence(gi_cf.history, top_k=top_k, save_path=os.path.join(images_dir, "metrics.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffIsomap + NCF bilevel training")
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--max-movies", type=int, default=800)
    parser.add_argument("--min-seq-len", type=int, default=20)
    parser.add_argument("--num-ng", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs-pure", type=int, default=50)
    parser.add_argument("--gradisomap-epochs", type=int, default=50)
    parser.add_argument("--patience-pure", type=int, default=40)
    parser.add_argument("--run-ncf", action="store_true", default=False)
    parser.add_argument("--run-gincf", action="store_true", default=False)
    parser.add_argument("--partial-warm-start", action="store_true", default=False)
    parser.add_argument("--n-run", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--outer-batch-size", type=int, default=4096)
    parser.add_argument("--select-by", choices=['hr', 'loss'], default='loss')
    parser.add_argument("--neg-exclude", choices=['full', 'train'], default='full',
                        help="'full' = symmetric negative exclusion; 'train' = He et al. NCF paper protocol")
    parser.add_argument("--lr-isomap", type=float, default=1e-3)
    parser.add_argument("--lr-ncf", type=float, default=1e-3,
                        help="lr for the inner/final NCF loops")
    parser.add_argument("--cf-epochs", type=int, default=50)
    parser.add_argument("--final-cf-epochs", type=int, default=20)
    parser.add_argument("--save-matrices", action="store_true", default=False)
    parser.add_argument("--n-neighbors", type=int, default=20, help="k for the Isomap kNN graph")
    parser.add_argument("--latent-len", type=int, default=1024)
    parser.add_argument("--use-init-assumption", action="store_true", default=False)
    parser.add_argument("--warm-start-inner", action="store_true", default=False,
                        help="fully reuse last outer epoch's NCF model")
    parser.add_argument("--outer-patience", type=int, default=40,
                        help="stop the gradisomap outer loop after this many outer")
    parser.add_argument("--no-item-projection", dest="use_item_projection",
                        action="store_false", default=True,
                        help="skip the learned item_GMF_linear/item_MLP_linear "
                             "layers; instead slice the raw MDS coordinates "
                             "directly (first factor_num / mlp_user_dim of "
                             "item_Z - eigh sorts components by descending "
                             "|eigenvalue|, so this is a fixed, non-learned "
                             "reduction). Requires --latent-len >= "
                             "factor_num*2**(num_layers-1).")
    parser.add_argument("--dist-type", type=str, default="pos",
                        choices=["pos", "uniform", "normal", "exp"],
                        help="random-matrix scheme for seeding the learnable "
                             "distance matrix when --use-init-assumption is NOT "
                             "given. 'pos' (default) is cdist of random "
                             "Gaussian points - the only one of the four that is "
                             "an honest Euclidean distance matrix, but its "
                             "pairwise distances concentrate tightly around a "
                             "common value in high dimension, making the initial "
                             "kNN ranking very sensitive to tiny perturbations. "
                             "'uniform'/'normal'/'exp' are i.i.d. entries "
                             "directly (symmetrized), not tied to any point "
                             "configuration. Ignored when --use-init-assumption "
                             "is given.")
    parser.add_argument("--no-procrustes", dest="use_procrustes_align",
                        action="store_false", default=True,
                        help="skip procrustes-aligning each epoch's raw Isomap "
                             "output to the previous epoch's frame before "
                             "feeding it to the inner NCF loop / outer gradient "
                             "step. Safe to disable when NEITHER "
                             "--partial-warm-start NOR --warm-start-inner is "
                             "given (nothing persists across epochs to become "
                             "miscalibrated by an arbitrary eigh rotation). ")
    parser.add_argument("--ncf-weight-decay", type=float, default=1e-5,
                        help="AdamW weight_decay for Pure NCF training ")
    parser.add_argument("--ncf-no-pos-weight", dest="ncf_use_pos_weight",
                        action="store_false", default=True,
                        help="Pure NCF's loss currently uses "
                             "BCEWithLogitsLoss(pos_weight=num_ng) - NOT what "
                             "the paper's Eq. 7 log loss is (a plain, unweighted "
                             "sum/mean over positives+negatives). Pass this to "
                             "match Eq. 7 literally")
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
        patience_pure=args.patience_pure,
        partial_warm_start=args.partial_warm_start,
        n_run=args.n_run,
        batch_size=args.batch_size,
        outer_batch_size=args.outer_batch_size,
        select_by=args.select_by,
        neg_exclude=args.neg_exclude,
        lr_isomap=args.lr_isomap,
        lr_ncf=args.lr_ncf,
        cf_epochs=args.cf_epochs,
        final_cf_epochs=args.final_cf_epochs,
        n_neighbors=args.n_neighbors,
        latent_len=args.latent_len,
        use_init_assumption=args.use_init_assumption,
        warm_start_inner=args.warm_start_inner,
        outer_patience=args.outer_patience,
        use_item_projection=args.use_item_projection,
        dist_type=args.dist_type,
        use_procrustes_align=args.use_procrustes_align,
        ncf_weight_decay=args.ncf_weight_decay,
        ncf_use_pos_weight=args.ncf_use_pos_weight,
        save_matrices=args.save_matrices,
    )
