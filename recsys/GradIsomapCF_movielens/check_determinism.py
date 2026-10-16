import argparse
import hashlib
import tempfile

import numpy as np
import pandas as pd
import torch

from determinism import seed_everything
from prepare_data import (prepare_sequences, subsample_users_items, train_val_test_split_next_item_df,
                          build_movie_user_matrix)
from new_datasets import NCFTestDatasetSampled
from fast_common import eval_tensors


def run_once(trainer, device, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(123)
    n = 6000
    df = pd.DataFrame({"userId": rng.integers(0, 70, n), "movieId": rng.integers(0, 90, n),
                       "rating": rng.integers(1, 6, n), "timestamp": rng.integers(0, 10**6, n)})
    df = df.drop_duplicates(["userId", "movieId"])
    dm, _ = prepare_sequences(df, build_user2seq=False)
    ds, u2s, nu, nm = subsample_users_items(dm, 60, 80, 5)
    train_events, val_next, _ = train_val_test_split_next_item_df(ds, 3)
    train_set = {u: set() for u in range(nu)}
    for u, m, r in train_events.tolist():
        train_set[int(u)].add(int(m))
    val_ds = NCFTestDatasetSampled(val_next, nm, train_set, num_ng=30, seed=123)
    feats = torch.tensor(build_movie_user_matrix(train_events, nm, nu), device=device)

    if trainer == "fast":
        from GradientIsomapCF_fast import GradientIsomapCF
    else:
        from GradientIsomapCF_log import GradientIsomapCF
    gi = GradientIsomapCF(
        train_feature=feats, train_events=train_events, num_users=nu, num_items=nm,
        user_pos_set=train_set, ng_seed=42, latent_len=8, n_neighbors=5, epochs=2, cf_epochs=2,
        final_cf_epochs=2, batch_size=256, lr_isomap=5e-2, lr_ncf=1e-3, factor_num=4, num_layers=2,
        logs_folder=tempfile.mkdtemp(), device=str(device), num_ng=2)
    if trainer == "fast":
        out = gi.train(val_tensors=eval_tensors(val_ds, device), top_k=10, device=device,
                       use_init_assumption=True)
    else:
        out = gi.train(val_loader=eval_tensors(val_ds, device), top_k=10, device=device)
    iso, ncf = out[:2]
    h = hashlib.sha256()
    for sd in (iso.state_dict(), ncf.state_dict()):
        for k in sorted(sd):
            h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest(), iso, ncf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", choices=["log", "fast"], default="fast")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.trainer == "log" and dev.type != "cuda":
        raise SystemExit("trainer=log использует Adam/Isomap.py с device='cuda' - нужна GPU")
    h1, *_ = run_once(args.trainer, dev)
    h2, *_ = run_once(args.trainer, dev)
    print("run1:", h1[:16], "run2:", h2[:16])
    assert h1 == h2, "NOT deterministic: weights differ between identical runs"
    print("OK: bit-identical weights across two identical runs")
