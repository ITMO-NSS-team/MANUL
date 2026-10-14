"""
Decisive test of whether NCF actually depends on the manifold's geometric
structure once the item-side projection can no longer freely re-interpret
it (2026-09-09). With freeze_item_projection=True (NeuMFOnManifold): the
MLP branch takes z_i unchanged (identity, latent_dim==mlp_user_dim by
design), the GMF branch takes a FIXED PCA projection of the actual item_Z
data (not random - a random frozen compression can be arbitrarily poorly
conditioned, per the user's concern) down to factor_num dims. Only the
user-side embeddings and the downstream MLP tower/predict layer remain
trainable.

Three conditions, same frozen-projection architecture, same training
settings (dropout=0.2, select_by=hr, patience=30, cap=200):
  - real: the actual optimized Z (matrices_epoch250.npz snapshot)
  - shuffled: same Z values, rows randomly permuted (each item still gets
    a unique 64-dim vector, but not ITS correct one) - tests whether the
    correct item<->coordinate assignment matters, now that the network
    can't relearn its way around a shuffle
  - random: same shape, fresh iid Gaussian noise matched to Z's per-dim
    mean/std - a more extreme "no real structure at all" control

If real clearly beats shuffled/random now (unlike the unfrozen
architecture, where this was never tested), that confirms freezing the
projection forces genuine dependence on the manifold's geometry. If real
still doesn't beat shuffled/random, the escape hatch is elsewhere
(downstream MLP tower / GMF's free user-side multiplication).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from ablation_geometry_vs_optimization import build_data, train_and_eval_ncf_on_fixed_Z

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}", flush=True)

Z_np = np.load(f"{HERE}/logs_movielens_isomap_cf/amazon_beauty_outer500_eta1e5_test_1e-05/matrices_epoch250.npz")["Z"]
real_Z = torch.tensor(Z_np, dtype=torch.float32)

rng = np.random.RandomState(0)
perm = rng.permutation(real_Z.shape[0])
shuffled_Z = real_Z[perm].clone()

random_Z = torch.tensor(
    rng.normal(loc=Z_np.mean(axis=0), scale=Z_np.std(axis=0), size=Z_np.shape),
    dtype=torch.float32,
)

data = build_data(
    max_users=300, max_movies=800, min_seq_len=2, num_ng=2,
    dataset_dir_name="amazon_beauty", device=device,
    tmp_logs_folder=f"{HERE}/frozen_proj_tmp_logs",
    dataset_type="amazon", amazon_category="Beauty_and_Personal_Care",
)
print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

CONDITIONS = [
    ("real", real_Z, 5),
    ("shuffled", shuffled_Z, 5),
    ("random", random_Z, 3),
]

all_results = {}
all_histories = {}
for label, Z, n_seeds in CONDITIONS:
    print(f"\n=== {label} Z, freeze_item_projection=True ===", flush=True)
    test_hrs, val_losses = [], []
    histories = []
    for seed in range(n_seeds):
        hr, ndcg, best_val_loss, history = train_and_eval_ncf_on_fixed_Z(
            Z, data, device, latent_dim=64, epochs=200, patience=30,
            seed=seed, select_by="hr", dropout=0.2, weight_decay=0.01,
            freeze_item_projection=True, return_history=True,
        )
        n_epochs_used = len(history["train_loss"])
        print(f"  seed={seed}: test_hr={hr:.4f} test_ndcg={ndcg:.4f} best_val_loss={best_val_loss:.4f} "
              f"n_epochs={n_epochs_used}", flush=True)
        test_hrs.append(hr)
        val_losses.append(best_val_loss)
        histories.append(history)
    test_hrs = np.array(test_hrs)
    val_losses = np.array(val_losses)
    all_results[label] = (test_hrs, val_losses)
    all_histories[label] = histories
    print(f"  test_hr: mean={test_hrs.mean():.4f} std={test_hrs.std():.4f} range=[{test_hrs.min():.4f},{test_hrs.max():.4f}]")
    print(f"  val_loss: mean={val_losses.mean():.4f} std={val_losses.std():.4f}")

with open(f"{HERE}/frozen_projection_geometry_test_histories.pkl", "wb") as f:
    pickle.dump(all_histories, f)
print(f"[Save] {HERE}/frozen_projection_geometry_test_histories.pkl")

print("\n\n=== SUMMARY ===")
print("(for reference, unfrozen dropout=0.2 baseline on REAL Z, 5 seeds: "
      "test_hr=0.1720+-0.0204, val_loss rel_spread=0.037)")
for label, (test_hrs, val_losses) in all_results.items():
    print(f"{label:>10}: test_hr mean={test_hrs.mean():.4f} std={test_hrs.std():.4f} "
          f"range=[{test_hrs.min():.4f},{test_hrs.max():.4f}]  val_loss mean={val_losses.mean():.4f} std={val_losses.std():.4f}")

if "real" in all_results and "shuffled" in all_results:
    from scipy import stats
    t, p = stats.ttest_ind(all_results["real"][0], all_results["shuffled"][0])
    print(f"\nreal vs shuffled test_hr: t={t:.3f} p={p:.4f} "
          f"({'REAL SIGNIFICANTLY BETTER - geometry matters now!' if p < 0.05 and all_results['real'][0].mean() > all_results['shuffled'][0].mean() else 'not significant / no clear difference'})")

save_dict = {}
for label, (test_hrs, val_losses) in all_results.items():
    save_dict[f"{label}_test_hr"] = test_hrs
    save_dict[f"{label}_val_loss"] = val_losses
np.savez(f"{HERE}/frozen_projection_geometry_test_results.npz", **save_dict)
print(f"\n[Save] {HERE}/frozen_projection_geometry_test_results.npz")
