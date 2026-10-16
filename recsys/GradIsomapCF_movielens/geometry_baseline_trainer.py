import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Adam.Isomap import IsomapNN


from fast_common import evaluate_topk_vec
from new_datasets import NCFTrainDatasetFutureBlind
from nonparam_baselines import HEADS, build_scorers, history_matrix, with_tiebreak


class GeometryBaselineTrainer:
    def __init__(self, train_feature, train_events, num_users, num_items, user_pos_set, ng_seed=42,
                 latent_len=64, n_neighbors=10, epochs=100, lr_isomap=1e-3, num_ng=2,
                 head="bary_cos", temp=10.0, bias=0.0, knn_k=20, use_pos_weight=True,
                 patience=12, row_chunk=1048576, device="cuda", seed=0, resample_negatives=True):
        if head not in HEADS:
            raise ValueError(f"head должен быть одним из {HEADS}, получено {head!r}")
        self.features = train_feature
        self.num_users, self.num_items = int(num_users), int(num_items)
        self.latent_len, self.n_neighbors = int(latent_len), int(n_neighbors)
        self.epochs, self.lr_isomap, self.num_ng = int(epochs), float(lr_isomap), int(num_ng)
        self.head, self.temp, self.bias, self.knn_k = head, float(temp), float(bias), int(knn_k)
        self.use_pos_weight, self.patience = bool(use_pos_weight), int(patience)
        self.row_chunk, self.device, self.seed = int(row_chunk), torch.device(device), int(seed)
        # True (по умолчанию, как раньше): свежие негативы на каждой эпохе; False: один набор, сэмплированный в конструкторе
        self.resample_negatives = bool(resample_negatives)

        self.train_pairs = np.array([(int(u), int(m)) for (u, m, r) in train_events], dtype=np.int64)
        self.dataset = NCFTrainDatasetFutureBlind(
            features_pos=[(int(u), int(m)) for u, m in self.train_pairs],
            num_items=self.num_items, user_pos_train_set=user_pos_set, num_ng=self.num_ng, seed=ng_seed)
        self.H, _ = history_matrix(self.train_pairs, self.num_users, self.num_items, self.device)
        self.cnt = self.H.sum(1)
        self.history = dict(epoch=[], train_loss=[], val_bce=[], val_bce_candidates=[],
                            val_hr=[], val_ndcg=[], geom_delta_init=[],
                            touched_frac=[], delta_touched=[], knn_churn=[],
                            grad_nonfinite_count=[], n_rows_used=[])

    # ---- логит для обучения (дифференцируем по z) ----
    def _score(self, z, u, i, y):
        if self.head == "item_knn":
            zn = F.normalize(z, dim=1)
            sim = zn @ zn.T
            sim.fill_diagonal_(float("-inf"))
            vals, idx = sim.topk(min(self.knn_k, self.num_items - 1), dim=1)
            in_hist = self.H[u.unsqueeze(1), idx[i]]
            s = (in_hist * vals[i].clamp(min=0.0)).sum(-1)
        else:
            zi = z[i]
            su = (self.H @ z)[u] - zi * y.unsqueeze(1)
            c = (self.cnt[u] - y).unsqueeze(1)
            bary = su / c
            if self.head == "bary_cos":
                s = (F.normalize(bary, dim=1) * F.normalize(zi, dim=1)).sum(-1)
            else:
                s = -(bary - zi).norm(dim=-1)
        return s

    def _logit(self, z, u, i, y):
        return self.temp * self._score(z, u, i, y) + self.bias

    def _rows(self):
        dev = self.device
        u = self.dataset.users_fill.to(dev)
        i = self.dataset.items_fill.to(dev)
        y = self.dataset.labels_fill.to(dev).to(torch.float32)
        keep = torch.ones_like(y, dtype=torch.bool) if self.head == "item_knn" else (self.cnt[u] >= 2)
        return u[keep], i[keep], y[keep]

    @torch.no_grad()
    def _score_stats(self, Z, pos_weight):
        u, i, y = self._rows()
        n = int(y.numel())
        acc = {1: [0.0, 0.0, 0, 0], 0: [0.0, 0.0, 0, 0]}      # sum, sumsq, count, zeros
        loss_sum = 0.0
        for st in range(0, n, self.row_chunk):
            sl = slice(st, min(st + self.row_chunk, n))
            sc = self._score(Z, u[sl], i[sl], y[sl])
            lg = self.temp * sc + self.bias
            loss_sum += float(F.binary_cross_entropy_with_logits(lg, y[sl], reduction="sum", pos_weight=pos_weight))
            for cls in (1, 0):
                m = (y[sl] == cls)
                v = sc[m].double()
                acc[cls][0] += float(v.sum()); acc[cls][1] += float((v * v).sum())
                acc[cls][2] += int(m.sum()); acc[cls][3] += int((v == 0).sum())
        out = {}
        for cls, name in ((1, "pos"), (0, "neg")):
            sm, sq, c, z0 = acc[cls]
            mean = sm / max(c, 1)
            out[name] = dict(mean=mean, std=float(max(sq / max(c, 1) - mean * mean, 0.0) ** 0.5),
                             zero_frac=z0 / max(c, 1), n=c)
        out["suggested_bias"] = -self.temp * 0.5 * (out["pos"]["mean"] + out["neg"]["mean"])
        out["init_bce"] = loss_sum / max(n, 1)
        return out

    @torch.no_grad()
    def _val_bce(self, raw, tensors, chunk_users=1024):
        users, items, labels = tensors
        pw = float(self.num_ng) if self.use_pos_weight else 1.0
        s_pos = s_neg = 0.0
        n_pos = n_neg = 0
        for st in range(0, users.shape[0], chunk_users):
            u = users[st:st + chunk_users].reshape(-1)
            i = items[st:st + chunk_users].reshape(-1)
            lab = labels[st:st + chunk_users].reshape(-1)
            lg = self.temp * raw(u, i) + self.bias
            y = (lab > 0.5).to(torch.float32)
            l = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
            pm, nm = (lab >= 0) & (y == 1), (lab >= 0) & (y == 0)
            s_pos += float(l[pm].sum()); n_pos += int(pm.sum())
            s_neg += float(l[nm].sum()); n_neg += int(nm.sum())
        mean_pos, mean_neg = s_pos / max(n_pos, 1), s_neg / max(n_neg, 1)
        matched = (pw * mean_pos + self.num_ng * mean_neg) / (1.0 + self.num_ng)
        candidates = (pw * s_pos + s_neg) / max(n_pos + n_neg, 1)
        return matched, candidates

    @torch.no_grad()
    def _eval(self, Z, tensors, top_k, with_bce=False):
        raw = build_scorers(Z, self.train_pairs, self.num_users, self.num_items,
                            knn_k=self.knn_k, seed=self.seed, names=[self.head], tiebreak=False)[self.head]
        fn = with_tiebreak(raw, self.seed + HEADS.index(self.head))     # те же ничьи, что и раньше
        hr, ndcg = evaluate_topk_vec(fn, *tensors, top_k, self.device)
        if not with_bce:
            return float(hr), float(ndcg)
        bm, bc = self._val_bce(raw, tensors)
        return float(hr), float(ndcg), float(bm), float(bc)

    def train(self, val_tensors, test_tensors, top_k=10, use_init_assumption=True, dist_type="pos"):
        dev = self.device
        t0 = time.time()
        feats = self.features.to(torch.float32).to(dev)
        if use_init_assumption:
            with torch.no_grad():
                D = torch.cdist(feats, feats)
        else:
            from GradientIsomapCF_fast import GradientIsomapCF as _GI     # только за случайной стартовой D
            D = _GI._generate_random_matrix(self.num_items, dist_type=dist_type, device=dev)
        D = D / D.max().clamp(min=1e-12)
        D0 = D.detach().clone()

        model = IsomapNN(D, n_components=self.latent_len, n_neighbors=self.n_neighbors,
                         eigval_choice="MDS").to(dev)
        opt = optim.AdamW(model.parameters(), lr=self.lr_isomap, weight_decay=0.0)
        with torch.no_grad():
            model.update_distance_matrix()
            D0 = model.distances_matrix.detach().clone()
        pos_weight = torch.tensor([self.num_ng], device=dev, dtype=torch.float32) if self.use_pos_weight else None

        model.eval()
        with torch.no_grad():
            Z0 = model().to(torch.float32)
        init = dict(val=dict(zip(("hr", "ndcg", "bce", "bce_candidates"),
                                 self._eval(Z0, val_tensors, top_k, with_bce=True))),
                    test=dict(zip(("hr", "ndcg"), self._eval(Z0, test_tensors, top_k))))
        print(f"[geom-baseline:{self.head}] init geometry: val HR@{top_k}={init['val']['hr']:.4f}, "
              f"val BCE={init['val']['bce']:.4f}, test HR@{top_k}={init['test']['hr']:.4f}")

        stats = self._score_stats(Z0, pos_weight)
        print(f"[geom-baseline:{self.head}] скоры на train-строках (init): "
              f"позитивы mean={stats['pos']['mean']:.3f} std={stats['pos']['std']:.3f} (нулевых {stats['pos']['zero_frac']:.1%}); "
              f"негативы mean={stats['neg']['mean']:.3f} std={stats['neg']['std']:.3f} (нулевых {stats['neg']['zero_frac']:.1%}); "
              f"BCE при temp={self.temp}, bias={self.bias}: {stats['init_bce']:.4f}; "
              f"bias для порога посередине между классами (--np-bias): {stats['suggested_bias']:.3f}")
        knn0 = torch.topk(D0, self.n_neighbors, dim=1, largest=False).indices

        best_hr, best_state, best_epoch, no_improve = -np.inf, None, None, 0
        for epoch in range(self.epochs):
            ts = time.time()
            model.train()
            opt.zero_grad()
            Z = model().to(torch.float32)
            Zd = Z.detach()

            if self.resample_negatives:
                self.dataset.ng_sample()
            u_all, i_all, y_all = self._rows()
            n_rows = int(y_all.numel())
            if n_rows == 0:
                raise RuntimeError("нет строк для обучения (у всех пользователей |H_u| < 2)")

            grad_accum = torch.zeros_like(Zd)
            loss_sum = 0.0
            for s in range(0, n_rows, self.row_chunk):
                sl = slice(s, min(s + self.row_chunk, n_rows))
                z = Zd.clone().requires_grad_(True)
                logit = self._logit(z, u_all[sl], i_all[sl], y_all[sl])
                lsum = F.binary_cross_entropy_with_logits(logit, y_all[sl], reduction="sum",
                                                          pos_weight=pos_weight)
                g, = torch.autograd.grad(lsum / n_rows, z)
                grad_accum += g
                loss_sum += float(lsum.item())
            Z.backward(gradient=grad_accum)

            bad = 0
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        bad += int((~torch.isfinite(p.grad)).sum().item())
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()

            with torch.no_grad():
                model.update_distance_matrix()
                Z_new = model().to(torch.float32)
                Dcur = model.distances_matrix.detach()
                diff = Dcur - D0
                delta0 = float(torch.norm(diff).item() / max(float(torch.norm(D0).item()), 1e-12))
                touched = diff.abs() > 1e-12                       # ячейки D, которые вообще сдвинулись
                touched_frac = float(touched.float().mean().item())
                delta_touched = float(torch.norm(diff[touched]).item()
                                      / max(float(torch.norm(D0[touched]).item()), 1e-12)) if touched.any() else 0.0
                knn_now = torch.topk(Dcur, self.n_neighbors, dim=1, largest=False).indices
                same = (knn_now.unsqueeze(2) == knn0.unsqueeze(1)).any(2).float().mean()
                knn_churn = float(1.0 - same.item())               # доля соседей в kNN-графе, изменившихся с начала
            hr_val, ndcg_val, bce_val, bce_val_c = self._eval(Z_new, val_tensors, top_k, with_bce=True)

            h = self.history
            h["epoch"].append(epoch)
            h["train_loss"].append(loss_sum / n_rows)
            h["val_bce"].append(bce_val)
            h["val_bce_candidates"].append(bce_val_c)
            h["val_hr"].append(hr_val)
            h["val_ndcg"].append(ndcg_val)
            h["geom_delta_init"].append(delta0)
            h["touched_frac"].append(touched_frac)
            h["delta_touched"].append(delta_touched)
            h["knn_churn"].append(knn_churn)
            h["grad_nonfinite_count"].append(bad)
            h["n_rows_used"].append(n_rows)
            print(f"[geom-baseline:{self.head}] epoch {epoch + 1}/{self.epochs} | train BCE={loss_sum / n_rows:.4f} "
                  f"| val BCE={bce_val:.4f} HR@{top_k}={hr_val:.4f} NDCG@{top_k}={ndcg_val:.4f} | delta_init={delta0:.3e} "
                  f"(тронуто {touched_frac:.2%} ячеек, delta_touched={delta_touched:.3f}, knn_churn={knn_churn:.3f}) "
                  f"| rows={n_rows} | {time.time() - ts:.1f}s")

            if hr_val > best_hr:
                best_hr, best_epoch, no_improve = hr_val, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"[geom-baseline:{self.head}] early stop: нет роста val HR {self.patience} эпох подряд")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            Z_final = model().to(torch.float32).detach()
        res = dict(
            head=self.head, temp=self.temp, bias=self.bias, knn_k=self.knn_k, trainable_head_params=0,
            best_epoch=best_epoch, epochs_run=len(self.history["epoch"]), total_time_s=time.time() - t0,
            init_geometry=init, init_score_stats=stats,
            trained_geometry=dict(
                val=dict(zip(("hr", "ndcg", "bce", "bce_candidates"),
                             self._eval(Z_final, val_tensors, top_k, with_bce=True))),
                test=dict(zip(("hr", "ndcg"), self._eval(Z_final, test_tensors, top_k)))),
        )
        return model, Z_final, res


def plot_geometry_baseline(history, res, save_path, top_k=10):
    """Кривые обучения режима 'геометрия + непараметрическая голова': train BCE, val HR/NDCG, сдвиг геометрии."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ep = [e + 1 for e in history["epoch"]]
    best = res.get("best_epoch")
    init_val = res["init_geometry"]["val"]
    fig, ax = plt.subplots(1, 4, figsize=(21, 4))

    ax[0].plot(ep, history["train_loss"], marker="o", ms=3, color="tab:blue", label="train BCE")
    ax[0].plot(ep, history["val_bce"], marker="o", ms=3, color="tab:red",
               label="val BCE")
    ax[0].axhline(init_val["bce"], ls="--", color="tab:red", alpha=0.5, label="val BCE на начальной геометрии")
    ax[0].set_title(f"BCE (голова {res['head']}, temp={res['temp']}, bias={res['bias']})")
    ax[1].plot(ep, history["val_hr"], marker="o", ms=3, color="tab:blue")
    ax[1].axhline(init_val["hr"], ls="--", color="gray", label="начальная геометрия (без обучения)")
    ax[1].set_title(f"val HR@{top_k}")
    ax[2].plot(ep, history["val_ndcg"], marker="o", ms=3, color="tab:orange")
    ax[2].axhline(init_val["ndcg"], ls="--", color="gray", label="начальная геометрия (без обучения)")
    ax[2].set_title(f"val NDCG@{top_k}")
    ax[3].plot(ep, history["delta_touched"], label="delta_touched: сдвиг тронутых ячеек D", color="tab:red")
    ax[3].plot(ep, history["geom_delta_init"], label="delta_init: по всей матрице D", color="tab:purple")
    ax[3].plot(ep, history["knn_churn"], label="knn_churn: доля сменившихся соседей", color="tab:green")
    ax[3].set_title("сдвиг геометрии от старта")
    for a in ax:
        a.set_xlabel("эпоха")
        a.grid(True, alpha=0.3)
        if best is not None:
            a.axvline(best + 1, ls=":", color="k", alpha=0.6)
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("DiffIsomap + непараметрическая голова (NCF не обучается); пунктир чёрным - лучшая эпоха по val HR")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)