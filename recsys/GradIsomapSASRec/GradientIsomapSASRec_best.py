"""
GradientIsomapSASRec: bilevel optimization with SASRec inner loop.

Follows GradientIsomapCF_log.py pattern from mnist_reg_example exactly, adapted for
sequential recommendation:
  - nan_to_num_ on gradients (not rollback) —  exact guard
  - select_by="hr" for early stopping and checkpoint selection
  - copy.deepcopy for all checkpoints
  - best outer isomap state tracked by select_by criterion, restored
    before final retrain
  - warm_start_inner: carry SASRec weights across outer steps
  - freeze_item_projection: fixed PCA item projection each outer step,
    item_projection_init_data = current item_Z (recomputed fresh every
    outer step, tracking the evolving manifold — same as NeuMFOnManifold)
  - had_bad_grad / grad_nonfinite_count in history
  - DataLoader shuffle generator seeded
  - single return (best isomap + final sasrec)
"""
import copy
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from Adam.Isomap import IsomapNN
from SASRecOnManifold_best import SASRecOnManifold, pad_item_Z
from sasrec_manifold_sampler import SASRecManifoldTrainSampler
from evaluation_sasrec_manifold import evaluate_topk_sasrec_isomap


# ──────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────

def _build_knn_adjacency(D_input: np.ndarray, k: int) -> np.ndarray:
    n = D_input.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(D_input[i])[1: k + 1]
        for nb in neighbors:
            adj[i, nb] = D_input[i, nb]
            adj[nb, i] = D_input[nb, i]
    return adj


def save_epoch_matrices(logs_folder, epoch, isomap_model, device):
    with torch.no_grad():
        D_input = isomap_model.distances_matrix.detach().cpu().numpy().astype(np.float32)
        D_geodesic = isomap_model.dist_matrix_.detach().cpu().numpy().astype(np.float32)
        Z = isomap_model.embedding_.detach().cpu().numpy().astype(np.float32)
    knn_adj = _build_knn_adjacency(D_input, isomap_model.n_neighbors)
    Z_t = torch.tensor(Z, dtype=torch.float32)
    D_latent = torch.cdist(Z_t, Z_t).numpy().astype(np.float32)
    save_path = os.path.join(logs_folder, f"matrices_epoch{epoch}.npz")
    np.savez_compressed(save_path, D_input=D_input, D_geodesic=D_geodesic,
                        knn_adj=knn_adj, Z=Z, D_latent=D_latent)
    print(f"[Save] Epoch {epoch + 1}: matrices → {save_path}", flush=True)


def save_history(logs_folder, history, filename="history.npz"):
    cleaned = {
        k: np.array([v if v is not None else np.nan for v in vals], dtype=np.float32)
        for k, vals in history.items()
    }
    np.savez_compressed(os.path.join(logs_folder, filename), **cleaned)
    print(f"[Save] History → {os.path.join(logs_folder, filename)}")


# ──────────────────────────────────────────────────────────
#  EarlyStopping
# ──────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Mirrors GradientIsomapCF_log.EarlyStopping.

    select_by="hr"   — stop by plain HR@10 patience (recommended).
                       Bypasses loss-based convergence/overfit-gap logic,
                       which is unreliable when train/val have different
                       negative-sampling ratios (see diary note).
    select_by="loss" — original convergence/overfit-gap logic.
    """
    def __init__(
        self,
        patience: int = 10,
        window: int = 5,
        smooth_window: int = 5,
        min_delta: float = 1e-4,
        convergence_delta: float = 0.005,
        overfit_gap: float = 0.05,
        overfit_patience: int = 5,
        select_by: str = "hr",
    ):
        self.patience = patience
        self.window = window
        self.smooth_window = smooth_window
        self.min_delta = min_delta
        self.convergence_delta = convergence_delta
        self.overfit_gap = overfit_gap
        self.overfit_patience = overfit_patience
        self.select_by = select_by
        self._no_improve_hr = 0

        self.best_val_loss = np.inf
        self.best_state_global = None
        self.best_epoch_global = -1

        self.best_val_in_window = np.inf
        self.best_state_window = None
        self.best_epoch_window = -1

        self.best_val_hr = -np.inf
        self.best_state_window_hr = None
        self.best_epoch_window_hr = -1

        self._no_improve = 0
        self._overfit_streak = 0
        self._val_history = []
        self._train_history = []
        self._window_states = []
        self._epoch = 0

    def _smooth(self, history):
        return float(np.mean(history[-self.smooth_window:]))

    def step(self, train_loss, val_loss, model, val_hr=None):
        self._val_history.append(val_loss)
        self._train_history.append(train_loss)
        epoch = self._epoch
        self._epoch += 1

        # deepcopy — fix for the bare state_dict reference bug
        state_copy = copy.deepcopy(model.state_dict())

        self._window_states.append((epoch, val_loss, state_copy))
        while len(self._window_states) > self.window:
            self._window_states.pop(0)
        best_in_window = min(self._window_states, key=lambda x: x[1])
        self.best_epoch_window = best_in_window[0]
        self.best_val_in_window = best_in_window[1]
        self.best_state_window = best_in_window[2]

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_state_global = state_copy
            self.best_epoch_global = epoch
            self._no_improve = 0
        else:
            self._no_improve += 1

        if val_hr is not None:
            if val_hr > self.best_val_hr + self.min_delta:
                self.best_val_hr = val_hr
                self.best_epoch_window_hr = epoch
                self.best_state_window_hr = state_copy
                self._no_improve_hr = 0
            else:
                self._no_improve_hr += 1

        if self.select_by == "hr":
            return val_hr is not None and self._no_improve_hr >= self.patience

        enough = len(self._val_history) >= self.smooth_window
        smooth_val = self._smooth(self._val_history)
        smooth_train = self._smooth(self._train_history)
        smooth_gap = smooth_val - smooth_train
        cond_converged = enough and (abs(smooth_gap) <= self.convergence_delta)
        if enough and smooth_gap > self.overfit_gap:
            self._overfit_streak += 1
        else:
            self._overfit_streak = 0
        cond_overfit = self._overfit_streak >= self.overfit_patience
        return (cond_converged or cond_overfit) and self._no_improve >= self.patience

    def restore_best_window(self, model):
        if self.best_state_window is not None:
            model.load_state_dict(self.best_state_window)
            print(f"  [EarlyStopping] window best: "
                  f"ep {self.best_epoch_window + 1}, val={self.best_val_in_window:.4f}")

    def restore_best_hr(self, model):
        if self.best_state_window_hr is not None:
            model.load_state_dict(self.best_state_window_hr)
            print(f"  [EarlyStopping] HR@10 best: "
                  f"ep {self.best_epoch_window_hr + 1}, hr={self.best_val_hr:.4f}")


# ──────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────

class GradientIsomapSASRec:
    """
    Bilevel optimization: outer loop trains IsomapNN geometry via SASRec
    gradient signal; inner loop trains SASRecOnManifold on fixed Z.

    All parameters mirror GradientIsomapCF.__init__() naming/semantics,
    adapted for sequential recommendation.
    """

    def __init__(
        self,
        train_feature: torch.Tensor,    # [num_items, feature_dim]
        user_train: dict,               # {uid: [item1, item2, ...]} TRAIN ONLY
        num_users: int,
        num_items: int,
        user_pos_set=None,              # for negative exclusion
        ng_seed: int = 42,
        # Isomap
        latent_len: int = 64,
        n_neighbors: int = 10,
        # Outer loop
        epochs: int = 30,
        lr_isomap: float = 5e-3,
        # Inner SASRec
        sasrec_config: dict = None,
        cf_epochs: int = 30,
        final_cf_epochs: int = 30,
        lr_sasrec: float = 1e-3,
        batch_size: int = 256,
        # Stopping
        select_by: str = "hr",
        inner_patience: int = 5,
        final_patience: int = 3,
        stop_criteria_value: float = 0.001,
        # Options 
        warm_start_inner: bool = False,
        freeze_item_projection: bool = False,
        # Misc
        logs_folder: str = None,
        device: str = None,
    ):
        self.train_feature = train_feature
        self.user_train = user_train
        self.num_users = num_users
        self.num_items = num_items
        self.user_pos_set = user_pos_set
        self.ng_seed = ng_seed

        self.latent_len = latent_len
        self.n_neighbors = n_neighbors
        self.epochs = epochs
        self.lr_isomap = lr_isomap

        self.sasrec_config = sasrec_config or {
            'hidden_units': 64, 'maxlen': 50, 'num_blocks': 2,
            'num_heads': 1, 'dropout_rate': 0.2, 'l2_emb': 0.0,
        }
        self.cf_epochs = cf_epochs
        self.final_cf_epochs = final_cf_epochs
        self.lr_sasrec = lr_sasrec
        self.batch_size = batch_size

        self.select_by = select_by
        self.inner_patience = inner_patience
        self.final_patience = final_patience
        self.stop_criteria_value = stop_criteria_value

        # warm_start_inner: mirrors NeuMFOnManifold's warm-start option.
        # Carry SASRec weights across outer steps instead of reinitialising.
        self.warm_start_inner = warm_start_inner
        self._warm_sasrec = None

        # freeze_item_projection: mirrors NeuMFOnManifold.freeze_item_projection.
        # PCA projection recomputed fresh each outer step from current item_Z.
        self.freeze_item_projection = freeze_item_projection

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if logs_folder is None:
            logs_folder = f"gi_sasrec_{datetime.now().strftime('%d%m%Y-%H.%M')}"
        self.logs_folder = logs_folder
        os.makedirs(self.logs_folder, exist_ok=True)
        print(f"Device: {self.device} | Logs: {self.logs_folder}")

        # ── Training sampler ──
        self.train_sampler = SASRecManifoldTrainSampler(
            user_train=user_train,
            n_items=num_items,
            maxlen=self.sasrec_config['maxlen'],
            user_pos_train_set=user_pos_set or {},
            seed=ng_seed,
        )
        # Seeded generator — mirrors DataLoader seed fix
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(ng_seed)
        self.train_loader = DataLoader(
            self.train_sampler, batch_size=batch_size,
            num_workers=0, generator=shuffle_gen,
        )

        n_users_with_seq = len([u for u, s in user_train.items() if len(s) >= 2])
        self.n_batches_inner = max(1, n_users_with_seq // batch_size)

        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'val_hr': [], 'val_ndcg': [],
            'had_bad_grad': [], 'grad_nonfinite_count': [],
        }
        self.cf_history = {
            'train_loss': [], 'val_loss': [], 'val_hr': [],
        }

    def _make_sasrec(
        self,
        item_Z_for_init: "torch.Tensor | None" = None,
    ) -> SASRecOnManifold:
        """
        Create a new SASRecOnManifold.
        item_Z_for_init is passed as item_projection_init_data when
        freeze_item_projection=True — mirrors NeuMFOnManifold's pattern
        where PCA is recomputed fresh from the current outer-step Z.
        """
        return SASRecOnManifold(
            config=self.sasrec_config,
            item_num=self.num_items,
            latent_dim=self.latent_len,
            freeze_item_projection=self.freeze_item_projection,
            item_projection_init_data=item_Z_for_init,
        ).to(self.device)

    def train(self, val_loader=None, top_k: int = 10, device=None):
        if device is None:
            device = self.device
        elif isinstance(device, str):
            device = torch.device(device)

        start_time = time.time()
        loss_fn = nn.BCEWithLogitsLoss()

        # ── D_input initialisation (identical to GradientIsomapCF) ──
        self.train_feature = self.train_feature.to(torch.float32).to(self.device)
        with torch.no_grad():
            dist_train = torch.cdist(self.train_feature, self.train_feature)
        max_val = dist_train.max()
        if max_val > 0:
            dist_train = dist_train / max_val

        np.save(os.path.join(self.logs_folder, "D_input_init.npy"),
                dist_train.cpu().numpy().astype(np.float32))
        print(f"[Save] D_input_init.npy")

        isomap_model = IsomapNN(
            dist_train,
            n_components=self.latent_len,
            n_neighbors=self.n_neighbors,
            eigval_choice='MDS',
        ).to(self.device)
        isomap_optim = optim.AdamW(isomap_model.parameters(), lr=self.lr_isomap)

        pad_token = self.num_items
        best_val_loss = np.inf
        best_val_hr_outer = -np.inf
        best_isomap_state = None

        batch_iter = iter(self.train_loader)

        # ════════════════════════════════════════════════
        #  OUTER LOOP
        # ════════════════════════════════════════════════
        for epoch in range(self.epochs):
            epoch_start = time.time()

            # ── Fix Z for inner loop ──
            isomap_model.eval()
            with torch.no_grad():
                item_Z_epoch_raw = isomap_model().to(torch.float32).detach()
            item_Z_epoch = pad_item_Z(item_Z_epoch_raw)  # [num_items+1, latent_dim]

            # ── Inner SASRec ──
            if self.warm_start_inner and self._warm_sasrec is not None:
                # Carry weights from previous outer step.
                # Re-init item_projection if freezing (new Z → new PCA).
                sasrec = self._warm_sasrec
                if self.freeze_item_projection:
                    # Recompute PCA from current Z (same as NCF does in outer loop)
                    sasrec._init_weights(self.latent_len, item_Z_epoch_raw)
                for p in sasrec.parameters():
                    p.requires_grad_(True)
                if self.freeze_item_projection:
                    sasrec.item_projection.weight.requires_grad_(False)
                    sasrec.item_projection.bias.requires_grad_(False)
                sasrec.train()
            else:
                # Fresh model each outer step — pass current Z for PCA init
                sasrec = self._make_sasrec(
                    item_Z_for_init=item_Z_epoch_raw if self.freeze_item_projection else None
                )

            # Fresh optimizer every step even under warm_start_inner —
            # only SASRec weights carry over, not Adam momentum state.
            # Mirrors GradientIsomapCF's comment on this choice exactly.
            sasrec_optim = optim.Adam(
                [p for p in sasrec.parameters() if p.requires_grad],
                lr=self.lr_sasrec, betas=(0.9, 0.98),
            )

            early_stop = EarlyStopping(
                patience=self.inner_patience,
                window=6,
                smooth_window=2,
                min_delta=1e-4,
                convergence_delta=0.001,
                overfit_gap=0.003,
                overfit_patience=1,
                select_by=self.select_by,
            )

            cf_train_losses, cf_val_losses, cf_val_hrs = [], [], []
            l2_emb = self.sasrec_config.get('l2_emb', 0.0)

            sasrec.train()
            for cf_ep in range(self.cf_epochs):
                total_loss = 0.0

                for _ in range(self.n_batches_inner):
                    seq, pos, neg = next(batch_iter)
                    seq = seq.to(self.device)
                    pos = pos.to(self.device)
                    neg = neg.to(self.device)

                    pos_logits, neg_logits = sasrec(seq, pos, neg, item_Z_epoch)
                    indices = torch.where(pos != pad_token)
                    pos_labels = torch.ones(pos_logits.shape, device=self.device)
                    neg_labels = torch.zeros(neg_logits.shape, device=self.device)

                    batch_loss = loss_fn(pos_logits[indices], pos_labels[indices])
                    batch_loss += loss_fn(neg_logits[indices], neg_labels[indices])

                    if l2_emb != 0:
                        # L2 only on trainable item_projection params
                        for p in sasrec.item_projection.parameters():
                            if p.requires_grad:
                                batch_loss += l2_emb * torch.norm(p) ** 2

                    sasrec_optim.zero_grad()
                    batch_loss.backward()
                    sasrec_optim.step()
                    total_loss += batch_loss.item()

                avg_train = total_loss / self.n_batches_inner
                cf_train_losses.append(avg_train)

                if val_loader is not None:
                    sasrec.eval()
                    total_vl, n_vl = 0.0, 0
                    hits = []
                    with torch.no_grad():
                        for v_seq, v_cands, v_labels in val_loader:
                            v_seq = v_seq.to(self.device)
                            v_cands = v_cands.to(self.device)
                            v_labels = v_labels.to(self.device)
                            v_logits = sasrec.predict_candidates(v_seq, v_cands, item_Z_epoch)
                            total_vl += loss_fn(v_logits, v_labels).item()
                            n_vl += 1
                            # Inline HR@10 — mirrors inner-loop val
                            for b in range(v_logits.shape[0]):
                                _, topk_idx = torch.topk(v_logits[b], 10)
                                hits.append(1.0 if 0 in topk_idx.tolist() else 0.0)
                    avg_vl = total_vl / max(1, n_vl)
                    avg_hr = float(np.mean(hits)) if hits else 0.0
                    cf_val_losses.append(avg_vl)
                    cf_val_hrs.append(avg_hr)

                    print(f"  [Inner SASRec] ep {cf_ep+1}/{self.cf_epochs} | "
                          f"train={avg_train:.4f} val={avg_vl:.4f} "
                          f"val_hr@10={avg_hr:.4f}", flush=True)

                    if early_stop.step(avg_train, avg_vl, sasrec, val_hr=avg_hr):
                        break
                    sasrec.train()
                else:
                    cf_val_losses.append(None)
                    cf_val_hrs.append(None)
                    print(f"  [Inner SASRec] ep {cf_ep+1}/{self.cf_epochs} | "
                          f"train={avg_train:.4f}", flush=True)

            # Restore best checkpoint by select_by criterion
            if self.select_by == "hr":
                early_stop.restore_best_hr(sasrec)
            else:
                early_stop.restore_best_window(sasrec)

            self.cf_history['train_loss'].append(cf_train_losses)
            self.cf_history['val_loss'].append(cf_val_losses)
            self.cf_history['val_hr'].append(cf_val_hrs)

            # ── Freeze SASRec for outer step ──
            sasrec.eval()
            for p in sasrec.parameters():
                p.requires_grad_(False)

            if self.warm_start_inner:
                self._warm_sasrec = sasrec

            # ── Outer step: update D_input ──
            isomap_model.train()
            isomap_optim.zero_grad()

            item_Z_full_raw = isomap_model().to(torch.float32)
            item_Z_full = pad_item_Z(item_Z_full_raw)

            # Outer loss over a few batches (same n=4 as mnist_reg_example pattern)
            outer_loss = torch.tensor(0.0, device=self.device)
            n_outer = 4
            for _ in range(n_outer):
                seq, pos, neg = next(batch_iter)
                seq, pos, neg = seq.to(self.device), pos.to(self.device), neg.to(self.device)
                pos_logits, neg_logits = sasrec(seq, pos, neg, item_Z_full)
                indices = torch.where(pos != pad_token)
                b_loss = (
                    loss_fn(pos_logits[indices],
                            torch.ones(len(indices[0]), device=self.device))
                    + loss_fn(neg_logits[indices],
                              torch.zeros(len(indices[0]), device=self.device))
                )
                outer_loss = outer_loss + b_loss
            outer_loss = outer_loss / n_outer
            outer_loss.backward()

            # ══════════════════════════════════════════════════════════
            #  NaN-guard 
            #  scan grads, zero non-finite entries via nan_to_num_,
            #  then step() unconditionally with the cleaned grads.
            #  NOT a rollback: finite grad components still update D_input.
            # ══════════════════════════════════════════════════════════
            had_bad_grad = False
            grad_nonfinite_count = 0
            for group in isomap_optim.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        nonfinite_mask = ~torch.isfinite(p.grad)
                        if nonfinite_mask.any():
                            had_bad_grad = True
                            grad_nonfinite_count += int(nonfinite_mask.sum().item())
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if had_bad_grad:
                print(f"  [Guard] Outer epoch {epoch}: {grad_nonfinite_count} "
                      f"non-finite gradient entries zeroed before step()", flush=True)

            isomap_optim.step()

            with torch.no_grad():
                isomap_model.update_distance_matrix()
                _ = isomap_model()

            save_epoch_matrices(self.logs_folder, epoch, isomap_model, self.device)

            avg_train_loss = float(outer_loss.item())
            elapsed = time.time() - epoch_start

            # ── Outer validation ──
            hr_val, ndcg_val, avg_val_loss = None, None, None
            if val_loader is not None:
                isomap_model.eval()
                hr_val, ndcg_val = evaluate_topk_sasrec_isomap(
                    sasrec, isomap_model, val_loader, top_k, self.device
                )
                avg_val_loss = avg_train_loss  # approximate

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_hr'].append(hr_val)
            self.history['val_ndcg'].append(ndcg_val)
            self.history['had_bad_grad'].append(had_bad_grad)
            self.history['grad_nonfinite_count'].append(grad_nonfinite_count)

            print(
                f"[Outer {epoch+1}/{self.epochs}] "
                f"train={avg_train_loss:.4f}, "
                f"val={avg_val_loss if avg_val_loss is not None else float('nan'):.4f}, "
                f"HR@{top_k}={hr_val if hr_val is not None else float('nan'):.4f}, "
                f"NDCG@{top_k}={ndcg_val if ndcg_val is not None else float('nan'):.4f}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )

            stop_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss

            # ── Best outer state tracking ──
            if self.select_by == "hr":
                outer_improved = hr_val is not None and hr_val > best_val_hr_outer
            else:
                outer_improved = stop_loss < best_val_loss
            if outer_improved:
                if self.select_by == "hr":
                    best_val_hr_outer = hr_val
                best_isomap_state = copy.deepcopy(isomap_model.state_dict())
            if stop_loss < best_val_loss:
                best_val_loss = stop_loss

            if stop_loss <= self.stop_criteria_value:
                print(f"Stop criteria: loss={stop_loss:.4f} <= {self.stop_criteria_value}")
                break

            if not self.warm_start_inner:
                del sasrec, sasrec_optim
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"\nOuter loop finished in "
              f"{time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"Best outer val/train loss = {best_val_loss:.4f}")

        # ════════════════════════════════════════════════
        #  Restore best outer geometry
        # ════════════════════════════════════════════════
        if best_isomap_state is not None:
            isomap_model.load_state_dict(best_isomap_state)
            if self.select_by == "hr":
                print(f"[Outer] Restored best-by-HR@10 geometry "
                      f"(val_hr={best_val_hr_outer:.4f}), not the last step's.")
            else:
                print(f"[Outer] Restored best-by-loss geometry "
                      f"(val_loss={best_val_loss:.4f}), not the last step's.")

        isomap_model.eval()
        with torch.no_grad():
            item_Z_final_raw = isomap_model().to(torch.float32).detach()
        item_Z_final = pad_item_Z(item_Z_final_raw)

        # ════════════════════════════════════════════════
        #  Final SASRec retrain on best Z
        # ════════════════════════════════════════════════
        final_sasrec = self._make_sasrec(
            item_Z_for_init=item_Z_final_raw if self.freeze_item_projection else None
        )
        final_optim = optim.Adam(
            [p for p in final_sasrec.parameters() if p.requires_grad],
            lr=self.lr_sasrec, betas=(0.9, 0.98),
        )

        best_val_loss_final = np.inf
        best_val_hr_final = -np.inf
        best_state_final = None
        no_improve_final = 0

        for ep in range(self.final_cf_epochs):
            final_sasrec.train()
            total_loss = 0.0

            for _ in range(self.n_batches_inner):
                seq, pos, neg = next(batch_iter)
                seq, pos, neg = seq.to(self.device), pos.to(self.device), neg.to(self.device)
                pos_logits, neg_logits = final_sasrec(seq, pos, neg, item_Z_final)
                indices = torch.where(pos != pad_token)
                b_loss = (
                    loss_fn(pos_logits[indices],
                            torch.ones(len(indices[0]), device=self.device))
                    + loss_fn(neg_logits[indices],
                              torch.zeros(len(indices[0]), device=self.device))
                )
                final_optim.zero_grad()
                b_loss.backward()
                final_optim.step()
                total_loss += b_loss.item()

            avg_train = total_loss / self.n_batches_inner

            if val_loader is not None:
                final_sasrec.eval()
                total_vl, n_vl = 0.0, 0
                hits = []
                with torch.no_grad():
                    for v_seq, v_cands, v_labels in val_loader:
                        v_seq = v_seq.to(self.device)
                        v_cands = v_cands.to(self.device)
                        v_labels = v_labels.to(self.device)
                        v_logits = final_sasrec.predict_candidates(v_seq, v_cands, item_Z_final)
                        total_vl += loss_fn(v_logits, v_labels).item()
                        n_vl += 1
                        for b in range(v_logits.shape[0]):
                            _, topk_idx = torch.topk(v_logits[b], 10)
                            hits.append(1.0 if 0 in topk_idx.tolist() else 0.0)

                avg_vl = total_vl / max(1, n_vl)
                avg_hr = float(np.mean(hits)) if hits else 0.0

                print(f"[Final SASRec] ep {ep+1}/{self.final_cf_epochs} | "
                      f"train={avg_train:.4f}, val={avg_vl:.4f}, "
                      f"val_hr@10={avg_hr:.4f}", flush=True)

                # select_by criterion for final checkpoint (with deepcopy fix applied)
                improved = (avg_hr > best_val_hr_final) if self.select_by == "hr" \
                    else (avg_vl < best_val_loss_final)
                if improved:
                    best_val_loss_final = avg_vl
                    best_val_hr_final = avg_hr
                    # deepcopy — fix for bare state_dict reference bug
                    best_state_final = copy.deepcopy(final_sasrec.state_dict())
                    no_improve_final = 0
                else:
                    no_improve_final += 1

                if no_improve_final >= self.final_patience:
                    print(f"[Final SASRec] early stop at ep {ep+1}")
                    break
                if avg_vl <= self.stop_criteria_value:
                    print(f"[Final SASRec] stop by threshold: val={avg_vl:.4f}")
                    break
            else:
                print(f"[Final SASRec] ep {ep+1}/{self.final_cf_epochs} | "
                      f"train={avg_train:.4f}", flush=True)

        if best_state_final is not None:
            final_sasrec.load_state_dict(best_state_final)

        self.isomap_model = isomap_model
        self.sasrec_model = final_sasrec

        torch.save(isomap_model.state_dict(),
                   os.path.join(self.logs_folder, "isomap_model_final.pt"))
        torch.save(final_sasrec.state_dict(),
                   os.path.join(self.logs_folder, "sasrec_model_final.pt"))
        print(f"[Save] Models saved → {self.logs_folder}")

        save_history(self.logs_folder, self.history, "history.npz")
        with open(os.path.join(self.logs_folder, "cf_history.json"), "w") as f:
            json.dump(self.cf_history, f, indent=4)

        return self.isomap_model, self.sasrec_model
    