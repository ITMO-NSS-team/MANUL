import torch
import torch.nn as nn
import torch.optim as optim

import copy
import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from Adam.Isomap_new import IsomapNN
import Adam.Isomap_new as IsomapModule
from NeuMFOnManifold_new import NeuMFOnManifold
from new_datasets2 import NCFTrainDatasetFutureBlind
from fast_common import evaluate_topk_vec, val_loss_vec

NCF_ADAMW_WEIGHT_DECAY = 1e-5


class EarlyStopping:
    def __init__(
            self,
            patience: int = 10,
            window: int = 5,
            smooth_window: int = 5,
            min_delta: float = 1e-4,
            convergence_delta: float = 0.005,
            overfit_gap: float = 0.05,
            overfit_patience: int = 5,
            select_by: str = "loss",
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
        self._val_hr_history = []
        self._window_states = []
        self._epoch = 0

    def _smooth(self, history: list) -> float:
        window = history[-self.smooth_window:]
        return float(np.mean(window))

    def step(self, train_loss: float, val_loss: float, model: nn.Module, val_hr: float = None) -> bool:
        self._val_history.append(val_loss)
        self._train_history.append(train_loss)
        epoch = self._epoch
        self._epoch += 1

        state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
            self._val_hr_history.append(val_hr)
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

        if enough and (smooth_gap > self.overfit_gap):
            self._overfit_streak += 1
        else:
            self._overfit_streak = 0

        cond_overfit = self._overfit_streak >= self.overfit_patience

        cond_no_improve = self._no_improve >= self.patience
        should_stop = (cond_converged or cond_overfit) and cond_no_improve

        return should_stop

    def restore_best_global(self, model: nn.Module):
        if self.best_state_global is not None:
            model.load_state_dict(self.best_state_global)

    def restore_best_window(self, model: nn.Module):
        if self.best_state_window is not None:
            model.load_state_dict(self.best_state_window)

    def restore_best_hr(self, model: nn.Module):
        if self.best_state_window_hr is not None:
            model.load_state_dict(self.best_state_window_hr)

    @property
    def converged(self) -> bool:
        if len(self._val_history) < self.smooth_window:
            return False
        smooth_gap = self._smooth(self._val_history) - self._smooth(self._train_history)
        return abs(smooth_gap) <= self.convergence_delta

    @property
    def history(self) -> dict:
        gaps = [abs(t - v) for t, v in zip(self._train_history, self._val_history)]
        return {
            'train': self._train_history.copy(),
            'val': self._val_history.copy(),
            'gap': gaps,
        }


def save_epoch_matrices(
        logs_folder: str,
        epoch: int,
        isomap_model,
):
    with torch.no_grad():
        D_input = isomap_model.distances_matrix.detach().cpu().numpy().astype(np.float32)
        D_geodesic = isomap_model.dist_matrix_.detach().cpu().numpy().astype(np.float32)
        Z = isomap_model.embedding_.detach().cpu().numpy().astype(np.float32)

    knn_adj = _build_knn_adjacency(D_input, isomap_model.n_neighbors)

    cdist_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Z_tensor = torch.tensor(Z, dtype=torch.float32, device=cdist_device)
    D_latent = torch.cdist(Z_tensor, Z_tensor).cpu().numpy().astype(np.float32)

    save_path = os.path.join(logs_folder, f"matrices_epoch{epoch}.npz")
    np.savez_compressed(
        save_path,
        D_input=D_input,
        D_geodesic=D_geodesic,
        knn_adj=knn_adj,
        Z=Z,
        D_latent=D_latent,
    )
    print(f"[Save] Epoch {epoch + 1}: matrices saved -> {save_path}")


def _build_knn_adjacency(D_input: np.ndarray, k: int) -> np.ndarray:
    n = D_input.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neigh = np.argpartition(D_input[i], k + 1)[:k + 1]
        neigh = neigh[neigh != i][:k]
        adj[i, neigh] = D_input[i, neigh]
        adj[neigh, i] = D_input[neigh, i]
    return adj


def save_history(logs_folder: str, history: dict, filename: str = "history.npz"):
    save_path = os.path.join(logs_folder, filename)

    cleaned = {}
    for key, values in history.items():
        cleaned[key] = np.array(
            [v if v is not None else np.nan for v in values],
            dtype=np.float32
        )

    np.savez_compressed(save_path, **cleaned)


def _knn_indices(D: torch.Tensor, k: int) -> torch.Tensor:
    n = D.size(0)
    Df = D.clone()
    Df.fill_diagonal_(float('inf'))
    kk = min(k, n - 1)
    return torch.topk(Df, kk, dim=1, largest=False).indices


def _knn_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    s, _ = torch.cat([A, B], dim=1).sort(dim=1)
    inter = (s[:, 1:] == s[:, :-1]).sum(dim=1)
    return float((inter.float() / A.size(1)).mean().item())


class GradientIsomapCF:
    def __init__(self,
                 train_feature: torch.Tensor,
                 train_events,
                 num_users: int,
                 num_items: int,
                 user_pos_set=None,
                 ng_seed=42,
                 latent_len: int = 128,
                 n_neighbors: int = 10,
                 epochs: int = 5,
                 cf_epochs: int = 2,
                 final_cf_epochs: int = 3,
                 batch_size: int = 1024,
                 outer_batch_size: int = 4096,
                 lr_isomap: float = 1e-4,
                 lr_ncf: float = 1e-3,
                 factor_num: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 model_type: str = 'NeuMF-end',
                 logs_folder: str = None,
                 device: str = None,
                 stop_criteria_value: float = 0.001,
                 num_ng: int = 3,
                 select_by: str = "loss",
                 final_patience: int = 3,
                 inner_patience: int = 5,
                 warm_start_inner: bool = False,
                 partial_warm_start: bool = False,
                 outer_patience: int = 12,
                 use_item_projection: bool = True,
                 use_procrustes_align: bool = True,
                 save_matrices: bool = False):

        self.features = train_feature
        self.train_events = np.array(train_events, dtype=np.int64)
        self.num_users = num_users
        self.num_items = num_items

        self.user_pos_set = user_pos_set
        self.ng_seed = ng_seed

        self.latent_len = latent_len
        self.n_neighbors = n_neighbors

        self.epochs = epochs
        self.cf_epochs = cf_epochs
        self.final_cf_epochs = final_cf_epochs

        self.batch_size = batch_size
        self.outer_batch_size = outer_batch_size
        self.lr_isomap = lr_isomap
        self.lr_ncf = lr_ncf
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.stop_criteria_value = stop_criteria_value
        self.num_ng = num_ng
        self.select_by = select_by
        self.final_patience = final_patience
        self.inner_patience = inner_patience
        self.warm_start_inner = warm_start_inner
        self.outer_patience = outer_patience
        self.use_item_projection = use_item_projection
        self.use_procrustes_align = use_procrustes_align
        self.partial_warm_start = partial_warm_start
        self.save_matrices = save_matrices
        self._warm_ncf_model = None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        IsomapModule.device = str(self.device)
        print(f"Device: {self.device}")

        if logs_folder is None:
            logs_folder = f"gradisomap_cf_{datetime.now().strftime('%d%m%Y-%H.%M')}"
        self.logs_folder = logs_folder
        os.makedirs(self.logs_folder, exist_ok=True)
        print(f"Logs folder: {self.logs_folder}")

        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_hr': [],
            'val_ndcg': [],
            'had_bad_grad': [],
            'grad_nonfinite_count': [],
            'geom_fro': [],
            'geom_delta_prev': [],
            'geom_delta_init': [],
            'geom_knn_overlap_init': [],
        }

        self.cf_history = {
            'train_loss': [],
            'val_loss': [],
            'val_hr': [],
        }

        self.isomap_model = None
        self.ncf_model = None

        if self.user_pos_set is None:
            raise ValueError()
        if len(self.user_pos_set) != self.num_users:
            raise ValueError(f"len(user_pos_set)={len(self.user_pos_set)} != num_users=")

        train_pairs = [(int(u), int(m)) for (u, m, r) in self.train_events]
        self.train_dataset = NCFTrainDatasetFutureBlind(
            features_pos=train_pairs,
            num_items=self.num_items,
            user_pos_train_set=self.user_pos_set,
            num_ng=self.num_ng,
            seed=self.ng_seed,
        )
        self.users_all = None
        self.items_all = None
        self.labels_all = None
        self._refresh_full_tensors()
        self._Z_prev_aligned = None

    def _refresh_full_tensors(self):
        self.users_all = self.train_dataset.users_fill.to(self.device)
        self.items_all = self.train_dataset.items_fill.to(self.device)
        self.labels_all = self.train_dataset.labels_fill.to(self.device)

    def _device_train_batches(self):
        n = self.users_all.numel()
        perm = torch.randperm(n, device=self.device)
        for s in range(0, n, self.batch_size):
            idx = perm[s:s + self.batch_size]
            yield self.users_all[idx], self.items_all[idx], self.labels_all[idx]

    @staticmethod
    def _generate_random_matrix(n_samples, dist_type='pos', device='cuda'):
        if dist_type == 'uniform':
            matrix = torch.rand(n_samples, n_samples, device=device)
        elif dist_type == 'normal':
            matrix = torch.randn(n_samples, n_samples, device=device).abs()
        elif dist_type == 'exp':
            matrix = torch.rand(n_samples, n_samples, device=device).pow(2)
        elif dist_type == 'pos':
            random_points = torch.randn(n_samples, 1024, device=device)
            dist = torch.cdist(random_points, random_points)
            return dist / dist.max()

        matrix = (matrix + matrix.T) / 2
        matrix.fill_diagonal_(0)
        return matrix / matrix.max()


    def train(
            self,
            val_tensors=None,
            top_k: int = 10,
            device=None,
            use_init_assumption: bool = True,
            dist_type: str = 'pos'
    ):
        vu, vi, vl = val_tensors if val_tensors is not None else (None, None, None)

        start_time = time.time()
        pos_weight = torch.tensor([self.num_ng], device=self.device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.features = self.features.to(torch.float32).to(self.device)
        num_items = self.features.shape[0]

        if use_init_assumption:
            with torch.no_grad():
                dist_train = torch.cdist(self.features, self.features)
        else:
            dist_train = self._generate_random_matrix(num_items, dist_type=dist_type, device=self.device)

        max_val = dist_train.max()
        if max_val > 0:
            dist_train = dist_train / max_val

        np.save(
            os.path.join(self.logs_folder, "D_input_init.npy"),
            dist_train.cpu().numpy().astype(np.float32)
        )
        print(f"[Save] Initial D_input -> D_input_init.npy")

        self._geom_D_init = dist_train.detach().clone()
        self._geom_D_prev = None
        self._geom_knn_init = None
        self._geom_Z_snaps = []

        isomap_model = IsomapNN(
            dist_train,
            n_components=self.latent_len,
            n_neighbors=self.n_neighbors,
            eigval_choice='MDS'
        ).to(self.device)

        isomap_optim = optim.AdamW(isomap_model.parameters(), lr=self.lr_isomap, weight_decay=0.0)

        best_val_loss = np.inf
        best_val_hr_outer = -np.inf
        best_isomap_state = None
        outer_no_improve = 0

        @staticmethod
        def _procrustes_align_torch(ref, mov):
            with torch.no_grad():
                mu_ref = ref.mean(0)
                mu_mov = mov.detach().mean(0)

                A = ref - mu_ref
                B = mov.detach() - mu_mov

                scale = torch.norm(A) / torch.clamp(torch.norm(B), min=1e-12)

                U, _, Vt = torch.linalg.svd((B * scale).T @ A)
                R = U @ Vt

            return mu_ref + (mov - mu_mov) * scale @ R

        for epoch in range(self.epochs):
            epoch_start = time.time()

            item_Z_raw = isomap_model().to(torch.float32)

            if (not self.use_procrustes_align) or self._Z_prev_aligned is None:
                item_Z_full = item_Z_raw
            else:
                item_Z_full = _procrustes_align_torch(
                    self._Z_prev_aligned,
                    item_Z_raw
                )

            item_Z_epoch = item_Z_full.detach()
            if epoch == 0 and self._geom_Z_snaps is not None:
                self._geom_Z_snaps.append(item_Z_epoch.cpu().numpy())

            if self.partial_warm_start and self._warm_ncf_model is not None:
                ncf_model = NeuMFOnManifold(
                    user_num=self.num_users,
                    latent_dim=self.latent_len,
                    factor_num=self.factor_num,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    model_type=self.model_type,
                    use_item_projection=self.use_item_projection
                ).to(self.device)
                warm = self._warm_ncf_model
                ncf_model.embed_user_GMF.weight.data.copy_(warm.embed_user_GMF.weight.data)
                ncf_model.embed_user_MLP.weight.data.copy_(warm.embed_user_MLP.weight.data)
                ncf_model.MLP_layers.load_state_dict(warm.MLP_layers.state_dict())
                print(f"[Partial warm] Transferred user emb + MLP tower; item proj + head reinit")
            elif self.warm_start_inner and self._warm_ncf_model is not None:
                ncf_model = self._warm_ncf_model
                for p in ncf_model.parameters():
                    p.requires_grad_(True)
                ncf_model.train()
            else:
                ncf_model = NeuMFOnManifold(
                    user_num=self.num_users,
                    latent_dim=self.latent_len,
                    factor_num=self.factor_num,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    model_type=self.model_type,
                    use_item_projection=self.use_item_projection
                ).to(self.device)
            ncf_optim = optim.AdamW(ncf_model.parameters(), lr=self.lr_ncf, weight_decay=NCF_ADAMW_WEIGHT_DECAY)

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
            cf_train_losses = []
            cf_val_losses = []
            cf_val_hrs = []

            ncf_model.train()
            for cf_ep in range(self.cf_epochs):
                self.train_dataset.ng_sample()
                self._refresh_full_tensors()
                total_cf_loss = 0.0
                n_batches = 0

                for batch_users, batch_items, batch_labels in self._device_train_batches():
                    preds_cf = ncf_model(batch_users, batch_items, item_Z_epoch)
                    loss_cf = loss_fn(preds_cf, batch_labels)

                    ncf_optim.zero_grad()
                    loss_cf.backward()
                    ncf_optim.step()

                    total_cf_loss += loss_cf.item()
                    n_batches += 1

                avg_cf_train_loss = total_cf_loss / max(1, n_batches)
                cf_train_losses.append(avg_cf_train_loss)

                if vu is not None:
                    ncf_model.eval()
                    score_fn = lambda u, i: ncf_model(u, i, item_Z_epoch)
                    avg_cf_val_loss = val_loss_vec(score_fn, vu, vi, vl, self.device)
                    avg_cf_val_hr, _ = evaluate_topk_vec(
                        score_fn, vu, vi, vl, top_k, self.device)
                    cf_val_losses.append(avg_cf_val_loss)
                    cf_val_hrs.append(avg_cf_val_hr)

                    print(f"[Inner NCF] ep {cf_ep + 1}/{self.cf_epochs} | "
                          f"train={avg_cf_train_loss:.4f}, val={avg_cf_val_loss:.4f}, "
                          f"val_hr@10={avg_cf_val_hr:.4f}")

                    if early_stop.step(avg_cf_train_loss, avg_cf_val_loss, ncf_model,
                                       val_hr=avg_cf_val_hr):
                        break

                    ncf_model.train()
                else:
                    cf_val_losses.append(None)
                    print(f"[Inner NCF] ep {cf_ep + 1}/{self.cf_epochs} | "
                          f"train={avg_cf_train_loss:.4f}")

            if self.select_by == "hr":
                early_stop.restore_best_hr(ncf_model)
            else:
                early_stop.restore_best_window(ncf_model)

            self.cf_history['train_loss'].append(cf_train_losses)
            self.cf_history['val_loss'].append(cf_val_losses)
            self.cf_history['val_hr'].append(cf_val_hrs)

            ncf_model.eval()
            for p in ncf_model.parameters():
                p.requires_grad_(False)

            if self.warm_start_inner or self.partial_warm_start:
                self._warm_ncf_model = ncf_model

            isomap_model.train()
            isomap_optim.zero_grad()

            self.train_dataset.ng_sample()
            self._refresh_full_tensors()
            n_all = self.users_all.numel()
            ob = self.outer_batch_size or self.batch_size
            chunk_starts = list(range(0, n_all, ob))

            grad_accum = torch.zeros_like(item_Z_full)
            bce_sum_value = 0.0

            for ci, s in enumerate(chunk_starts):
                sl = slice(s, min(s + ob, n_all))
                z_chunk = item_Z_full.detach().requires_grad_(True)
                preds_all = ncf_model(self.users_all[sl], self.items_all[sl], z_chunk)
                lsum = F.binary_cross_entropy_with_logits(
                    preds_all, self.labels_all[sl], reduction='sum', pos_weight=pos_weight
                )
                grad_z, = torch.autograd.grad(lsum / n_all, z_chunk)
                grad_accum += grad_z
                bce_sum_value += float(lsum.item())

            avg_train_loss = bce_sum_value / n_all

            item_Z_full.backward(gradient=grad_accum)

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
                print(f"\n[Guard] Outer epoch {epoch}: {grad_nonfinite_count} non-finite "
                      f"gradient entries zeroed before isomap_optim.step()\n")

            isomap_optim.step()

            with torch.no_grad():
                isomap_model.update_distance_matrix()
                item_Z_new_raw = isomap_model().to(torch.float32)
                if self.use_procrustes_align:
                    item_Z_new = _procrustes_align_torch(item_Z_epoch, item_Z_new_raw)
                else:
                    item_Z_new = item_Z_new_raw
                self._Z_prev_aligned = item_Z_new.detach()

            with torch.no_grad():
                D_now = isomap_model.distances_matrix.detach()
                fro_now = float(torch.norm(D_now).item())
                norm_init = max(float(torch.norm(self._geom_D_init).item()), 1e-12)
                delta_init = float(torch.norm(D_now - self._geom_D_init).item()) / norm_init
                if self._geom_D_prev is not None:
                    norm_prev = max(float(torch.norm(self._geom_D_prev).item()), 1e-12)
                    delta_prev = float(torch.norm(D_now - self._geom_D_prev).item()) / norm_prev
                else:
                    delta_prev = 0.0
                if self._geom_knn_init is None:
                    self._geom_knn_init = _knn_indices(self._geom_D_init, self.n_neighbors)
                knn_ovl_init = _knn_overlap(
                    self._geom_knn_init, _knn_indices(D_now, self.n_neighbors))
                self._geom_D_prev = D_now.clone()
                self._geom_Z_snaps.append(item_Z_new.cpu().numpy())

            self.history['geom_fro'].append(fro_now)
            self.history['geom_delta_prev'].append(delta_prev)
            self.history['geom_delta_init'].append(delta_init)
            self.history['geom_knn_overlap_init'].append(knn_ovl_init)
            print(f"[Geom] Δprev={delta_prev:.3e}  Δinit={delta_init:.3e}  "
                  f"kNN_overlap(vs init)={knn_ovl_init:.3f}  ||D||_F={fro_now:.2f}")

            if self.save_matrices:
                save_epoch_matrices(
                    logs_folder=self.logs_folder,
                    epoch=epoch,
                    isomap_model=isomap_model,
                    device=self.device,
                )

            elapsed = time.time() - epoch_start

            if vu is not None:
                ncf_model.eval()
                score_fn_new = lambda u, i: ncf_model(u, i, item_Z_new)
                avg_val_loss = val_loss_vec(score_fn_new, vu, vi, vl, self.device)
                hr_val, ndcg_val = evaluate_topk_vec(
                    score_fn_new, vu, vi, vl, top_k, self.device)
            else:
                avg_val_loss, hr_val, ndcg_val = None, None, None

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_hr'].append(hr_val)
            self.history['val_ndcg'].append(ndcg_val)
            self.history['had_bad_grad'].append(had_bad_grad)
            self.history['grad_nonfinite_count'].append(grad_nonfinite_count)

            print(f"\n[Outer {epoch + 1}/{self.epochs}] "
                  f"train={avg_train_loss:.4f}, "
                  f"val={avg_val_loss if avg_val_loss is not None else float('nan'):.4f}, "
                  f"HR@{top_k}={hr_val if hr_val is not None else float('nan'):.4f}, "
                  f"NDCG@{top_k}={ndcg_val if ndcg_val is not None else float('nan'):.4f}, "
                  f"time={elapsed:.1f}s\n")

            stop_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss

            if self.select_by == "hr":
                outer_improved = hr_val is not None and hr_val > best_val_hr_outer
            else:
                outer_improved = stop_loss < best_val_loss
            if outer_improved:
                if self.select_by == "hr":
                    best_val_hr_outer = hr_val
                best_isomap_state = copy.deepcopy(isomap_model.state_dict())
                outer_no_improve = 0
            else:
                outer_no_improve += 1
            if stop_loss < best_val_loss:
                best_val_loss = stop_loss

            if stop_loss <= self.stop_criteria_value:
                print(f"Stop criteria: loss={stop_loss:.4f} <= {self.stop_criteria_value}")
                break

            if outer_no_improve >= self.outer_patience:
                print(f"[Outer]\nEarly stop: no improvement ({'HR@'+str(top_k) if self.select_by == 'hr' else 'loss'}) "
                      f"for {self.outer_patience} outer epochs (best so far restored below).\n")
                break

            del ncf_model, ncf_optim
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"\nOuter loop finished in "
              f"{time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"Best outer val/train loss = {best_val_loss:.4f}")

        if best_isomap_state is not None:
            isomap_model.load_state_dict(best_isomap_state)
            if self.select_by == "hr":
                print(f"[Outer] Restored best-by-HR@10 geometry (val_hr={best_val_hr_outer:.4f}), "
                      f"not the last outer step's.")
            else:
                print(f"[Outer] Restored best-by-loss geometry (val_loss={best_val_loss:.4f}), "
                      f"not the last outer step's.")

        isomap_model.eval()
        with torch.no_grad():
            item_Z_final = isomap_model().to(torch.float32).detach()

        final_ncf = NeuMFOnManifold(
            user_num=self.num_users,
            latent_dim=self.latent_len,
            factor_num=self.factor_num,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type=self.model_type,
            use_item_projection=self.use_item_projection
        ).to(self.device)

        final_optim = optim.AdamW(final_ncf.parameters(), lr=self.lr_ncf, weight_decay=NCF_ADAMW_WEIGHT_DECAY)

        patience_final = self.final_patience
        best_val_loss_final = np.inf
        best_val_hr_final = -np.inf
        best_state_final = None
        no_improve_final = 0

        for ep in range(self.final_cf_epochs):
            final_ncf.train()
            total_train_loss = 0.0
            n_train_batches = 0

            self.train_dataset.ng_sample()
            self._refresh_full_tensors()

            for batch_users, batch_items, batch_labels in self._device_train_batches():
                preds = final_ncf(batch_users, batch_items, item_Z_final)
                loss = loss_fn(preds, batch_labels)

                final_optim.zero_grad()
                loss.backward()
                final_optim.step()

                total_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)

            if vu is not None:
                final_ncf.eval()
                score_fn_fin = lambda u, i: final_ncf(u, i, item_Z_final)
                avg_val_loss = val_loss_vec(score_fn_fin, vu, vi, vl, self.device)
                avg_val_hr_final, _ = evaluate_topk_vec(
                    score_fn_fin, vu, vi, vl, top_k, self.device)

                print(f"[Final NCF] ep {ep + 1}/{self.final_cf_epochs} | "
                      f"train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, "
                      f"val_hr@10={avg_val_hr_final:.4f}")

                improved = (avg_val_hr_final > best_val_hr_final) if self.select_by == "hr" \
                    else (avg_val_loss < best_val_loss_final)
                if improved:
                    best_val_loss_final = avg_val_loss
                    best_val_hr_final = avg_val_hr_final
                    best_state_final = copy.deepcopy(final_ncf.state_dict())
                    no_improve_final = 0
                else:
                    no_improve_final += 1

                if no_improve_final >= patience_final:
                    print(f"[Final NCF] early stop at ep {ep + 1}")
                    break

                if avg_val_loss <= self.stop_criteria_value:
                    print(f"[Final NCF] stop by threshold: val={avg_val_loss:.4f}")
                    break
            else:
                print(f"[Final NCF] ep {ep + 1}/{self.final_cf_epochs} | "
                      f"train={avg_train_loss:.4f}")

        if best_state_final is not None:
            final_ncf.load_state_dict(best_state_final)

        self.isomap_model = isomap_model
        self.ncf_model = final_ncf

        if len(self._geom_Z_snaps) > 0:
            z_traj = np.stack(self._geom_Z_snaps, axis=0).astype(np.float32)
            np.save(os.path.join(self.logs_folder, "Z_trajectory.npy"), z_traj)
            print("[Save] Z_trajectory.npy")

        torch.save(
            self.isomap_model.state_dict(),
            os.path.join(self.logs_folder, "isomap_model_final.pt")
        )
        torch.save(
            self.ncf_model.state_dict(),
            os.path.join(self.logs_folder, "ncf_model_final.pt")
        )
        print(f"[Save] Models saved in {self.logs_folder}")

        save_history(self.logs_folder, self.history, filename="history.npz")
        cf_history_path = os.path.join(self.logs_folder, "cf_history.json")
        with open(cf_history_path, "w") as f:
            json.dump(self.cf_history, f, indent=4)

        return self.isomap_model, self.ncf_model, item_Z_final
