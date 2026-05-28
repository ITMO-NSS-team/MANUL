import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from MANUL.Adam.Isomap import IsomapNN
from NeuMFOnManifold import NeuMFOnManifold
from evaluation import evaluate_topk_isomap


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
    ):
        self.patience = patience
        self.window = window
        self.smooth_window = smooth_window
        self.min_delta = min_delta
        self.convergence_delta = convergence_delta
        self.overfit_gap = overfit_gap
        self.overfit_patience = overfit_patience

        self.best_val_loss = np.inf
        self.best_state_global = None
        self.best_epoch_global = -1

        self.best_val_in_window = np.inf
        self.best_state_window = None
        self.best_epoch_window = -1

        self._no_improve = 0
        self._overfit_streak = 0
        self._val_history = []
        self._train_history = []
        self._window_states = []
        self._epoch = 0

    def _smooth(self, history: list) -> float:
        window = history[-self.smooth_window:]
        return float(np.mean(window))

    def step(self, train_loss: float, val_loss: float, model: nn.Module) -> bool:
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

        enough = len(self._val_history) >= self.smooth_window
        smooth_val = self._smooth(self._val_history)
        smooth_train = self._smooth(self._train_history)
        raw_gap = val_loss - train_loss
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
            print(
                f"  [EarlyStopping] Глобально лучшая: "
                f"ep {self.best_epoch_global + 1}, val={self.best_val_loss:.4f}"
            )

    def restore_best_window(self, model: nn.Module):
        if self.best_state_window is not None:
            model.load_state_dict(self.best_state_window)
            print(
                f"  [EarlyStopping] Лучшая в окне (window={self.window}): "
                f"ep {self.best_epoch_window + 1}, val={self.best_val_in_window:.4f}"
            )

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
        device: torch.device,
):
    with torch.no_grad():
        D_input = isomap_model.distances_matrix.detach().cpu().numpy().astype(np.float32)
        D_geodesic = isomap_model.dist_matrix_.detach().cpu().numpy().astype(np.float32)
        Z = isomap_model.embedding_.detach().cpu().numpy().astype(np.float32)

    knn_adj = _build_knn_adjacency(D_input, isomap_model.n_neighbors)

    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    D_latent = torch.cdist(Z_tensor, Z_tensor).numpy().astype(np.float32)

    save_path = os.path.join(logs_folder, f"matrices_epoch{epoch}.npz")
    np.savez_compressed(
        save_path,
        D_input=D_input,
        D_geodesic=D_geodesic,
        knn_adj=knn_adj,
        Z=Z,
        D_latent=D_latent,
    )
    print(f"[Save] Эпоха {epoch + 1}: матрицы сохранены → {save_path}")


def _build_knn_adjacency(D_input: np.ndarray, k: int) -> np.ndarray:
    n = D_input.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        neighbors = np.argsort(D_input[i])[1: k + 1]
        for nb in neighbors:
            adj[i, nb] = D_input[i, nb]
            adj[nb, i] = D_input[nb, i]

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
    print(f"[Save] История обучения сохранена → {save_path}")


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
                 batch_size: int = 2048,
                 lr_isomap: float = 1e-4,
                 lr_ncf: float = 1e-3,
                 factor_num: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 model_type: str = 'NeuMF-end',
                 logs_folder: str = None,
                 device: str = None,
                 stop_criteria_value: float = 0.001,
                 num_ng: int = 3):

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
        self.lr_isomap = lr_isomap
        self.lr_ncf = lr_ncf
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.stop_criteria_value = stop_criteria_value
        self.num_ng = num_ng

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
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
        }

        self.cf_history = {
            'train_loss': [],
            'val_loss': [],
        }

        self.isomap_model = None
        self.ncf_model = None

        self.interactions = self._build_implicit_interactions()
        self._build_dataloader_and_full_tensors()

    def _build_implicit_interactions2(self):
        print("Building implicit interactions with negative sampling...")
        train_mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        pos_pairs = []

        for (u, m, r) in self.train_events:
            u, m = int(u), int(m)
            pos_pairs.append((u, m))
            train_mat[u, m] = 1.0

        interactions = [(u, m, 1) for (u, m) in pos_pairs]

        rng = np.random.default_rng()
        for (u, m) in pos_pairs:
            for _ in range(self.num_ng):
                j = int(rng.integers(low=0, high=self.num_items))
                while (u, j) in train_mat:
                    j = int(rng.integers(low=0, high=self.num_items))
                interactions.append((u, j, 0))

        interactions = np.array(interactions, dtype=np.int64)
        print(f"Позитивов: {len(pos_pairs)}, всего: {len(interactions)}")
        return interactions

    def _build_implicit_interactions(self):
        print("Building implicit interactions with negative sampling...")

        pos_pairs = []
        for (u, m, r) in self.train_events:
            u, m = int(u), int(m)
            pos_pairs.append((u, m))

        # Если user_pos_set не передан — строим только из train
        # (менее корректно, но совместимо со старым поведением)
        if self.user_pos_set is not None:
            check_set = self.user_pos_set
            print("  Негативы проверяются по ПОЛНОМУ user_pos_set (train+val+test)")
        else:
            # Строим train_only множества
            train_only = {}
            for (u, m) in pos_pairs:
                if u not in train_only:
                    train_only[u] = set()
                train_only[u].add(m)
            check_set = train_only
            print("  ВНИМАНИЕ: негативы проверяются только по train взаимодействиям")

        interactions = [(u, m, 1) for (u, m) in pos_pairs]

        # Воспроизводимый генератор
        rng = np.random.default_rng(self.ng_seed)

        for (u, m) in pos_pairs:
            u = int(u)
            for _ in range(self.num_ng):
                j = int(rng.integers(low=0, high=self.num_items))
                while j in check_set.get(u, set()):
                    j = int(rng.integers(low=0, high=self.num_items))
                interactions.append((u, j, 0))

        interactions = np.array(interactions, dtype=np.int64)
        print(f"Позитивов: {len(pos_pairs)}, "
              f"всего интеракций (pos+neg): {len(interactions)}")
        return interactions

    def _build_dataloader_and_full_tensors(self):
        users = torch.tensor(self.interactions[:, 0], dtype=torch.long)
        items = torch.tensor(self.interactions[:, 1], dtype=torch.long)
        labels = torch.tensor(self.interactions[:, 2], dtype=torch.float32)

        dataset = TensorDataset(users, items, labels)
        self.inter_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.users_all = users.to(self.device)
        self.items_all = items.to(self.device)
        self.labels_all = labels.to(self.device)

    @staticmethod
    def _generate_random_matrix(n_samples, dist_type='normal', device='cuda'):
        if dist_type == 'uniform':
            matrix = torch.rand(n_samples, n_samples, device=device)
        elif dist_type == 'normal':
            matrix = torch.randn(n_samples, n_samples, device=device).abs()
        elif dist_type == 'exp':
            matrix = torch.rand(n_samples, n_samples, device=device).pow(2)
        matrix = (matrix + matrix.T) / 2
        matrix.fill_diagonal_(0)
        return matrix / matrix.max()

    def train(self, val_loader=None, top_k: int = 10, device=None,
              use_init_assumption: bool = True):

        if device is None:
            device = self.device
        elif isinstance(device, str):
            device = torch.device(device)

        start_time = time.time()
        loss_fn = nn.BCEWithLogitsLoss()

        self.features = self.features.to(torch.float32).to(self.device)
        num_items = self.features.shape[0]

        if use_init_assumption:
            with torch.no_grad():
                dist_train = torch.cdist(self.features, self.features)
        else:
            dist_train = self._generate_random_matrix(num_items, device=self.device)

        max_val = dist_train.max()
        if max_val > 0:
            dist_train = dist_train / max_val

        np.save(
            os.path.join(self.logs_folder, "D_input_init.npy"),
            dist_train.cpu().numpy().astype(np.float32)
        )
        print(f"[Save] Начальная D_input сохранена → D_input_init.npy")

        isomap_model = IsomapNN(
            dist_train,
            n_components=self.latent_len,
            n_neighbors=self.n_neighbors,
            eigval_choice='MDS'
        ).to(self.device)

        isomap_optim = optim.AdamW(isomap_model.parameters(), lr=self.lr_isomap)

        best_val_loss = np.inf

        for epoch in range(self.epochs):
            epoch_start = time.time()

            isomap_model.eval()
            with torch.no_grad():
                item_Z_epoch = isomap_model().to(torch.float32).detach()

            ncf_model = NeuMFOnManifold(
                user_num=self.num_users,
                latent_dim=self.latent_len,
                factor_num=self.factor_num,
                num_layers=self.num_layers,
                dropout=self.dropout,
                model_type=self.model_type
            ).to(self.device)
            ncf_optim = optim.AdamW(ncf_model.parameters(), lr=self.lr_ncf)

            early_stop = EarlyStopping(
                patience=5,
                window=6,
                smooth_window=2,
                min_delta=1e-4,
                convergence_delta=0.001,
                overfit_gap=0.003,
                overfit_patience=1,
            )
            cf_train_losses = []
            cf_val_losses = []

            ncf_model.train()
            for cf_ep in range(self.cf_epochs):
                total_cf_loss = 0.0
                n_batches = 0

                for batch_users, batch_items, batch_labels in self.inter_loader:
                    batch_users = batch_users.to(self.device)
                    batch_items = batch_items.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    preds_cf = ncf_model(batch_users, batch_items, item_Z_epoch)
                    loss_cf = loss_fn(preds_cf, batch_labels)

                    ncf_optim.zero_grad()
                    loss_cf.backward()
                    ncf_optim.step()

                    total_cf_loss += loss_cf.item()
                    n_batches += 1

                avg_cf_train_loss = total_cf_loss / max(1, n_batches)
                cf_train_losses.append(avg_cf_train_loss)

                if val_loader is not None:
                    ncf_model.eval()
                    total_val_loss = 0.0
                    n_val_batches = 0

                    with torch.no_grad():
                        for val_users, val_items, val_labels in val_loader:
                            val_users = val_users.to(self.device)
                            val_items = val_items.to(self.device)
                            val_labels = val_labels.to(self.device)

                            preds_val = ncf_model(val_users, val_items, item_Z_epoch)
                            loss_val = loss_fn(preds_val, val_labels)

                            total_val_loss += loss_val.item()
                            n_val_batches += 1

                    avg_cf_val_loss = total_val_loss / max(1, n_val_batches)
                    cf_val_losses.append(avg_cf_val_loss)

                    print(f"  [Inner NCF] ep {cf_ep + 1}/{self.cf_epochs} | "
                          f"train={avg_cf_train_loss:.4f}, val={avg_cf_val_loss:.4f}")

                    if early_stop.step(avg_cf_train_loss, avg_cf_val_loss, ncf_model):
                        break

                    ncf_model.train()
                else:
                    cf_val_losses.append(None)
                    print(f"  [Inner NCF] ep {cf_ep + 1}/{self.cf_epochs} | "
                          f"train={avg_cf_train_loss:.4f}")

            early_stop.restore_best_window(ncf_model)

            self.cf_history['train_loss'].append(cf_train_losses)
            self.cf_history['val_loss'].append(cf_val_losses)

            ncf_model.eval()
            for p in ncf_model.parameters():
                p.requires_grad_(False)

            isomap_model.train()
            isomap_optim.zero_grad()

            item_Z_full = isomap_model().to(torch.float32)  # forward с grad
            preds_all = ncf_model(self.users_all, self.items_all, item_Z_full)
            bce_loss = loss_fn(preds_all, self.labels_all)

            bce_loss.backward()
            isomap_optim.step()

            with torch.no_grad():
                isomap_model.update_distance_matrix()
                _ = isomap_model()

            save_epoch_matrices(
                logs_folder=self.logs_folder,
                epoch=epoch,
                isomap_model=isomap_model,
                device=self.device,
            )

            avg_train_loss = float(bce_loss.item())
            elapsed = time.time() - epoch_start

            if val_loader is not None:
                isomap_model.eval()
                ncf_model.eval()

                with torch.no_grad():
                    item_Z_val = isomap_model().to(torch.float32)
                    total_val_loss_outer = 0.0
                    n_val_batches_outer = 0

                    for val_users, val_items, val_labels in val_loader:
                        val_users = val_users.to(self.device)
                        val_items = val_items.to(self.device)
                        val_labels = val_labels.to(self.device)

                        preds_val_outer = ncf_model(val_users, val_items, item_Z_val)
                        loss_val_outer = loss_fn(preds_val_outer, val_labels)

                        total_val_loss_outer += loss_val_outer.item()
                        n_val_batches_outer += 1

                avg_val_loss = total_val_loss_outer / max(1, n_val_batches_outer)
            else:
                avg_val_loss = None

            # HR и NDCG
            if val_loader is not None:
                hr_val, ndcg_val = evaluate_topk_isomap(
                    ncf_model, isomap_model, val_loader, top_k, device
                )
            else:
                hr_val, ndcg_val = None, None

            # Обновляем историю
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_hr'].append(hr_val)
            self.history['val_ndcg'].append(ndcg_val)

            print(f"[Outer {epoch + 1}/{self.epochs}] "
                  f"train={avg_train_loss:.4f}, "
                  f"val={avg_val_loss if avg_val_loss is not None else float('nan'):.4f}, "
                  f"HR@{top_k}={hr_val if hr_val is not None else float('nan'):.4f}, "
                  f"NDCG@{top_k}={ndcg_val if ndcg_val is not None else float('nan'):.4f}, "
                  f"time={elapsed:.1f}s")

            stop_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
            if stop_loss < best_val_loss:
                best_val_loss = stop_loss
            if stop_loss <= self.stop_criteria_value:
                print(f"Stop criteria: loss={stop_loss:.4f} <= {self.stop_criteria_value}")
                break

            del ncf_model, ncf_optim
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"\nOuter loop finished in "
              f"{time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"Best outer val/train loss = {best_val_loss:.4f}")

        isomap_model.eval()
        with torch.no_grad():
            item_Z_final = isomap_model().to(torch.float32).detach()

        final_ncf = NeuMFOnManifold(
            user_num=self.num_users,
            latent_dim=self.latent_len,
            factor_num=self.factor_num,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type=self.model_type
        ).to(self.device)

        final_optim = optim.AdamW(final_ncf.parameters(), lr=self.lr_ncf)

        patience_final = 3
        best_val_loss_final = np.inf
        best_state_final = None
        no_improve_final = 0

        for ep in range(self.final_cf_epochs):
            final_ncf.train()
            total_train_loss = 0.0
            n_train_batches = 0

            for batch_users, batch_items, batch_labels in self.inter_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device)

                preds = final_ncf(batch_users, batch_items, item_Z_final)
                loss = loss_fn(preds, batch_labels)

                final_optim.zero_grad()
                loss.backward()
                final_optim.step()

                total_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)

            if val_loader is not None:
                final_ncf.eval()
                total_val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for val_users, val_items, val_labels in val_loader:
                        val_users = val_users.to(self.device)
                        val_items = val_items.to(self.device)
                        val_labels = val_labels.to(self.device)

                        preds_val = final_ncf(val_users, val_items, item_Z_final)
                        loss_val = loss_fn(preds_val, val_labels)

                        total_val_loss += loss_val.item()
                        n_val_batches += 1

                avg_val_loss = total_val_loss / max(1, n_val_batches)

                print(f"[Final NCF] ep {ep + 1}/{self.final_cf_epochs} | "
                      f"train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss_final:
                    best_val_loss_final = avg_val_loss
                    best_state_final = final_ncf.state_dict()
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

        torch.save(
            self.isomap_model.state_dict(),
            os.path.join(self.logs_folder, "isomap_model_final.pt")
        )
        torch.save(
            self.ncf_model.state_dict(),
            os.path.join(self.logs_folder, "ncf_model_final.pt")
        )
        print(f"[Save] Модели сохранены в {self.logs_folder}")

        save_history(self.logs_folder, self.history, filename="history.npz")
        cf_history_path = os.path.join(self.logs_folder, "cf_history.json")
        with open(cf_history_path, "w") as f:
            json.dump(self.cf_history, f, indent=4)

        return self.isomap_model, self.ncf_model
