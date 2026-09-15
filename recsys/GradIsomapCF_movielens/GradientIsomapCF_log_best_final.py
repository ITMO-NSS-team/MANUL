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

from Adam.Isomap import IsomapNN
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

        def _flush_histories(logs_folder: str, history: dict, cf_history: dict):
                """Атомарная запись обеих историй на диск после каждой outer-эпохи."""
        
                # --- outer history (npz) ---
                cleaned = {}
                for key, values in history.items():
                    cleaned[key] = np.array(
                        [v if v is not None else np.nan for v in values],
                        dtype=np.float32,
                    )
                np.savez_compressed(
                    os.path.join(logs_folder, "history.npz"), **cleaned
                )
        
                # --- inner history (json) ---
                # cf_history содержит списки списков → json-сериализуемо
                cf_path = os.path.join(logs_folder, "cf_history.json")
                with open(cf_path, "w", encoding="utf-8") as f:
                    json.dump(cf_history, f, indent=2, default=float)

        if device is None:
            device = self.device
        elif isinstance(device, str):
            device = torch.device(device)

        start_time = time.time()
        loss_fn = nn.BCEWithLogitsLoss()

        # на случай старого __init__ без этих ключей
        self.cf_history.setdefault('train_loss', [])
        self.cf_history.setdefault('val_loss', [])
        self.cf_history.setdefault('val_hr', [])
        self.cf_history.setdefault('val_ndcg', [])

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
                # ── создаём файлы истории сразу, чтобы при прерывании было что читать ──
        
        _flush_histories(self.logs_folder, self.history, self.cf_history)
        print(f"[Save] Пустые файлы истории инициализированы")

        isomap_model = IsomapNN(
            dist_train,
            n_components=self.latent_len,
            n_neighbors=self.n_neighbors,
            eigval_choice='MDS'
        ).to(self.device)

        isomap_optim = optim.AdamW(isomap_model.parameters(), lr=self.lr_isomap)

        best_val_loss = np.inf

        # лучшее состояние Isomap по outer-метрикам (primary: HR, secondary: NDCG)
        best_outer = {
            "epoch": -1,
            "val_loss": np.inf,
            "val_hr": -1.0,
            "val_ndcg": -1.0,
            "isomap_state": None,
            "D_input": None,
            "D_geodesic": None,
            "Z": None,
        }

        def _maybe_update_best_outer(epoch, avg_val_loss, hr_val, ndcg_val):
            """Обновляет best_outer и пишет checkpoint на диск."""
            improved = False
            if hr_val is not None:
                if hr_val > best_outer["val_hr"] + 1e-6:
                    improved = True
                elif (
                    abs(hr_val - best_outer["val_hr"]) <= 1e-6
                    and ndcg_val is not None
                    and ndcg_val > best_outer["val_ndcg"] + 1e-6
                ):
                    improved = True
                elif (
                    abs(hr_val - best_outer["val_hr"]) <= 1e-6
                    and (ndcg_val is None or abs(ndcg_val - best_outer["val_ndcg"]) <= 1e-6)
                    and avg_val_loss is not None
                    and avg_val_loss < best_outer["val_loss"] - 1e-6
                ):
                    improved = True
            elif avg_val_loss is not None and avg_val_loss < best_outer["val_loss"] - 1e-6:
                improved = True

            if not improved:
                return False

            isomap_model.eval()
            with torch.no_grad():
                # forward, чтобы embedding_/dist_matrix_ были актуальны
                Z_t = isomap_model()
                Z_best = Z_t.detach().cpu().numpy().astype(np.float32)
                D_input_best = (
                    isomap_model.distances_matrix.detach().cpu().numpy().astype(np.float32)
                )
                D_geodesic_best = (
                    isomap_model.dist_matrix_.detach().cpu().numpy().astype(np.float32)
                )

            best_outer.update({
                "epoch": int(epoch),
                "val_loss": float(avg_val_loss) if avg_val_loss is not None else float("inf"),
                "val_hr": float(hr_val) if hr_val is not None else -1.0,
                "val_ndcg": float(ndcg_val) if ndcg_val is not None else -1.0,
                "isomap_state": {
                    k: v.detach().cpu().clone()
                    for k, v in isomap_model.state_dict().items()
                },
                "D_input": D_input_best,
                "D_geodesic": D_geodesic_best,
                "Z": Z_best,
            })

            np.savez_compressed(
                os.path.join(self.logs_folder, "matrices_BEST_outer.npz"),
                D_input=D_input_best,
                D_geodesic=D_geodesic_best,
                Z=Z_best,
                epoch=np.array([epoch], dtype=np.int32),
                val_loss=np.array([best_outer["val_loss"]], dtype=np.float32),
                val_hr=np.array([best_outer["val_hr"]], dtype=np.float32),
                val_ndcg=np.array([best_outer["val_ndcg"]], dtype=np.float32),
            )
            torch.save(
                best_outer["isomap_state"],
                os.path.join(self.logs_folder, "isomap_model_BEST_outer.pt"),
            )
            print(
                f"  ★ New BEST outer: ep={epoch + 1}, "
                f"val={best_outer['val_loss']:.4f}, "
                f"HR@{top_k}={best_outer['val_hr']:.4f}, "
                f"NDCG@{top_k}={best_outer['val_ndcg']:.4f}"
            )
            return True

        # ======================== OUTER LOOP ========================
        for epoch in range(self.epochs):
            epoch_start = time.time()

            # фиксируем Z предметов на время inner-NCF
            isomap_model.eval()
            with torch.no_grad():
                item_Z_epoch = isomap_model().to(torch.float32).detach()

            ncf_model = NeuMFOnManifold(
                user_num=self.num_users,
                latent_dim=self.latent_len,
                factor_num=self.factor_num,
                num_layers=self.num_layers,
                dropout=self.dropout,
                model_type=self.model_type,
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
            cf_val_hrs = []
            cf_val_ndcgs = []

            for cf_ep in range(self.cf_epochs):
                # ---- train one inner epoch ----
                ncf_model.train()
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
                    # ---- val BCE ----
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

                    # ---- val HR / NDCG (isomap не менялся → Z тот же, что item_Z_epoch) ----
                    hr_inner, ndcg_inner = evaluate_topk_isomap(
                        ncf_model, isomap_model, val_loader, top_k, device
                    )
                    cf_val_hrs.append(hr_inner)
                    cf_val_ndcgs.append(ndcg_inner)

                    print(
                        f"  [Inner NCF] ep {cf_ep + 1}/{self.cf_epochs} | "
                        f"train={avg_cf_train_loss:.4f}, val={avg_cf_val_loss:.4f}, "
                        f"HR@{top_k}={hr_inner:.4f}, NDCG@{top_k}={ndcg_inner:.4f}"
                    )

                    # stop-критерий по-прежнему на val BCE loss
                    if early_stop.step(avg_cf_train_loss, avg_cf_val_loss, ncf_model):
                        print(f"  [Inner NCF] early stop at ep {cf_ep + 1}")
                        break
                else:
                    cf_val_losses.append(None)
                    cf_val_hrs.append(None)
                    cf_val_ndcgs.append(None)
                    print(
                        f"  [Inner NCF] ep {cf_ep + 1}/{self.cf_epochs} | "
                        f"train={avg_cf_train_loss:.4f}"
                    )

            # откат NCF на лучший val-loss в окне (как было)
            early_stop.restore_best_window(ncf_model)

            self.cf_history['train_loss'].append(cf_train_losses)
            self.cf_history['val_loss'].append(cf_val_losses)
            self.cf_history['val_hr'].append(cf_val_hrs)
            self.cf_history['val_ndcg'].append(cf_val_ndcgs)

            # ================== Isomap step (outer) ==================
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

            # ================== Outer val ==================
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

                hr_val, ndcg_val = evaluate_topk_isomap(
                    ncf_model, isomap_model, val_loader, top_k, device
                )
            else:
                avg_val_loss = None
                hr_val, ndcg_val = None, None

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_hr'].append(hr_val)
            self.history['val_ndcg'].append(ndcg_val)

            print(
                f"[Outer {epoch + 1}/{self.epochs}] "
                f"train={avg_train_loss:.4f}, "
                f"val={avg_val_loss if avg_val_loss is not None else float('nan'):.4f}, "
                f"HR@{top_k}={hr_val if hr_val is not None else float('nan'):.4f}, "
                f"NDCG@{top_k}={ndcg_val if ndcg_val is not None else float('nan'):.4f}, "
                f"time={elapsed:.1f}s"
            )

            # best-by-HR/NDCG checkpoint (пункт 1)
            if val_loader is not None:
                _maybe_update_best_outer(epoch, avg_val_loss, hr_val, ndcg_val)

                        # ── инкрементальное сохранение после каждой outer-эпохи ──
            _flush_histories(self.logs_folder, self.history, self.cf_history)
            print(
                f"[Save] Outer epoch {epoch + 1}: "
                f"history + cf_history сброшены на диск"
            )

            stop_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
            if stop_loss < best_val_loss:
                best_val_loss = stop_loss
            if stop_loss <= self.stop_criteria_value:
                print(
                    f"Stop criteria: loss={stop_loss:.4f} <= {self.stop_criteria_value}"
                )
                del ncf_model, ncf_optim
                torch.cuda.empty_cache()
                break

            del ncf_model, ncf_optim
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(
            f"\nOuter loop finished in "
            f"{time.strftime('%H:%M:%S', time.gmtime(total_time))}"
        )
        print(f"Best outer val/train loss = {best_val_loss:.4f}")
        if best_outer["isomap_state"] is not None:
            print(
                f"Best outer by rank: ep={best_outer['epoch'] + 1}, "
                f"HR@{top_k}={best_outer['val_hr']:.4f}, "
                f"NDCG@{top_k}={best_outer['val_ndcg']:.4f}, "
                f"val_loss={best_outer['val_loss']:.4f}"
            )
        else:
            print("Best outer by rank: (none — no val_loader or no improvement)")

        # ======================== FINAL STAGE ========================
        # два состояния Z: LAST (конец outer) и BEST (по HR/NDCG)
        isomap_model.eval()
        state_last = {
            k: v.detach().cpu().clone()
            for k, v in isomap_model.state_dict().items()
        }
        with torch.no_grad():
            item_Z_last = isomap_model().to(torch.float32).detach()

        if best_outer["isomap_state"] is not None:
            isomap_model.load_state_dict(best_outer["isomap_state"])
            isomap_model.eval()
            with torch.no_grad():
                # sync internal buffers after load
                item_Z_best = isomap_model().to(torch.float32).detach()
            # вернём last — _fit будет сам переключать state под eval
            isomap_model.load_state_dict(state_last)
            with torch.no_grad():
                _ = isomap_model()
        else:
            item_Z_best = item_Z_last
            best_outer["isomap_state"] = state_last

        def _fit_final_ncf(item_Z, isomap_state_for_eval, tag: str):
            """
            Обучает NCF на фиксированном item_Z.
            Перед HR/NDCG загружает isomap_state_for_eval, чтобы
            evaluate_topk_isomap(isomap_model()) совпал с item_Z.
            """
            model = NeuMFOnManifold(
                user_num=self.num_users,
                latent_dim=self.latent_len,
                factor_num=self.factor_num,
                num_layers=self.num_layers,
                dropout=self.dropout,
                model_type=self.model_type,
            ).to(self.device)
            opt = optim.AdamW(
                model.parameters(), lr=self.lr_ncf, weight_decay=1e-4
            )

            best_vl = np.inf
            best_state = None
            no_imp = 0
            patience_final = 5

            for ep in range(self.final_cf_epochs):
                model.train()
                total_tr, n_tr = 0.0, 0
                for bu, bi, by in self.inter_loader:
                    bu = bu.to(self.device)
                    bi = bi.to(self.device)
                    by = by.to(self.device)
                    preds = model(bu, bi, item_Z)
                    loss = loss_fn(preds, by)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_tr += loss.item()
                    n_tr += 1
                avg_tr = total_tr / max(1, n_tr)

                if val_loader is not None:
                    model.eval()
                    total_vl, n_vl = 0.0, 0
                    with torch.no_grad():
                        for vu, vi, vy in val_loader:
                            vu = vu.to(self.device)
                            vi = vi.to(self.device)
                            vy = vy.to(self.device)
                            total_vl += loss_fn(model(vu, vi, item_Z), vy).item()
                            n_vl += 1
                    avg_vl = total_vl / max(1, n_vl)
                    print(
                        f"[Final NCF/{tag}] ep {ep + 1}/{self.final_cf_epochs} | "
                        f"train={avg_tr:.4f}, val={avg_vl:.4f}"
                    )

                    if avg_vl < best_vl - 1e-6:
                        best_vl = avg_vl
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        no_imp = 0
                    else:
                        no_imp += 1

                    if no_imp >= patience_final:
                        print(f"[Final NCF/{tag}] early stop at ep {ep + 1}")
                        break
                    if avg_vl <= self.stop_criteria_value:
                        print(
                            f"[Final NCF/{tag}] stop by threshold: val={avg_vl:.4f}"
                        )
                        break
                else:
                    print(
                        f"[Final NCF/{tag}] ep {ep + 1}/{self.final_cf_epochs} | "
                        f"train={avg_tr:.4f}"
                    )

            if best_state is not None:
                model.load_state_dict(best_state)

            metrics = {
                "final_val_loss": float(best_vl) if val_loader is not None else None,
                "HR": None,
                "NDCG": None,
            }

            if val_loader is not None:
                # синхронизируем isomap с тем Z, на котором учили NCF
                isomap_model.load_state_dict(isomap_state_for_eval)
                isomap_model.eval()
                with torch.no_grad():
                    _ = isomap_model()

                model.eval()
                hr, ndcg = evaluate_topk_isomap(
                    model, isomap_model, val_loader, top_k, device
                )
                metrics["HR"] = float(hr)
                metrics["NDCG"] = float(ndcg)
                print(
                    f"[Final NCF/{tag}] "
                    f"HR@{top_k}={hr:.4f}, NDCG@{top_k}={ndcg:.4f}"
                )

            torch.save(
                model.state_dict(),
                os.path.join(self.logs_folder, f"ncf_model_final_{tag}.pt"),
            )
            return model, metrics

        # 1) LAST
        ncf_last, metrics_last = _fit_final_ncf(
            item_Z_last, state_last, tag="LAST"
        )

        # 2) BEST
        if best_outer["epoch"] >= 0 and best_outer["isomap_state"] is not None:
            ncf_best, metrics_best = _fit_final_ncf(
                item_Z_best, best_outer["isomap_state"], tag="BEST"
            )
        else:
            ncf_best, metrics_best = ncf_last, metrics_last
            print("[Final NCF/BEST] skipped (no best_outer) — using LAST")

        # isomap: оставляем BEST в модели, если он был; иначе LAST
        if best_outer["epoch"] >= 0 and best_outer["isomap_state"] is not None:
            isomap_model.load_state_dict(best_outer["isomap_state"])
        else:
            isomap_model.load_state_dict(state_last)
        isomap_model.eval()
        with torch.no_grad():
            _ = isomap_model()

        summary = {
            "best_outer_epoch": int(best_outer["epoch"]),
            "best_outer_val_loss": best_outer["val_loss"],
            "best_outer_HR": best_outer["val_hr"],
            "best_outer_NDCG": best_outer["val_ndcg"],
            "final_LAST": metrics_last,
            "final_BEST": metrics_best,
        }
        summary_path = os.path.join(self.logs_folder, "final_metrics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4, default=float)
        print("[Summary]", json.dumps(summary, indent=2, default=float))

        #self.isomap_model = isomap_model
        ## основная ncf — BEST (по outer rank); LAST лежит отдельно на диске
        #self.ncf_model = ncf_best
#
        #torch.save(
        #    self.isomap_model.state_dict(),
        #    os.path.join(self.logs_folder, "isomap_model_final.pt"),
        #)
        ## дубль best-ncf под старым именем для совместимости
        #torch.save(
        #    self.ncf_model.state_dict(),
        #    os.path.join(self.logs_folder, "ncf_model_final.pt"),
        #)
        ## last isomap state тоже сохраняем
        #torch.save(
        #    state_last,
        #    os.path.join(self.logs_folder, "isomap_model_LAST_outer.pt"),
        #)
        #print(f"[Save] Модели сохранены в {self.logs_folder}")
#
        #save_history(self.logs_folder, self.history, filename="history.npz")
        #cf_history_path = os.path.join(self.logs_folder, "cf_history.json")
        #with open(cf_history_path, "w") as f:
        #    json.dump(self.cf_history, f, indent=4, default=float)
#
        #return self.isomap_model, self.ncf_model
                # ── LAST isomap: отдельный экземпляр с весами последней эпохи ──
        import copy
        isomap_model_last = copy.deepcopy(isomap_model)
        isomap_model_last.load_state_dict(state_last)
        isomap_model_last.eval()
        with torch.no_grad():
            _ = isomap_model_last()   # синхронизируем внутренние буферы

        # ── сохраняем все четыре модели как атрибуты ──
        self.isomap_model      = isomap_model       # BEST (уже загружен выше)
        self.ncf_model         = ncf_best           # BEST
        self.isomap_model_last = isomap_model_last  # LAST
        self.ncf_model_last    = ncf_last           # LAST

        torch.save(
            self.isomap_model.state_dict(),
            os.path.join(self.logs_folder, "isomap_model_final.pt"),
        )
        torch.save(
            self.ncf_model.state_dict(),
            os.path.join(self.logs_folder, "ncf_model_final.pt"),
        )
        torch.save(
            state_last,
            os.path.join(self.logs_folder, "isomap_model_LAST_outer.pt"),
        )
        print(f"[Save] Модели сохранены в {self.logs_folder}")

        #save_history(self.logs_folder, self.history, filename="history.npz")
        #cf_history_path = os.path.join(self.logs_folder, "cf_history.json")
        #with open(cf_history_path, "w") as f:
        #    json.dump(self.cf_history, f, indent=4, default=float)
                # ── финальный сброс (включает данные final stage, если добавятся) ──
        _flush_histories(self.logs_folder, self.history, self.cf_history)
        print(f"[Save] Финальная история сохранена")

        # возвращаем оба чекпоинта явно
        return (self.isomap_model, self.ncf_model), \
               (self.isomap_model_last, self.ncf_model_last)