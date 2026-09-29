import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from MANUL.Adam.Isomap import IsomapNN
from NeuMFOnManifold import NeuMFOnManifold
from evaluation import evaluate_topk_isomap


class GradientIsomapCF:

    def __init__(self,
                 train_feature: torch.Tensor,
                 train_events,
                 num_users: int,
                 num_items: int,
                 latent_len: int,
                 n_neighbors: int = 10,
                 epochs: int = 5,
                 cf_epochs: int = 2,
                 final_cf_epochs = 3,
                 batch_size: int = 2048,
                 lr_isomap: float = 1e-4,
                 lr_ncf: float = 1e-3,
                 factor_num: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 model_type: str = 'NeuMF-end',
                 logs_folder: str | None = None,
                 device: str | None = None,
                 stop_criteria_value: float = 0.001,
                 num_ng: int = 3):

        self.features = train_feature
        self.train_events = np.array(train_events, dtype=np.int64)
        self.num_users = num_users
        self.num_items = num_items
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
        print(f"Device is {self.device}")

        if logs_folder is None:
            logs_folder = f"gradisomap_cf_{datetime.now().strftime('%d%m%Y-%H.%M')}"
        self.logs_folder = logs_folder
        os.makedirs(self.logs_folder, exist_ok=True)
        print(f"Logs folder: {self.logs_folder}")

        # История для графиков
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_hr': [],
            'val_ndcg': [],
        }

        self.cf_history = {
            'train_loss': [],
            'val_loss': []
        }

        self.dist_history = {
            'epoch': [],
            'fro_norm': [],  # ||D||_F
            'delta_prev': [],  # относительное изменение к предыдущей матрице
            'delta_init': []  # относительное изменение к начальной матрице
        }
        self._D_init = None
        self._D_prev = None

        self.interactions = self._build_implicit_interactions()
        self._build_dataloader_and_full_tensors()

        self.isomap_model = None
        self.ncf_model = None

    def _build_implicit_interactions(self):
        print("Building implicit interactions with negative sampling...")
        train_mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        pos_pairs = []

        for (u, m, r) in self.train_events:
            u = int(u)
            m = int(m)
            pos_pairs.append((u, m))
            train_mat[u, m] = 1.0

        interactions = []

        for (u, m) in pos_pairs:
            interactions.append((u, m, 1))

        rng = np.random.default_rng()
        for (u, m) in pos_pairs:
            for _ in range(self.num_ng):
                j = int(rng.integers(low=0, high=self.num_items))
                while (u, j) in train_mat:
                    j = int(rng.integers(low=0, high=self.num_items))
                interactions.append((u, j, 0))

        interactions = np.array(interactions, dtype=np.int64)
        print(f"Позитивов: {len(pos_pairs)}, всего интеракций (pos+neg): {len(interactions)}")
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

    def train(self, val_loader=None, top_k: int = 10, device=None, use_init_assumption: bool = True):

        if device is None:
            device = self.device
        elif isinstance(device, str):
            device = torch.device(device)

        start_time = time.time()

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

        isomap_model = IsomapNN(
            dist_train,
            n_components=self.latent_len,
            n_neighbors=self.n_neighbors,
            eigval_choice='MDS'
        ).to(self.device)

        isomap_optim = optim.AdamW(isomap_model.parameters(), lr=self.lr_isomap)
        loss_fn = nn.BCEWithLogitsLoss()

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

            ncf_model.train()
            cf_train_losses = []
            cf_val_losses = []

            """
            best_inner_val_loss = float('inf')
            no_improve_inner = 0

            diff_threshold = 0.005  # насколько близко должны быть train и val лоссы
            patience_inner = 5  # сколько inner-эпох подряд вал-лосс не должен улучшаться
            """

            best_inner_val_loss = float('inf')

            no_improve_val = 0  # сколько эпох подряд val не улучшался
            train_lt_val_streak = 0  # сколько эпох подряд train < val (в рамках "нет улучшения val")

            patience_val = 5  # val не уменьшается 5 эпох подряд
            patience_overfit = 2  # train < val 2 эпохи подряд
            eps_improve = 1e-4  # что считать "улучшением" val
            overfit_gap = 0.0  # зазор от шума; можно поставить 0.001..0.01 при желании

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

                    print(f"[Inner NCF] epoch {cf_ep + 1}/{self.cf_epochs}, "
                          f"NCF-train={avg_cf_train_loss:.4f}, NCF-val={avg_cf_val_loss:.4f}")

                    # --- early stopping по переобучению ---
                    improved = avg_cf_val_loss < best_inner_val_loss - eps_improve

                    if improved:
                        best_inner_val_loss = avg_cf_val_loss
                        no_improve_val = 0
                        train_lt_val_streak = 0
                    else:
                        no_improve_val += 1

                        # считаем streak train<val только в эпохи, когда val не улучшается
                        if (avg_cf_train_loss + overfit_gap) < avg_cf_val_loss:
                            train_lt_val_streak += 1
                        else:
                            train_lt_val_streak = 0

                    # Условие: train<val 2 эпохи подряд И val не улучшался 5 эпох подряд
                    if (train_lt_val_streak >= patience_overfit) and (no_improve_val >= patience_val):
                        print(f"[Inner NCF] early stop at inner epoch {cf_ep + 1}: "
                              f"train<val for {train_lt_val_streak} epochs, "
                              f"val no improve for {no_improve_val} epochs")
                        ncf_model.train()
                        break

                    ncf_model.train()
                else:
                    cf_val_losses.append(None)
                    print(f"[Inner NCF] epoch {cf_ep + 1}/{self.cf_epochs}, "
                          f"NCF-train={avg_cf_train_loss:.4f}, NCF-val=None")

            self.cf_history['train_loss'].append(cf_train_losses)
            self.cf_history['val_loss'].append(cf_val_losses)

            ncf_model.eval()
            for p in ncf_model.parameters():
                p.requires_grad_(False)

            isomap_model.train()
            isomap_optim.zero_grad()

            item_Z_full = isomap_model().to(torch.float32)  # Z с grad
            users_all = self.users_all.to(self.device)
            items_all = self.items_all.to(self.device)
            labels_all = self.labels_all.to(self.device)

            preds_all = ncf_model(users_all, items_all, item_Z_full)
            bce_loss = loss_fn(preds_all, labels_all)

            loss_iso_train = bce_loss

            loss_iso_train.backward()
            isomap_optim.step()

            with torch.no_grad():
                D = isomap_model.distances_matrix.detach().cpu()

                # if epoch in {0, self.epochs // 2, self.epochs - 1}:
                if epoch > -1:
                    np.save(
                        os.path.join(self.logs_folder, f"D_epoch{epoch}.npy"),
                        D
                    )

                fro = torch.norm(D).item()

                if self._D_init is None:
                    # на первой outer-эпохе задаём "нулевую точку"
                    self._D_init = D.clone()
                    delta_prev = 0.0
                    delta_init = 0.0
                else:
                    diff_prev = torch.norm(D - self._D_prev)
                    diff_init = torch.norm(D - self._D_init)
                    delta_prev = (diff_prev / torch.norm(self._D_prev)).item()
                    delta_init = (diff_init / torch.norm(self._D_init)).item()

                self._D_prev = D.clone()

                self.dist_history['epoch'].append(epoch)
                self.dist_history['fro_norm'].append(fro)
                self.dist_history['delta_prev'].append(delta_prev)
                self.dist_history['delta_init'].append(delta_init)

                print(f"[Isomap ΔD] outer={epoch+1}: "
                      f"||D||_F={fro:.4f}, "
                      f"Δprev={delta_prev:.4e}, Δinit={delta_init:.4e}")

            avg_train_loss = float(loss_iso_train.item())
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

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history.setdefault('val_loss', []).append(avg_val_loss)

            print(f"[GI+NCF-reinit] epoch {epoch+1}/{self.epochs}, "
                  f"Isomap-train={avg_train_loss:.4f}, "
                  f"Isomap-val={avg_val_loss if avg_val_loss is not None else float('nan'):.4f}, "
                  f"time={elapsed:.1f}s")

            if val_loader is not None:
                hr_val, ndcg_val = evaluate_topk_isomap(ncf_model, isomap_model, val_loader, top_k, device)
                self.history['val_hr'].append(hr_val)
                self.history['val_ndcg'].append(ndcg_val)
                print(f"[GI+NCF-reinit] epoch {epoch+1}: HR@{top_k}_val={hr_val:.4f}, "
                      f"NDCG@{top_k}_val={ndcg_val:.4f}")
            else:
                self.history['val_hr'].append(None)
                self.history['val_ndcg'].append(None)

            stop_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss

            if stop_loss < best_val_loss:
                best_val_loss = stop_loss
            if stop_loss <= self.stop_criteria_value:
                print(f"Stop criteria reached: val_loss={stop_loss:.4f} <= {self.stop_criteria_value}")
                break

            del ncf_model, ncf_optim
            torch.cuda.empty_cache()

        total = time.time() - start_time
        print(f"GradientIsomapCF (reinit, intrinsic-style) finished in {time.strftime('%H:%M:%S', time.gmtime(total))}")
        print(f"Best outer val/train Isomap-loss(BCE)={best_val_loss:.4f}")

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
        loss_fn = nn.BCEWithLogitsLoss()

        patience = 3
        best_val_loss_final = np.inf
        best_state_final = None
        no_improve = 0

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

                print(f"[GI+NCF-final] inner epoch {ep}: "
                      f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss_final:
                    best_val_loss_final = avg_val_loss
                    best_state_final = final_ncf.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"[GI+NCF-final] early stopping at epoch {ep+1} (no improvement {patience} epochs)")
                    break

                if avg_val_loss <= self.stop_criteria_value:
                    print(
                        f"[GI+NCF-final] stop by threshold: val_loss={avg_val_loss:.4f} <= {self.stop_criteria_value}")
                    break

            else:
                print(f"[GI+NCF-final] inner epoch {ep+1}: train_loss={avg_train_loss:.4f}")

        if best_state_final is not None:
            final_ncf.load_state_dict(best_state_final)

        self.isomap_model = isomap_model
        self.ncf_model = final_ncf

        torch.save(self.isomap_model.state_dict(),
                   os.path.join(self.logs_folder, "last_isomap_model.pt"))

        return self.isomap_model, self.ncf_model
