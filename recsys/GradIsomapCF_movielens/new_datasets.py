import numpy as np
import torch
from torch.utils.data import Dataset

PAD_LABEL = -1.0
_DENSE_LIMIT = 200_000_000


class _Forbidden:
    def __init__(self, pos_set_dict, num_items, min_users=0):
        self.num_items = int(num_items)
        us, its = [], []
        for u, items in pos_set_dict.items():
            if items:
                its.append(np.fromiter(items, dtype=np.int64, count=len(items)))
                us.append(np.full(len(items), int(u), dtype=np.int64))
        us = np.concatenate(us) if us else np.empty(0, dtype=np.int64)
        its = np.concatenate(its) if its else np.empty(0, dtype=np.int64)
        if its.size and (its.min() < 0 or its.max() >= self.num_items):
            raise ValueError("user_pos_set содержит item вне диапазона [0, num_items)")

        n_users = max(len(pos_set_dict),
                      (int(max(pos_set_dict)) + 1) if pos_set_dict else 0,
                      int(min_users))
        self.num_users = n_users
        if n_users * self.num_items <= _DENSE_LIMIT:
            self.mask = np.zeros((n_users, self.num_items), dtype=bool)
            self.mask[us, its] = True
            self.keys = None
            self.row_count = self.mask.sum(axis=1).astype(np.int64)
        else:
            self.mask = None
            self.keys = np.unique(us * self.num_items + its)
            self.row_count = np.bincount(self.keys // self.num_items,
                                         minlength=n_users).astype(np.int64)

    def contains(self, users, items):
        if self.mask is not None:
            return self.mask[users, items]
        if self.keys.size == 0:
            return np.zeros(users.shape, dtype=bool)
        k = users * self.num_items + items
        pos = np.minimum(np.searchsorted(self.keys, k), self.keys.size - 1)
        return self.keys[pos] == k

    def allowed_items(self, u):
        if self.mask is not None:
            return np.flatnonzero(~self.mask[u])
        lo, hi = u * self.num_items, (u + 1) * self.num_items
        taken = self.keys[(self.keys >= lo) & (self.keys < hi)] - lo
        return np.setdiff1d(np.arange(self.num_items, dtype=np.int64), taken)


def _sample_negatives(users, num_items, forbidden, rng, extra=None, max_rounds=20):
    M = users.shape[0]
    negs = np.zeros(M, dtype=np.int64)
    ok = np.zeros(M, dtype=bool)
    pend = np.arange(M)
    for _ in range(max_rounds):
        if pend.size == 0:
            break
        cand = rng.integers(0, num_items, size=pend.size)
        bad = forbidden.contains(users[pend], cand)
        if extra is not None:
            bad = bad | (cand == extra[pend])
        good = ~bad
        negs[pend[good]] = cand[good]
        ok[pend[good]] = True
        pend = pend[~good]

    if pend.size:
        ex = extra[pend] if extra is not None else np.zeros(pend.size, dtype=np.int64)
        group = users[pend] * (num_items + 1) + (ex + 1 if extra is not None else 0)
        for g in np.unique(group):
            sel = pend[group == g]
            u = int(users[sel[0]])
            allowed = forbidden.allowed_items(u)
            if extra is not None:
                allowed = allowed[allowed != extra[sel[0]]]
            if allowed.size == 0:
                continue
            negs[sel] = allowed[rng.integers(0, allowed.size, size=sel.size)]
            ok[sel] = True
    return negs, ok


class NCFTrainDatasetFutureBlind(Dataset):
    def __init__(self, features_pos, num_items, user_pos_train_set, num_ng=4, seed=42):
        pos = np.asarray(features_pos, dtype=np.int64).reshape(-1, 2)
        self.users_pos = pos[:, 0].copy()
        self.items_pos = pos[:, 1].copy()
        self.num_items = int(num_items)
        self.num_ng = int(num_ng)
        self.user_pos_train_set = user_pos_train_set
        self.rng = np.random.default_rng(seed)

        min_users = int(self.users_pos.max()) + 1 if self.users_pos.size else 0
        self.forbidden = _Forbidden(user_pos_train_set, self.num_items, min_users)
        self.ng_sample()

    def ng_sample(self):
        n_pos = self.users_pos.shape[0]
        users_rep = np.repeat(self.users_pos, self.num_ng)
        negs, ok = _sample_negatives(users_rep, self.num_items, self.forbidden, self.rng)
        if not bool(ok.all()):
            bad = np.unique(users_rep[~ok])
            raise ValueError(
                f"{bad.size} пользовател(ь/ей) (например {bad[:5].tolist()}) уже взаимодействовали "
                f"со ВСЕМИ {self.num_items} айтемами - негативы сэмплировать не из чего "
                f"(в старой версии здесь был бесконечный цикл). Увеличьте число айтемов "
                f"или исключите этих пользователей.")

        users_all = np.concatenate([self.users_pos, users_rep])
        items_all = np.concatenate([self.items_pos, negs])
        labels_all = np.zeros(users_all.shape[0], dtype=np.float32)
        labels_all[:n_pos] = 1.0

        self.users_fill = torch.from_numpy(users_all)
        self.items_fill = torch.from_numpy(items_all)
        self.labels_fill = torch.from_numpy(labels_all)

    def __len__(self):
        return self.users_fill.shape[0]

    def __getitem__(self, idx):
        return self.users_fill[idx], self.items_fill[idx], self.labels_fill[idx]


class NCFTestDatasetSampled(Dataset):
    def __init__(self, next_triples, num_items, user_pos_all_set, num_ng=99, seed=123):
        triples = np.asarray(next_triples, dtype=np.int64).reshape(-1, 3)
        users = triples[:, 0].copy()
        targets = triples[:, 2].copy()
        T = users.shape[0]
        self.num_items = int(num_items)
        self.num_ng = int(num_ng)
        self.user_pos_all_set = user_pos_all_set
        self.rng = np.random.default_rng(seed)

        min_users = int(users.max()) + 1 if T else 0
        self.forbidden = _Forbidden(user_pos_all_set, self.num_items, min_users)

        # Сколько негативов реально доступно каждому пользователю:
        # |items| - |pos_all ∪ {target}|. Rejection sampling без этого ограничения
        # зацикливался на плотных пользователях (ML-10M).
        if T:
            target_in = self.forbidden.contains(users, targets)
            n_avail = self.num_items - (self.forbidden.row_count[users] + (~target_in).astype(np.int64))
        else:
            n_avail = np.zeros(0, dtype=np.int64)
        n_neg = np.minimum(self.num_ng, np.maximum(n_avail, 0))
        n_users_truncated = int((n_neg < self.num_ng).sum())

        users_rep = np.repeat(users, self.num_ng)
        targets_rep = np.repeat(targets, self.num_ng)
        negs, ok = _sample_negatives(users_rep, self.num_items, self.forbidden, self.rng,
                                     extra=targets_rep)
        negs = negs.reshape(T, self.num_ng)
        ok = ok.reshape(T, self.num_ng)
        valid = (np.arange(self.num_ng)[None, :] < n_neg[:, None]) & ok

        block = 1 + self.num_ng
        users2 = np.repeat(users[:, None], block, axis=1)
        items2 = np.concatenate([targets[:, None], np.where(valid, negs, 0)], axis=1)
        labels2 = np.zeros((T, block), dtype=np.float32)
        labels2[:, 0] = 1.0
        neg_labels = labels2[:, 1:]
        neg_labels[~valid] = PAD_LABEL

        users_fill = users2.reshape(-1).astype(np.int64)
        items_fill = items2.reshape(-1).astype(np.int64)
        labels_fill = labels2.reshape(-1)

        self.users_fill = torch.from_numpy(users_fill)
        self.items_fill = torch.from_numpy(items_fill)
        self.labels_fill = torch.from_numpy(labels_fill)

        real = labels_fill >= 0
        self.users = users_fill[real]
        self.items = items_fill[real]
        self.labels = labels_fill[real]

        if n_users_truncated > 0:
            print(f"[NCFTestDatasetSampled] WARNING: {n_users_truncated} user(s) had fewer than "
                  f"num_ng={self.num_ng} candidate negatives available (positive set too dense "
                  f"relative to num_items={self.num_items}) - they get fewer negatives "
                  f"(остальные слоты помечены PAD и игнорируются при оценке). Ranking task is "
                  f"easier for these users; consider a larger item pool if this count is more "
                  f"than a handful.")

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
