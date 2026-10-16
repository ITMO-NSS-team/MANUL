import numpy as np
import torch
from torch.utils.data import Dataset


def _sample_negatives_vectorized(users_arr, num_items, forbidden_mask, rng, extra_exclude_items=None):
    M = users_arr.shape[0]
    result = np.empty(M, dtype=np.int64)

    pending_pos = np.arange(M)
    pending_users = users_arr
    pending_extra = extra_exclude_items

    while pending_pos.size > 0:
        cand = rng.integers(0, num_items, size=pending_pos.size)
        bad = forbidden_mask[pending_users, cand]
        if pending_extra is not None:
            bad = bad | (cand == pending_extra)
        good = ~bad

        result[pending_pos[good]] = cand[good]

        pending_pos = pending_pos[~good]
        pending_users = pending_users[~good]
        if pending_extra is not None:
            pending_extra = pending_extra[~good]

    return result


def _sample_negatives_padded(users_arr, num_items, forbidden_mask, rng, extra_exclude_items=None, max_unique_rounds=20):
    M = users_arr.shape[0]
    result = np.full(M, -1, dtype=np.int64)

    pending_pos = np.arange(M)
    pending_users = users_arr
    pending_extra = extra_exclude_items

    for _ in range(max_unique_rounds):
        if pending_pos.size == 0:
            break
        cand = rng.integers(0, num_items, size=pending_pos.size)
        bad = forbidden_mask[pending_users, cand]
        if pending_extra is not None:
            bad = bad | (cand == pending_extra)
        good = ~bad

        result[pending_pos[good]] = cand[good]

        pending_pos = pending_pos[~good]
        pending_users = pending_users[~good]
        if pending_extra is not None:
            pending_extra = pending_extra[~good]

    n_forced = pending_pos.size
    if n_forced > 0:
        for u in np.unique(pending_users):
            u_slot_mask = pending_users == u
            u_positions = pending_pos[u_slot_mask]

            valid_items = np.where(~forbidden_mask[u])[0]
            if pending_extra is not None:
                u_target = pending_extra[u_slot_mask][0]
                valid_items = valid_items[valid_items != u_target]

            if valid_items.size == 0:
                raise ValueError()

            chosen = rng.choice(valid_items, size=u_positions.size, replace=True)
            result[u_positions] = chosen

    return result, n_forced


def _build_forbidden_mask(pos_set_dict, num_users, num_items):
    mask = np.zeros((num_users, num_items), dtype=bool)
    for u, items in pos_set_dict.items():
        if items:
            idx = np.fromiter(items, dtype=np.int64, count=len(items))
            mask[u, idx] = True
    return mask


class NCFTrainDatasetFutureBlind(Dataset):
    def __init__(self, features_pos, num_items, user_pos_train_set, num_ng=4, seed=42):
        pos_arr = np.asarray(features_pos, dtype=np.int64).reshape(-1, 2)
        self.users_pos = pos_arr[:, 0]
        self.items_pos = pos_arr[:, 1]
        self.num_items = int(num_items)
        self.num_ng = int(num_ng)
        self.rng = np.random.default_rng(seed)

        num_users = len(user_pos_train_set)
        self.forbidden_mask = _build_forbidden_mask(user_pos_train_set, num_users, self.num_items)

        self.ng_sample()

    def ng_sample(self):
        n_pos = self.users_pos.shape[0]
        users_rep = np.repeat(self.users_pos, self.num_ng)
        neg_items = _sample_negatives_vectorized(users_rep, self.num_items, self.forbidden_mask, self.rng)

        users_all = np.concatenate([self.users_pos, users_rep])
        items_all = np.concatenate([self.items_pos, neg_items])
        labels_all = np.empty(users_all.shape[0], dtype=np.float32)
        labels_all[:n_pos] = 1.0
        labels_all[n_pos:] = 0.0

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
        users = triples[:, 0]
        targets = triples[:, 2]
        self.num_items = int(num_items)
        self.num_ng = int(num_ng)
        self.rng = np.random.default_rng(seed)

        num_users = len(user_pos_all_set)
        self.forbidden_mask = _build_forbidden_mask(user_pos_all_set, num_users, self.num_items)

        T = users.shape[0]
        users_rep = np.repeat(users, self.num_ng)
        targets_rep = np.repeat(targets, self.num_ng)
        neg_items, n_forced = _sample_negatives_padded(
            users_rep, self.num_items, self.forbidden_mask, self.rng,
            extra_exclude_items=targets_rep,
        )
        if n_forced > 0:
            print(f"[NCFTestDatasetSampled] {n_forced} негативов из {users_rep.shape[0]} "
                  f"добраны с повторами (не хватило уникальных вариантов); "
                  f"длина блока сохранена = {1+self.num_ng} для всех юзеров.")

        block = 1 + self.num_ng
        total = T * block
        users_fill = np.empty(total, dtype=np.int64)
        items_fill = np.empty(total, dtype=np.int64)
        labels_fill = np.empty(total, dtype=np.float32)

        users_fill[0::block] = users
        items_fill[0::block] = targets
        labels_fill[0::block] = 1.0

        neg_reshaped = neg_items.reshape(T, self.num_ng)
        users_neg_reshaped = users_rep.reshape(T, self.num_ng)
        for k in range(self.num_ng):
            users_fill[1 + k::block] = users_neg_reshaped[:, k]
            items_fill[1 + k::block] = neg_reshaped[:, k]
            labels_fill[1 + k::block] = 0.0

        self.users_fill = torch.from_numpy(users_fill)
        self.items_fill = torch.from_numpy(items_fill)
        self.labels_fill = torch.from_numpy(labels_fill)

    def __len__(self):
        return self.users_fill.shape[0]

    def __getitem__(self, idx):
        return self.users_fill[idx], self.items_fill[idx], self.labels_fill[idx]