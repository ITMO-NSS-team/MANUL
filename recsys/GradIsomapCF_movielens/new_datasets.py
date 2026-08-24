import numpy as np
import torch
from torch.utils.data import Dataset


class NCFTrainDatasetFutureBlind(Dataset):
    def __init__(self, features_pos, num_items, user_pos_train_set, num_ng=4, seed=42):
        self.features_pos = [(int(u), int(i)) for (u, i) in features_pos]
        self.num_items = int(num_items)
        self.user_pos_train_set = user_pos_train_set
        self.num_ng = int(num_ng)
        self.rng = np.random.default_rng(seed)
        self.ng_sample()

    def ng_sample(self):
        features_ng = []
        for (u, i) in self.features_pos:
            for _ in range(self.num_ng):
                j = int(self.rng.integers(self.num_items))
                # проверяем только train positives
                while j in self.user_pos_train_set[u]:
                    j = int(self.rng.integers(self.num_items))
                features_ng.append((u, j))

        labels_pos = [1.0] * len(self.features_pos)
        labels_ng = [0.0] * len(features_ng)

        self.features_fill = self.features_pos + features_ng
        self.labels_fill = labels_pos + labels_ng

    def __len__(self):
        return len(self.features_fill)

    def __getitem__(self, idx):
        u, i = self.features_fill[idx]
        y = self.labels_fill[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


class NCFTestDatasetSampled(Dataset):
    """
    Для каждого пользователя: 1 positive target + num_ng negatives.
    Негативы сэмплим из items, с которыми пользователь НИКОГДА не взаимодействовал
    (по user_pos_all_set).
    """
    def __init__(self, next_triples, num_items, user_pos_all_set, num_ng=99, seed=123):
        self.num_items = int(num_items)
        self.user_pos_all_set = user_pos_all_set
        self.num_ng = int(num_ng)
        self.rng = np.random.default_rng(seed)

        self.users = []
        self.items = []
        self.labels = []

        n_users_truncated = 0

        for (u, last_item, target_m) in next_triples:
            u = int(u)
            target_m = int(target_m)

            self.users.append(u)
            self.items.append(target_m)
            self.labels.append(1.0)

            # Rejection sampling below assumes at least num_ng candidate
            # items exist outside the user's positive set - false in
            # general (e.g. ml-10m's most-active users can have rated
            # 99%+ of a small item pool), and previously caused the loop
            # below to spin forever when it wasn't. Cap to what's actually
            # available so a dense user degrades evaluation quality for
            # themselves (fewer negatives -> an easier ranking task, noted
            # via n_users_truncated) rather than hanging the whole run.
            excluded = self.user_pos_all_set[u] | {target_m}
            n_available = self.num_items - len(excluded)
            n_negs_for_user = min(self.num_ng, max(n_available, 0))
            if n_negs_for_user < self.num_ng:
                n_users_truncated += 1

            negs = 0
            while negs < n_negs_for_user:
                j = int(self.rng.integers(self.num_items))
                # исключаем ВСЕ позитивы пользователя (train+val+test)
                if j in self.user_pos_all_set[u]:
                    continue
                if j == target_m:
                    continue

                self.users.append(u)
                self.items.append(j)
                self.labels.append(0.0)
                negs += 1

        if n_users_truncated > 0:
            print(f"[NCFTestDatasetSampled] WARNING: {n_users_truncated} user(s) had fewer than "
                  f"num_ng={self.num_ng} candidate negatives available (positive set too dense "
                  f"relative to num_items={self.num_items}) - sampled fewer negatives for them "
                  f"instead of hanging. Ranking task is easier for these users; consider a larger "
                  f"item pool if this count is more than a handful.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
