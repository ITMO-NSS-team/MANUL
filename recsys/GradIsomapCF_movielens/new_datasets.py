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

        for (u, last_item, target_m) in next_triples:
            u = int(u)
            target_m = int(target_m)

            self.users.append(u)
            self.items.append(target_m)
            self.labels.append(1.0)

            negs = 0
            while negs < self.num_ng:
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

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
