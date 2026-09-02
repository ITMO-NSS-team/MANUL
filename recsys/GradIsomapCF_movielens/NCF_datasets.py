import torch
from torch.utils.data import Dataset

import numpy as np


class NCFTrainDataset(Dataset):

    def __init__(self, features_pos, num_items, user_pos_set, num_ng=4, seed=42):
        self.features_pos = list(features_pos)
        self.num_items = int(num_items)
        self.user_pos_set = user_pos_set
        self.num_ng = int(num_ng)
        self.rng = np.random.default_rng(seed)

        self.ng_sample()

    def ng_sample(self):
        features_ng = []
        for (u, i) in self.features_pos:
            u = int(u)
            for _ in range(self.num_ng):
                j = int(self.rng.integers(self.num_items))
                while j in self.user_pos_set[u]:
                    j = int(self.rng.integers(self.num_items))
                features_ng.append((u, j))

        labels_pos = [1] * len(self.features_pos)
        labels_ng = [0] * len(features_ng)

        self.features_fill = self.features_pos + features_ng
        self.labels_fill = labels_pos + labels_ng

    def __len__(self):
        return len(self.features_fill)

    def __getitem__(self, idx):
        u, i = self.features_fill[idx]
        y = self.labels_fill[idx]
        return (torch.tensor(u, dtype=torch.long),
                torch.tensor(i, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32))


class NCFTestDataset(Dataset):

    def __init__(self, test_next, num_items, user_pos_set, num_ng=99, seed=123):
        self.num_items = int(num_items)
        self.user_pos_set = user_pos_set
        self.num_ng = int(num_ng)
        self.rng = np.random.default_rng(seed)

        self.users = []
        self.items = []
        self.labels = []

        for (u, _, target_m) in test_next:
            u = int(u)
            target_m = int(target_m)

            self.users.append(u)
            self.items.append(target_m)
            self.labels.append(1.0)

            negs = 0
            while negs < self.num_ng:
                j = int(self.rng.integers(self.num_items))
                if j in self.user_pos_set[u] or j == target_m:
                    continue

                self.users.append(u)
                self.items.append(j)
                self.labels.append(0.0)
                negs += 1

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (torch.tensor(self.users[idx], dtype=torch.long),
                torch.tensor(self.items[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))
