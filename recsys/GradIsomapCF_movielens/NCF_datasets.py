import torch
from torch.utils.data import Dataset

import numpy as np


class NCFTrainDataset(Dataset):
    def __init__(self, features_pos, num_items, train_mat, num_ng=4):
        self.features_pos = features_pos
        self.num_items = num_items
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.ng_sample()

    def ng_sample(self):
        features_ng = []
        for (u, i) in self.features_pos:
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_items)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_items)
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
    def __init__(self, test_next, num_items, train_mat, num_ng=99):
        self.users = []
        self.items = []
        self.labels = []
        for (u, _, target_m) in test_next:
            self.users.append(u)
            self.items.append(target_m)
            self.labels.append(1.0)
            negs = 0
            while negs < num_ng:
                j = np.random.randint(num_items)
                if (u, j) in train_mat or j == target_m:
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
