"""
Samplers for SASRecOnManifold training and evaluation.

Training sampler: adapted from batch_sequence_sampler (sampler.py) in the
repo - pure Python version (no numba dependency), same sample_fill logic.

Evaluation dataset: NCFTestDatasetSampled-style, adapted for sequential
input (left-padded sequence prefix as model input).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# ─────────────────────────────────────────────────────────────
#  Training sampler: mirrors batch_sequence_sampler from repo
# ─────────────────────────────────────────────────────────────

def _random_neq(n: int, s: set, rng: np.random.RandomState) -> int:
    """Sample random item not in set s. Mirrors random_neq from sampler.py."""
    t = rng.randint(n)
    while t in s:
        t = rng.randint(n)
    return t


def _sample_fill(
    user_items: list,
    n_items: int,
    maxlen: int,
    rng: np.random.RandomState,
    pad_token: int,
):
    """
    Mirrors sample_fill from sampler.py.
    Builds (seq, pos, neg) for a single user.
    """
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full(maxlen, pad_token, dtype=np.int32)

    nxt = user_items[-1]
    idx = maxlen - 1
    ts = set(user_items)

    for item in reversed(user_items[:-1]):
        seq[idx] = item
        pos[idx] = nxt
        neg[idx] = _random_neq(n_items, ts, rng)
        nxt = item
        idx -= 1
        if idx == -1:
            break

    return seq, pos, neg


class SASRecManifoldTrainSampler(IterableDataset):
    """
    Infinite sampler yielding (seq, pos, neg) batches.
    Mirrors batch_sequence_sampler from sampler.py but as a
    PyTorch IterableDataset so DataLoader handles batching.

    Negatives are drawn excluding ALL items the user has seen
    (user_pos_train_set — TRAIN ONLY, no leakage from val/test).
    Negative set is rebuilt fresh each sample (no epoch-level
    re-use), matching the repo's original sampler behaviour.
    """

    def __init__(
        self,
        user_train: dict,           # {user_id: [item1, item2, ...]} sorted by time
        n_items: int,
        maxlen: int,
        user_pos_train_set: dict,   # {user_id: set(items)} — TRAIN ONLY
        seed: int = 42,
    ):
        self.user_train = {
            u: seq for u, seq in user_train.items() if len(seq) >= 2
        }
        self.user_ids = list(self.user_train.keys())
        self.n_items = n_items
        self.maxlen = maxlen
        self.pad_token = n_items
        self.user_pos_train_set = user_pos_train_set
        self.seed = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        n_users = len(self.user_ids)
        while True:
            uid = self.user_ids[rng.randint(n_users)]
            user_items = self.user_train[uid]
            seq, pos, neg = _sample_fill(
                user_items, self.n_items, self.maxlen, rng, self.pad_token
            )
            yield (
                torch.as_tensor(seq, dtype=torch.long),
                torch.as_tensor(pos, dtype=torch.long),
                torch.as_tensor(neg, dtype=torch.long),
            )


# ─────────────────────────────────────────────────────────────
#  Evaluation dataset: sampled ranking (1 pos + num_ng neg)
# ─────────────────────────────────────────────────────────────

class SASRecManifoldTestDataset(Dataset):
    """
    For each user in next_triples:
      input_seq  : full train history left-padded to maxlen
      candidates : [target_item] + [num_ng negatives]
      labels     : [1.0] + [0.0] * num_ng

    Same protocol as NCFTestDatasetSampled but with sequence input.
    Negatives excluded from user_pos_all_set (caller provides appropriate
    exclusion set: train-only for val, train+val for test).
    """

    def __init__(
        self,
        user_train: dict,           # {user_id: [item1, item2, ...]}
        next_triples,               # [(user, last_train_item, target_item)]
        num_items: int,
        user_pos_all_set: dict,     # exclusion set (no neg leakage)
        maxlen: int = 50,
        num_ng: int = 99,
        seed: int = 123,
    ):
        self.maxlen = maxlen
        self.pad_token = num_items
        rng = np.random.default_rng(seed)

        self.input_seqs = []
        self.candidates = []
        self.labels = []

        for (u, _, target_m) in next_triples:
            u = int(u)
            target_m = int(target_m)

            # Build input sequence from train history
            seq = user_train.get(u, [])
            if len(seq) >= maxlen:
                padded = list(seq[-maxlen:])
            else:
                padded = [self.pad_token] * (maxlen - len(seq)) + list(seq)

            # Candidates
            cands = [target_m]
            pos_set = user_pos_all_set.get(u, set())
            attempts = 0
            max_attempts = num_ng * 20
            while len(cands) - 1 < num_ng and attempts < max_attempts:
                j = int(rng.integers(0, num_items))
                if j in pos_set or j == target_m:
                    attempts += 1
                    continue
                cands.append(j)
                attempts += 1

            labs = [1.0] + [0.0] * (len(cands) - 1)

            self.input_seqs.append(padded)
            self.candidates.append(cands)
            self.labels.append(labs)

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_seqs[idx], dtype=torch.long),
            torch.tensor(self.candidates[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
    