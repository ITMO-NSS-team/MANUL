import torch
import torch.nn.functional as F


class TensorBatchLoader:
    def __init__(self, batch_size, shuffle=True, seed=0, device=None):
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.device = torch.device(device) if device is not None else None
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed) if seed is not None else 0)
        self.users = self.items = self.labels = None

    def set_tensors(self, users, items, labels):
        if self.device is not None:
            users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)
        self.users, self.items, self.labels = users, items, labels

    def __len__(self):
        n = 0 if self.users is None else int(self.users.numel())
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = int(self.users.numel())
        if self.shuffle:
            perm = torch.randperm(n, generator=self._gen)
            if self.device is not None:
                perm = perm.to(self.device)
        else:
            perm = torch.arange(n, device=self.users.device)
        for s in range(0, n, self.batch_size):
            idx = perm[s:s + self.batch_size]
            yield self.users[idx], self.items[idx], self.labels[idx]


def eval_tensors(dataset, device):
    """NCFTestDatasetSampled ---> три тензора формы (T, 1 + num_ng); позитив в колонке 0."""
    B = 1 + int(dataset.num_ng)
    u = dataset.users_fill.to(device).reshape(-1, B)
    i = dataset.items_fill.to(device).reshape(-1, B)
    l = dataset.labels_fill.to(device).reshape(-1, B)
    return u, i, l


def as_eval_tensors(obj, device):
    if obj is None:
        return None
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        return tuple(t.to(device) for t in obj)
    ds = getattr(obj, "dataset", obj)
    if all(hasattr(ds, a) for a in ("users_fill", "items_fill", "labels_fill", "num_ng")):
        return eval_tensors(ds, device)
    return None


@torch.no_grad()
def evaluate_topk_vec(
        score_fn,
        users,
        items,
        labels,
        top_k,
        device,
        chunk_users=1024,
        return_mrr=False
):
    T, B = users.shape
    k = min(int(top_k), B)
    hits = 0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    rank_vals = torch.arange(k, device=device).view(1, -1)
    miss = torch.full_like(rank_vals, k)
    for s in range(0, T, chunk_users):
        u = users[s:s + chunk_users].reshape(-1)
        i = items[s:s + chunk_users].reshape(-1)
        lab = labels[s:s + chunk_users]
        p = score_fn(u, i).view(-1, B)
        p = p.masked_fill(lab < 0, float('-inf'))
        topk = torch.topk(p, k, dim=1).indices
        ranks = torch.where(topk == 0, rank_vals.expand_as(topk), miss.expand_as(topk))
        min_rank = ranks.min(dim=1).values
        hit_mask = min_rank < k
        hits += int(hit_mask.sum())
        ndcg_sum += float((1.0 / torch.log2(min_rank.float() + 2.0) * hit_mask).sum())
        if return_mrr:
            mrr_sum += float((1.0 / (min_rank.float() + 1.0) * hit_mask).sum())
    hr = hits / max(1, T)
    ndcg = ndcg_sum / max(1, T)
    if return_mrr:
        return hr, ndcg, mrr_sum / max(1, T)
    return hr, ndcg


@torch.no_grad()
def val_loss_vec(score_fn, users, items, labels, device, chunk_users=1024, pos_weight=None):
    T, B = users.shape
    total, n = 0.0, 0
    for s in range(0, T, chunk_users):
        u = users[s:s + chunk_users].reshape(-1)
        i = items[s:s + chunk_users].reshape(-1)
        l = labels[s:s + chunk_users].reshape(-1)
        p = score_fn(u, i)
        valid = l >= 0
        loss = F.binary_cross_entropy_with_logits(
            p, l.clamp(min=0.0), reduction='none', pos_weight=pos_weight)
        total += float((loss * valid).sum())
        n += int(valid.sum())
    return total / max(1, n)