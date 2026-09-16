import torch
import torch.nn.functional as F


def build_forbidden_tensor(pos_set_dict, num_users, num_items, device):
    mask = torch.zeros(num_users, num_items, dtype=torch.bool, device=device)
    us, its = [], []
    for u, items in pos_set_dict.items():
        if items:
            us.extend([int(u)] * len(items))
            its.extend(int(m) for m in items)
    if us:
        mask[torch.tensor(us, dtype=torch.long, device=device),
             torch.tensor(its, dtype=torch.long, device=device)] = True
    return mask


def sample_negatives_device(users_rep, num_items, forbidden, max_rounds=100):
    negs = torch.randint(0, num_items, (users_rep.numel(),), device=users_rep.device)
    bad = forbidden[users_rep, negs]
    rounds = 0
    while bool(bad.any()) and rounds < max_rounds:
        m = int(bad.sum())
        cand = torch.randint(0, num_items, (m,), device=users_rep.device)
        negs[bad] = cand
        bad = forbidden[users_rep, negs]
        rounds += 1
    if bool(bad.any()):
        for pos in torch.nonzero(bad).flatten().tolist():
            u = int(users_rep[pos].item())
            allowed = torch.nonzero(~forbidden[u]).flatten()
            if allowed.numel() == 0:
                raise ValueError()
            negs[pos] = allowed[torch.randint(0, allowed.numel(), (1,), device=allowed.device)]
    return negs


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
    hits = 0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    rank_vals = torch.arange(top_k, device=device).view(1, -1)
    miss = torch.full_like(rank_vals, top_k)
    for s in range(0, T, chunk_users):
        u = users[s:s + chunk_users].reshape(-1)
        i = items[s:s + chunk_users].reshape(-1)
        p = score_fn(u, i).view(-1, B)
        topk = torch.topk(p, top_k, dim=1).indices
        ranks = torch.where(topk == 0, rank_vals.expand_as(topk), miss.expand_as(topk))
        min_rank = ranks.min(dim=1).values
        hit_mask = min_rank < top_k
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
def val_loss_vec(score_fn, users, items, labels, device, chunk_users=1024):
    T, B = users.shape
    total, n = 0.0, 0
    for s in range(0, T, chunk_users):
        u = users[s:s + chunk_users].reshape(-1)
        i = items[s:s + chunk_users].reshape(-1)
        l = labels[s:s + chunk_users].reshape(-1)
        p = score_fn(u, i)
        total += float(F.binary_cross_entropy_with_logits(p, l, reduction='sum'))
        n += l.numel()
    return total / max(1, n)


def eval_tensors(dataset, device):
    B = 1 + int(dataset.num_ng)
    u = dataset.users_fill.to(device).reshape(-1, B)
    i = dataset.items_fill.to(device).reshape(-1, B)
    l = dataset.labels_fill.to(device).reshape(-1, B)
    return u, i, l