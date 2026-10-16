import torch
import torch.nn.functional as F

HEADS = ("bary_cos", "bary_euclid", "item_knn")
_MAX_DENSE = 500_000_000   # предел для плотной матрицы (users x items) в item_knn


def history_matrix(train_pairs, num_users, num_items, device):
    u = torch.as_tensor(train_pairs[:, 0], dtype=torch.long, device=device)
    i = torch.as_tensor(train_pairs[:, 1], dtype=torch.long, device=device)
    H = torch.zeros(num_users, num_items, device=device)
    H[u, i] = 1.0                      # повторы схлопываются
    return H, i


def with_tiebreak(score_fn, seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    def fn(u, i):
        s = score_fn(u, i)
        noise = torch.rand(s.numel(), generator=gen).to(s.device)
        return s + 1e-6 * noise
    return fn


@torch.no_grad()
def build_scorers(Z, train_pairs, num_users, num_items, knn_k=20, seed=0, names=None, tiebreak=True):
    names = list(HEADS) if names is None else list(names)
    Z = Z.detach().to(torch.float32)
    device = Z.device
    H, train_items = history_matrix(train_pairs, num_users, num_items, device)
    scorers = {}

    if "bary_cos" in names or "bary_euclid" in names:
        cnt = H.sum(1, keepdim=True)
        bary = (H @ Z) / cnt.clamp(min=1.0)
        glob = Z[train_items].mean(0, keepdim=True)
        bary = torch.where(cnt > 0, bary, glob.expand_as(bary))
        Zn, bary_n = F.normalize(Z, dim=1), F.normalize(bary, dim=1)
        scorers["bary_cos"] = lambda u, i: (bary_n[u] * Zn[i]).sum(-1)
        scorers["bary_euclid"] = lambda u, i: -(bary[u] - Z[i]).norm(dim=-1)

    if "item_knn" in names:
        if num_users * num_items > _MAX_DENSE:
            raise ValueError("item_knn: матрица users x items слишком велика для плотного расчёта "
                             f"({num_users} x {num_items}); уменьшите датасет")
        Zn = F.normalize(Z, dim=1)
        S = Zn @ Zn.T
        S.fill_diagonal_(float("-inf"))
        vals, idx = S.topk(min(int(knn_k), num_items - 1), dim=1)
        W = torch.zeros(num_items, num_items, device=device)
        W.scatter_(1, idx, vals.clamp(min=0.0))
        R = H @ W.T
        scorers["item_knn"] = lambda u, i: R[u, i]

    order = {n: k for k, n in enumerate(HEADS)}
    return {n: (with_tiebreak(scorers[n], seed + order[n]) if tiebreak else scorers[n]) for n in names}