import numpy as np
import torch
import torch.nn as nn

from new_datasets import NCFTrainDatasetFutureBlind, NCFTestDatasetSampled
from fast_common import TensorBatchLoader, eval_tensors, evaluate_topk_vec, val_loss_vec

rng = np.random.default_rng(0)
U, I = 50, 150
pos = {u: set(rng.choice(I, size=int(rng.integers(5, 110)), replace=False).tolist()) for u in range(U)}
pairs = [(u, m) for u, s in pos.items() for m in sorted(s)]
trip = [(u, 0, int(rng.integers(0, I))) for u in range(U)]
hist = {u: set(s) for u, s in pos.items()}

# 1) воспроизводимость датасетов
a = NCFTrainDatasetFutureBlind(pairs, I, pos, num_ng=3, seed=1)
b = NCFTrainDatasetFutureBlind(pairs, I, pos, num_ng=3, seed=1)
assert torch.equal(a.items_fill, b.items_fill)
ta = NCFTestDatasetSampled(trip, I, hist, num_ng=99, seed=5)
tb = NCFTestDatasetSampled(trip, I, hist, num_ng=99, seed=5)
assert torch.equal(ta.items_fill, tb.items_fill) and torch.equal(ta.labels_fill, tb.labels_fill)
print("OK: datasets reproducible for the same seed")

# 2) TensorBatchLoader: каждый элемент ровно один раз за эпоху, одинаковый порядок при одинаковом seed
l1 = TensorBatchLoader(64, True, seed=3); l2 = TensorBatchLoader(64, True, seed=3)
for l in (l1, l2):
    l.set_tensors(a.users_fill, a.items_fill, a.labels_fill)
e1 = [torch.cat([x[1] for x in l1]) for _ in range(2)]; e2 = [torch.cat([x[1] for x in l2]) for _ in range(2)]
assert torch.equal(e1[0], e2[0]) and torch.equal(e1[1], e2[1]) and not torch.equal(e1[0], e1[1])
assert torch.equal(torch.sort(e1[0]).values, torch.sort(a.items_fill).values)
print("OK: TensorBatchLoader is a seeded permutation of the data")

# 3) evaluate_topk_vec == наивный цикл по пользователям (включая PAD-слоты)
T, B = len(trip), 100
u, i, l = eval_tensors(ta, "cpu")
g = torch.Generator().manual_seed(0)
table = torch.randn(U, I, generator=g)
score_fn = lambda uu, ii: table[uu, ii]
hr, ndcg, mrr = evaluate_topk_vec(score_fn, u, i, l, 10, "cpu", return_mrr=True)
H, N, M = [], [], []
for r in range(T):
    keep = l[r] >= 0
    s = score_fn(u[r][keep], i[r][keep])
    idx = torch.topk(s, min(10, s.numel())).indices.tolist()
    if 0 in idx:
        rk = idx.index(0); H.append(1); N.append(1 / np.log2(rk + 2)); M.append(1 / (rk + 1))
    else:
        H.append(0); N.append(0); M.append(0)
assert abs(hr - np.mean(H)) < 1e-9 and abs(ndcg - np.mean(N)) < 1e-9 and abs(mrr - np.mean(M)) < 1e-9, (hr, np.mean(H))
print(f"OK: evaluate_topk_vec == naive loop (HR={hr:.3f}, NDCG={ndcg:.3f}, MRR={mrr:.3f})")

# 4) val_loss_vec == BCEWithLogits по реальным элементам
pw = torch.tensor([2.0])
mask = (l >= 0).reshape(-1)
logits = score_fn(u.reshape(-1), i.reshape(-1))[mask]
ref = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, l.reshape(-1)[mask]).item()
got = val_loss_vec(score_fn, u, i, l, "cpu", pos_weight=pw)
assert abs(ref - got) < 1e-5, (ref, got)
print("OK: val_loss_vec == BCEWithLogitsLoss over non-PAD entries")
