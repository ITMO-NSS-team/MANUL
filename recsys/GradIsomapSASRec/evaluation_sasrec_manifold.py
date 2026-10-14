"""
Evaluation for SASRecOnManifold.
Mirrors evaluate_topk_isomap from evaluation.py but adapted for
sequential input (each batch yields input_seq, candidates, labels).
"""
import numpy as np
import torch


def evaluate_topk_sasrec_isomap(
    model,
    isomap_model,
    test_loader,
    top_k: int = 10,
    device=None,
    return_mrr: bool = False,
):
    """
    Sampled-ranking evaluation for SASRecOnManifold.

    test_loader yields (input_seq, candidates, labels):
      input_seq  : [B, maxlen]
      candidates : [B, C] — C = 1 pos + (C-1) neg
      labels     : [B, C] — 1.0 for positive, 0.0 for negatives

    Positive is identified by label==1.0, not by position — robust to
    any ordering (fixes the item[0] assumption in evaluate_topk_isomap).
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    isomap_model.eval()

    from SASRecOnManifold_best import pad_item_Z
    with torch.no_grad():
        item_Z_raw = isomap_model().to(torch.float32)
        item_Z = pad_item_Z(item_Z_raw).to(device)

    HR, NDCG, MRR = [], [], []

    with torch.no_grad():
        for input_seq, candidates, labels in test_loader:
            input_seq = input_seq.to(device)    # [B, maxlen]
            candidates = candidates.to(device)  # [B, C]
            labels = labels.to(device)          # [B, C]

            logits = model.predict_candidates(
                input_seq, candidates, item_Z
            )  # [B, C]

            for b in range(logits.shape[0]):
                b_logits = logits[b]
                b_labels = labels[b]

                pos_indices = (b_labels == 1.0).nonzero(as_tuple=True)[0]
                if len(pos_indices) == 0:
                    continue
                pos_idx = pos_indices[0].item()
                pos_score = b_logits[pos_idx].item()

                # Rank = number of candidates scoring strictly higher
                rank = (b_logits > pos_score).sum().item()

                if rank < top_k:
                    HR.append(1.0)
                    NDCG.append(1.0 / np.log2(rank + 2))
                else:
                    HR.append(0.0)
                    NDCG.append(0.0)
                MRR.append(1.0 / (rank + 1))

    hr = float(np.mean(HR)) if HR else 0.0
    ndcg = float(np.mean(NDCG)) if NDCG else 0.0
    mrr = float(np.mean(MRR)) if MRR else 0.0

    if return_mrr:
        return hr, ndcg, mrr
    return hr, ndcg
