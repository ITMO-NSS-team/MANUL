import torch

import numpy as np


def hit_metric(gt_item, pred_items):
    return 1 if gt_item in pred_items else 0


def ndcg_metric(gt_item, pred_items):
    if gt_item in pred_items:
        idx = pred_items.index(gt_item)
        return 1.0 / np.log2(idx + 2)
    return 0.0


def evaluate_topk_pure(model, test_loader, top_k, device):
    HR, NDCG = [], []
    model.eval()
    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)

            preds = model(user, item)
            _, indices = torch.topk(preds, top_k)
            recommends = item[indices].cpu().numpy().tolist()
            gt_item = item[0].item()

            HR.append(hit_metric(gt_item, recommends))
            NDCG.append(ndcg_metric(gt_item, recommends))

    return float(np.mean(HR)), float(np.mean(NDCG))


def evaluate_topk_isomap(ncf_model, isomap_model, test_loader, top_k, device):

    if isinstance(device, str):
        device = torch.device(device)

    ncf_model.eval()
    isomap_model.eval()

    HR, NDCG = [], []

    with torch.no_grad():
        item_Z = isomap_model().to(torch.float32)

    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)

            preds = ncf_model(user, item, item_Z)

            _, indices = torch.topk(preds, top_k)
            recommends = item[indices].cpu().numpy().tolist()

            gt_item = item[0].item()

            HR.append(hit_metric(gt_item, recommends))
            NDCG.append(ndcg_metric(gt_item, recommends))

    hr_mean = float(np.mean(HR)) if HR else 0.0
    ndcg_mean = float(np.mean(NDCG)) if NDCG else 0.0
    return hr_mean, ndcg_mean
