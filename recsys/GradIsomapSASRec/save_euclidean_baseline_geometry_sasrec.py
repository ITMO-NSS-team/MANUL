"""
Euclidean baseline for SASRec: standard learnable embeddings.
"""
import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Импортируем классический SASRec (без manifold)
from run_classic_sasrec import SASRec 
from ablation_geometry_vs_optimization_sasrec import build_data_sasrec
from evaluation_sasrec_manifold import evaluate_topk_sasrec_isomap
from sasrec_manifold_sampler import SASRecManifoldTestDataset

def train_euclidean_sasrec(data, device, sasrec_config, lr=1e-3, epochs=50, patience=5):
    """Учим классический SASRec с нуля."""
    model = SASRec(sasrec_config, data["num_movies"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.BCEWithLogitsLoss()
    pad_token = data["num_movies"]

    best_val_hr = -float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for seq, pos, neg in data["train_loader"]:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = model(seq, pos, neg)
            indices = torch.where(pos != pad_token)
            
            loss = loss_fn(pos_logits[indices], torch.ones_like(pos_logits[indices]))
            loss += loss_fn(neg_logits[indices], torch.zeros_like(neg_logits[indices]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= 100: break

        # Валидация (используем хак с IdentityIsomap, чтобы переиспользовать функцию оценки)
        model.eval()
        # Для классического SASRec нам нужно посчитать скоры вручную или адаптировать evaluate
        # Здесь упрощенный inline eval для HR@10
        hits = []
        with torch.no_grad():
            for seq, cands, labels in data["val_loader"]:
                seq, cands = seq.to(device), cands.to(device)
                log_feats = model.log2feats(seq)
                final_feat = log_feats[:, -1, :]
                cand_embs = model.item_emb(cands)
                logits = (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1)
                
                pos_score = logits[:, 0]
                rank = (logits > pos_score.unsqueeze(1)).sum(dim=1)
                hits.extend((rank < 10).cpu().numpy().astype(float).tolist())
        
        hr = np.mean(hits)
        print(f"  ep {ep+1} val_hr@10={hr:.4f}")

        if hr > best_val_hr:
            best_val_hr = hr
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break

    model.load_state_dict(best_state)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="amazon_beauty")
    parser.add_argument("--dataset_type", default="amazon")
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sasrec_config = {'hidden_units': 64, 'maxlen': 50, 'num_blocks': 2, 'num_heads': 1, 'dropout_rate': 0.2}

    tmp_logs = os.path.join(HERE, "euclidean_tmp_logs")
    data = build_data_sasrec(args.max_users, args.max_movies, 5, args.dataset_dir_name, device, tmp_logs, args.dataset_type, args.amazon_category)

    print("--- training Euclidean SASRec baseline ---")
    model = train_euclidean_sasrec(data, device, sasrec_config)

    # Достаем выученные эмбеддинги товаров
    item_emb = model.item_emb.weight[:-1].detach().cpu().numpy() # Убираем padding token
    
    # Считаем матрицу расстояний между ними
    D = np.linalg.norm(item_emb[:, None, :] - item_emb[None, :, :], axis=-1)
    
    out_path = os.path.join(HERE, "euclidean_sasrec_baseline_geometry.npz")
    np.savez(out_path, D=D)
    print(f"[Save] Euclidean geometry saved to {out_path}")

if __name__ == "__main__":
    main()
