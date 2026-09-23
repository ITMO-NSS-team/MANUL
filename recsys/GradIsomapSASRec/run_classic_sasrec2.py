import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ==========================================================
# 0. ПУТИ И ИМПОРТЫ (Используем функции подготовки данных руководителя)
# ==========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
# Путь к папке GradIsomapCF_movielens (рядом с GradIsomapSASRec)
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, GINCF_DIR)

from prepare_data import (
    prepare_sequences, subsample_users_items, train_val_test_split_next_item
)

# ==========================================================
# 1. КЛАССИЧЕСКАЯ АРХИТЕКТУРА SASRec
# ==========================================================

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(nn.Module):
    def __init__(self, config, item_num):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num + 1, config['hidden_units'], padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)

        for _ in range(config['num_blocks']):
            self.attention_layernorms.append(nn.LayerNorm(config['hidden_units'], eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(config['hidden_units'], config['num_heads'], config['dropout_rate'])
            )
            self.forward_layernorms.append(nn.LayerNorm(config['hidden_units'], eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(config['hidden_units'], config['dropout_rate']))
        
        self.initialize()

    def initialize(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param.data)

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == self.pad_token)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits


# ==========================================================
# 2. ПОДГОТОВКА ДАННЫХ (Строгая, как в run_experiment.py)
# ==========================================================

def load_amazon_data(dataset_dir_name="amazon_beauty", category="Beauty_and_Personal_Care", 
                     max_users=300, max_movies=800, min_seq_len=5):
    """Загрузка данных через функции из GradIsomapCF_movielens/prepare_data.py"""
    
    dataset_dir = os.path.join(GINCF_DIR, "data", dataset_dir_name)
    ratings_path = os.path.join(dataset_dir, f"{category}.csv")
    
    print(f"Загрузка данных из {ratings_path}...")
    df = pd.read_csv(ratings_path)
    
    # Приводим к единому виду (как в run_experiment.py)
    df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
    df = df[["userId", "movieId", "rating", "timestamp"]]

    # Используем строгие функции подготовки из репозитория
    df_mapped, user2seq = prepare_sequences(df)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies,
        min_seq_len=min_seq_len,
    )

    print("\nTrain/Val/Test split...")
    train_events, val_next, test_next = train_val_test_split_next_item(
        user2seq_sub, min_len=3,
    )

    # Собираем словари для SASRec
    user_train = defaultdict(list)
    for (u, m, r) in train_events:
        user_train[int(u)].append(int(m))
    train_seqs = dict(user_train)
    
    val_targets = {int(u): int(target) for (u, _, target) in val_next}
    test_targets = {int(u): int(target) for (u, _, target) in test_next}
    
    # Собираем множества всех позитивных взаимодействий (для исключения из негативов при оценке)
    user_pos_set = {u: set(seq) for u, seq in train_seqs.items()}
    for u, target in val_targets.items():
        user_pos_set[u].add(target)
        
    print(f"Подготовлено: {num_users} юзеров, {num_movies} товаров. "
          f"Train: {len(train_seqs)}, Val: {len(val_targets)}, Test: {len(test_targets)}")
          
    return train_seqs, val_targets, test_targets, num_users, num_movies, user_pos_set


# ==========================================================
# 3. DATASETS И SAMPLERS
# ==========================================================

class TrainDataset(Dataset):
    """Сэмплер для обучения: (seq, pos, neg)"""
    def __init__(self, user_seqs, num_items, maxlen):
        self.user_seqs = {u: seq for u, seq in user_seqs.items() if len(seq) >= 2}
        self.users = list(self.user_seqs.keys())
        self.num_items = num_items
        self.maxlen = maxlen
        self.pad_token = num_items

    def __len__(self):
        return len(self.users) * 10  # Виртуальная длина

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        u = self.users[idx % len(self.users)]
        seq_items = self.user_seqs[u]
        
        seq = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        pos = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        neg = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        
        nxt = seq_items[-1]
        idx_pos = self.maxlen - 1
        seen_set = set(seq_items)
        
        for item in reversed(seq_items[:-1]):
            seq[idx_pos] = item
            pos[idx_pos] = nxt
            
            neg_item = rng.randint(self.num_items)
            while neg_item in seen_set:
                neg_item = rng.randint(self.num_items)
            neg[idx_pos] = neg_item
            
            nxt = item
            idx_pos -= 1
            if idx_pos == -1: break
            
        return torch.tensor(seq), torch.tensor(pos), torch.tensor(neg)

class TestDataset(Dataset):
    """Сэмплер для оценки: (seq, candidates, labels)"""
    def __init__(self, user_seqs, targets, num_items, maxlen, user_pos_set, num_neg=99):
        self.data = []
        self.pad_token = num_items
        rng = np.random.default_rng(42)
        
        for u, target_item in targets.items():
            if u not in user_seqs: continue
            seq = user_seqs[u]
            
            if len(seq) >= maxlen:
                padded_seq = seq[-maxlen:]
            else:
                padded_seq = [self.pad_token] * (maxlen - len(seq)) + seq
                
            candidates = [target_item]
            
            # Исключаем из негативов ВСЕ позитивные товары пользователя (чтобы оценка была честной)
            pos_set = user_pos_set.get(u, set())
            
            attempts = 0
            while len(candidates) < num_neg + 1 and attempts < 1000:
                neg_item = rng.integers(0, num_items)
                if neg_item not in pos_set and neg_item != target_item:
                    candidates.append(neg_item)
                attempts += 1
                
            labels = [1.0] + [0.0] * (len(candidates) - 1)
            self.data.append((padded_seq, candidates, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, cands, labs = self.data[idx]
        return (torch.tensor(seq, dtype=torch.long), 
                torch.tensor(cands, dtype=torch.long), 
                torch.tensor(labs, dtype=torch.float32))


# ==========================================================
# 4. ОБУЧЕНИЕ И ОЦЕНКА
# ==========================================================
def evaluate_loss(model, loader, device, num_items, criterion):
    """Считает средний BCE loss на датасете (val loss)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for seq, cands, labels in loader:
            seq = seq.to(device)
            cands = cands.to(device)
            labels = labels.to(device)

            log_feats = model.log2feats(seq)
            final_feat = log_feats[:, -1, :]          # (B, H)
            cand_embs = model.item_emb(cands)          # (B, C, H)
            logits = (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1)  # (B, C)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(1, num_batches)


def evaluate_model(model, test_loader, device, top_k=10):
    model.eval()
    HR, NDCG = [], []
    
    with torch.no_grad():
        for seq, cands, labels in test_loader:
            seq = seq.to(device)
            cands = cands.to(device)
            
            log_feats = model.log2feats(seq)
            final_feat = log_feats[:, -1, :] 
            
            cand_embs = model.item_emb(cands) 
            logits = (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1) 
            
            for b in range(logits.shape[0]):
                pos_idx = 0 
                pos_score = logits[b, pos_idx].item()
                rank = (logits[b] > pos_score).sum().item()
                
                if rank < top_k:
                    HR.append(1.0)
                    NDCG.append(1.0 / np.log2(rank + 2))
                else:
                    HR.append(0.0)
                    NDCG.append(0.0)
                    
    return np.mean(HR), np.mean(NDCG)


def main(history_path):
    # --- НАСТРОЙКИ ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    CONFIG = {
        'hidden_units': 64,
        'maxlen': 50,
        'num_blocks': 2,
        'num_heads': 1,
        'dropout_rate': 0.2,
        'learning_rate': 1e-3,
        'batch_size': 2048,
        'epochs': 50,
        'patience': 5
    }
    
    torch.manual_seed(42)
    np.random.seed(42)

    # --- ДАННЫЕ ---
    train_seqs, val_targets, test_targets, num_users, num_items, user_pos_set = load_amazon_data(
        max_users=300, max_movies=800, min_seq_len=5
    )
    
    train_dataset = TrainDataset(train_seqs, num_items, CONFIG['maxlen'])
    val_dataset = TestDataset(train_seqs, val_targets, num_items, CONFIG['maxlen'], user_pos_set)
    
    # Для теста добавляем val_targets в историю
    test_seqs_for_eval = {u: seq + [val_targets[u]] for u, seq in train_seqs.items() if u in val_targets}
    test_dataset = TestDataset(test_seqs_for_eval, test_targets, num_items, CONFIG['maxlen'], user_pos_set)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- МОДЕЛЬ ---
    model = SASRec(CONFIG, num_items).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nЗапуск обучения классического SASRec на {DEVICE}...")
    
    # --- ЦИКЛ ОБУЧЕНИЯ ---
    best_val_hr = -1.0
    best_model_state = None
    no_improve = 0

    history = {
        "config": CONFIG,
        "device": DEVICE,
        "num_users": int(num_users),
        "num_items": int(num_items),
        "epochs": [],           # список словарей по эпохам
        "best_val_hr": None,
        "best_epoch": None,
        "test_hr": None,
        "test_ndcg": None,
    }

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for seq, pos, neg in train_loader:
            seq, pos, neg = seq.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            pos_logits, neg_logits = model(seq, pos, neg)
            
            indices = torch.where(pos != num_items)
            
            loss = criterion(pos_logits[indices], torch.ones_like(pos_logits[indices]))
            loss += criterion(neg_logits[indices], torch.zeros_like(neg_logits[indices]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(1, num_batches)
        
        # --- ВАЛИДАЦИЯ ---
        val_hr, val_ndcg = evaluate_model(model, val_loader, DEVICE, top_k=10)
        val_loss = evaluate_loss(model, val_loader, DEVICE, num_items, criterion)

        print(f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f}")

        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "val_loss": float(val_loss),
            "val_hr@10": float(val_hr),
            "val_ndcg@10": float(val_ndcg),
        })

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        # Early Stopping
        if val_hr > best_val_hr:
            best_val_hr = val_hr
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= CONFIG['patience']:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

    # --- ФИНАЛЬНЫЙ ТЕСТ ---
    print("\nЗагрузка лучших весов и оценка на Test set...")
    model.load_state_dict(best_model_state)
    test_hr, test_ndcg = evaluate_model(model, test_loader, DEVICE, top_k=10)
    
    print(f"\n=== РЕЗУЛЬТАТЫ КЛАССИЧЕСКОГО SASRec ===")
    print(f"Best Val HR@10:  {best_val_hr:.4f}")
    print(f"Test HR@10:      {test_hr:.4f}")
    print(f"Test NDCG@10:    {test_ndcg:.4f}")
    print(f"=======================================\n")

    history["best_val_hr"] = float(best_val_hr)
    history["test_hr"] = float(test_hr)
    history["test_ndcg"] = float(test_ndcg)

    # Определяем, на какой эпохе был лучший val_hr (для удобства)
    if history["epochs"]:
        best_ep = max(history["epochs"], key=lambda e: e["val_hr@10"])
        history["best_epoch"] = best_ep["epoch"]

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(history_path="recsys/GradIsomapSASRec/classic_sasrec/history_sasrec_03.json")
