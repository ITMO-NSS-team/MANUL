import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# 1. КЛАССИЧЕСКАЯ АРХИТЕКТУРА SASRec (из твоего репозитория)
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

        # Классический Embedding (вместо item_projection из Isomap)
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
# 2. ПОДГОТОВКА ДАННЫХ (Amazon Beauty)
# ==========================================================

def load_and_prepare_data(csv_path, max_users=300, max_items=800, min_seq_len=5):
    """Загрузка, фильтрация (5-core) и разбиение данных Amazon."""
    print("Загрузка данных...")
    # Ожидаем формат Amazon Reviews '23 или похожий. Если колонки называются иначе, поправь.
    df = pd.read_csv(csv_path)
    
    # Приводим к единому виду
    col_map = {}
    if 'user_id' in df.columns: col_map['user_id'] = 'userId'
    if 'parent_asin' in df.columns: col_map['parent_asin'] = 'itemId'
    elif 'asin' in df.columns: col_map['asin'] = 'itemId'
    if 'timestamp' in df.columns: col_map['timestamp'] = 'timestamp'
    df = df.rename(columns=col_map)
    df = df[['userId', 'itemId', 'timestamp']].dropna()

    # Фильтрация (упрощенный k-core, чтобы оставить только активных)
    for _ in range(5):
        item_counts = df['itemId'].value_counts()
        user_counts = df['userId'].value_counts()
        df = df[df['itemId'].isin(item_counts[item_counts >= 5].index)]
        df = df[df['userId'].isin(user_counts[user_counts >= 5].index)]

    # Маппинг ID в индексы от 0
    users = df['userId'].unique()
    items = df['itemId'].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: idx for idx, i in enumerate(items)}
    
    df['userId'] = df['userId'].map(user2idx)
    df['itemId'] = df['itemId'].map(item2idx)
    
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    # Ограничение выборки (для быстрого теста, как в твоем пайплайне)
    if num_users > max_users:
        valid_users = np.random.choice(num_users, max_users, replace=False)
        df = df[df['userId'].isin(valid_users)]
    if num_items > max_items:
        valid_items = np.random.choice(num_items, max_items, replace=False)
        df = df[df['itemId'].isin(valid_items)]
        
    # Пересчитываем индексы после сабсэмплинга
    users = df['userId'].unique()
    items = df['itemId'].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: idx for idx, i in enumerate(items)}
    df['userId'] = df['userId'].map(user2idx)
    df['itemId'] = df['itemId'].map(item2idx)
    num_users = len(user2idx)
    num_items = len(item2idx)

    # Сортировка по времени и группировка по пользователям
    df = df.sort_values('timestamp')
    user_seqs = df.groupby('userId')['itemId'].apply(list).to_dict()
    
    # Фильтрация по минимальной длине
    user_seqs = {u: seq for u, seq in user_seqs.items() if len(seq) >= min_seq_len}
    
    # Разбиение Train / Val / Test (последние 2 элемента)
    train_seqs, val_targets, test_targets = {}, {}, {}
    for u, seq in user_seqs.items():
        train_seqs[u] = seq[:-2]
        val_targets[u] = seq[-2]
        test_targets[u] = seq[-1]
        
    print(f"Подготовлено: {num_users} юзеров, {num_items} товаров. "
          f"Train: {len(train_seqs)}, Val: {len(val_targets)}, Test: {len(test_targets)}")
          
    return train_seqs, val_targets, test_targets, num_users, num_items


# ==========================================================
# 3. DATASETS И SAMPLERS
# ==========================================================

class TrainDataset(Dataset):
    """Сэмплер для обучения: (seq, pos, neg)"""
    def __init__(self, user_seqs, num_items, maxlen, num_neg=1):
        self.user_seqs = {u: seq for u, seq in user_seqs.items() if len(seq) >= 2}
        self.users = list(self.user_seqs.keys())
        self.num_items = num_items
        self.maxlen = maxlen
        self.num_neg = num_neg
        self.pad_token = num_items

    def __len__(self):
        # Виртуальная длина, чтобы DataLoader мог итерироваться
        return len(self.users) * 10 

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
            
            # Семплируем негатив
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
    def __init__(self, user_seqs, targets, num_items, maxlen, num_neg=99):
        self.data = []
        self.pad_token = num_items
        rng = np.random.default_rng(42)
        
        for u, target_item in targets.items():
            if u not in user_seqs: continue
            seq = user_seqs[u]
            
            # Паддинг
            if len(seq) >= maxlen:
                padded_seq = seq[-maxlen:]
            else:
                padded_seq = [self.pad_token] * (maxlen - len(seq)) + seq
                
            # Кандидаты (1 позитив + 99 негативов)
            candidates = [target_item]
            seen_set = set(seq)
            attempts = 0
            while len(candidates) < num_neg + 1 and attempts < 1000:
                neg_item = rng.integers(0, num_items)
                if neg_item not in seen_set and neg_item != target_item:
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

def evaluate_model(model, test_loader, device, top_k=10):
    model.eval()
    HR, NDCG = [], []
    
    with torch.no_grad():
        for seq, cands, labels in test_loader:
            seq = seq.to(device)
            cands = cands.to(device)
            
            # Получаем скрытые состояния
            log_feats = model.log2feats(seq)
            final_feat = log_feats[:, -1, :]  # Берем последний шаг [B, H]
            
            # Эмбеддинги кандидатов
            cand_embs = model.item_emb(cands) # [B, C, H]
            
            # Скалярное произведение
            logits = (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1) # [B, C]
            
            for b in range(logits.shape[0]):
                pos_idx = 0 # В TestDataset позитив всегда на 0-м месте
                pos_score = logits[b, pos_idx].item()
                
                # Ранг = количество кандидатов с более высоким скором
                rank = (logits[b] > pos_score).sum().item()
                
                if rank < top_k:
                    HR.append(1.0)
                    NDCG.append(1.0 / np.log2(rank + 2))
                else:
                    HR.append(0.0)
                    NDCG.append(0.0)
                    
    return np.mean(HR), np.mean(NDCG)


def main():
    # --- НАСТРОЙКИ ---
    CSV_PATH = "recsys/GradIsomapCF_movielens/data/amazon_beauty/Beauty_and_Personal_Care.csv" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    CONFIG = {
        'hidden_units': 64,
        'maxlen': 50,
        'num_blocks': 2,
        'num_heads': 1,
        'dropout_rate': 0.2,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'epochs': 30,
        'patience': 5
    }
    
    torch.manual_seed(42)
    np.random.seed(42)

    # --- ДАННЫЕ ---
    train_seqs, val_targets, test_targets, num_users, num_items = load_and_prepare_data(
        CSV_PATH, max_users=300, max_items=800, min_seq_len=5
    )
    
    train_dataset = TrainDataset(train_seqs, num_items, CONFIG['maxlen'])
    val_dataset = TestDataset(train_seqs, val_targets, num_items, CONFIG['maxlen'])
    test_dataset = TestDataset(train_seqs, test_targets, num_items, CONFIG['maxlen'])
    
    # Для валидации используем все истории (включая val targets), чтобы тест был честным
    test_seqs_for_eval = {u: seq + [val_targets[u]] for u, seq in train_seqs.items() if u in val_targets}
    test_dataset_full = TestDataset(test_seqs_for_eval, test_targets, num_items, CONFIG['maxlen'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset_full, batch_size=32, shuffle=False)

    # --- МОДЕЛЬ ---
    model = SASRec(CONFIG, num_items).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nЗапуск обучения классического SASRec на {DEVICE}...")
    
    # --- ЦИКЛ ОБУЧЕНИЯ ---
    best_val_hr = -1.0
    best_model_state = None
    no_improve = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for seq, pos, neg in train_loader:
            seq, pos, neg = seq.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            pos_logits, neg_logits = model(seq, pos, neg)
            
            # Маскируем паддинг
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
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | "
              f"Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f}")
        
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

if __name__ == "__main__":
    main()
