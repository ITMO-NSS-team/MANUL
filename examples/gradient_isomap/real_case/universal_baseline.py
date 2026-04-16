import os
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.utils import split_data
from dataset_loader import evaluate_classifier


def make_classifier(input_dim, hidden_dim, n_classes=10, dropout=0.3):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, n_classes, dtype=torch.float64)
    )


def baseline_train_test(folder_path, baseline_model, epochs, batch_size,
                         learning_rate, early_stopping_patience):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    baseline_folder = os.path.join(folder_path, f'baseline_{timestamp}')
    os.makedirs(baseline_folder, exist_ok=True)

    # Load dataset name from metadata
    meta_path = os.path.join(folder_path, 'experiment_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)
    dataset_name = meta.get('dataset_type', 'unknown')
    n_classes = meta.get('n_classes', 10)

    features = np.load(f'{folder_path}/all_features.npy')
    labels = np.load(f'{folder_path}/all_targets.npy')
    X_train, X_val, X_test, y_train_int, y_val_int, y_test_int = split_data(features, labels)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float64),
                      torch.tensor(y_train_int, dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float64),
                      torch.tensor(y_val_int, dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float64),
                      torch.tensor(y_test_int, dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0)

    baseline_model = baseline_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss, patience_counter = float('inf'), 0
    best_epoch, best_model_state = 0, None

    start_time = time.time()
    for epoch in range(epochs):
        baseline_model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(baseline_model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        baseline_model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                epoch_val += criterion(baseline_model(bx), by).item()
        val_losses.append(epoch_val / len(val_loader))

        if epoch % 50 == 0:
            print(f'  Epoch {epoch+1}/{epochs}: Train={train_losses[-1]:.6f}, Val={val_losses[-1]:.6f}')

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            best_model_state = baseline_model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f'  Early stopping at epoch {epoch+1}, best: {best_epoch}')
            break

    if best_model_state is not None:
        baseline_model.load_state_dict(best_model_state)

    metrics = {'dataset': dataset_name, 'experiment_type': 'Baseline'}
    for split_name, X_split, y_int in [
        ('train', X_train, y_train_int),
        ('val', X_val, y_val_int),
        ('test', X_test, y_test_int),
    ]:
        acc, mse = evaluate_classifier(baseline_model, X_split, y_int, n_classes, device)
        metrics[f'{split_name}_accuracy'] = acc
        metrics[f'{split_name}_mse'] = mse

    print(f"  Baseline: test_acc={metrics['test_accuracy']:.4f}, test_mse={metrics['test_mse']:.6f}")
    gap = metrics['train_accuracy'] - metrics['test_accuracy']
    if gap > 0.15:
        print(f"  WARNING: overfit gap = {gap:.3f}")

    # Get test predictions for visualization
    baseline_model.eval()
    with torch.no_grad():
        test_preds_raw = baseline_model(torch.tensor(X_test, dtype=torch.float64).to(device)).cpu().numpy()
    test_pred = np.argmax(test_preds_raw, axis=1)
    test_true = y_test_int

    pd.DataFrame([metrics]).to_csv(os.path.join(baseline_folder, 'metrics.csv'), index=False)
    pd.DataFrame({'epoch': range(1, len(train_losses)+1),
                  'train_loss': train_losses, 'val_loss': val_losses
                  }).to_csv(os.path.join(baseline_folder, 'convergence_log.csv'), index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', alpha=0.7, label=f'Best ({best_epoch})')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
    plt.title(f'{dataset_name.upper()} Baseline Training')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(baseline_folder, 'training_plot.png'), dpi=150)
    plt.close()

    # PCA visualization
    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=test_true, cmap='tab10', s=8, alpha=0.6)
    axes[0].set_title('True Labels')
    axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=test_pred, cmap='tab10', s=8, alpha=0.6)
    axes[1].set_title(f'Predicted (acc={metrics["test_accuracy"]:.3f})')
    plt.suptitle(f'{dataset_name.upper()} Baseline — PCA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(baseline_folder, 'pca_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return metrics, baseline_folder
