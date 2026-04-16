import os
import json
from datetime import datetime
import numpy as np
import torch.nn as nn
import pandas as pd
import torch

from regularizator.GraphRegTrainer import GraphRegTrainer
from utils.utils import split_data
from dataset_loader import evaluate_classifier
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def manifold_regularization(folder_path, model, num_epochs, batch_size,
                             learning_rate, early_stop_patience,
                             lambda_method, lambda_graph=1.0, knn_k=100):
    """
    Train classifier with graph regularization.

    Args:
        folder_path: folder with manifold learning artifacts
        model: nn.Module
        num_epochs, batch_size, learning_rate, early_stop_patience: training params
        lambda_method: 'sobol' or None
        lambda_graph: graph loss weight (used when lambda_method is None)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(folder_path, f'regularization_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)

    # Load metadata
    meta_path = os.path.join(folder_path, 'experiment_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)
    dataset_name = meta.get('dataset_type', 'unknown')
    n_classes = meta.get('n_classes', 10)

    fps_indices = np.load(f'{folder_path}/fps_indices.npy')
    dist_matrix = np.load(f'{folder_path}/best_distance_matrix.npy')
    features = np.load(f'{folder_path}/all_features.npy')
    labels = np.load(f'{folder_path}/all_targets.npy')

    X_train, X_val, X_test, y_train_int, y_val_int, y_test_int = split_data(features, labels)
    y_train_ce = y_train_int.astype(np.int64)
    y_val_ce = y_val_int.astype(np.int64)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = GraphRegTrainer(
        train_features=X_train,
        train_target=y_train_ce,
        val_features=X_val,
        val_targets=y_val_ce,
        weights_matrix=dist_matrix,
        base_indices=fps_indices,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        cache_folder=experiment_folder,
        knn_k=knn_k,
    )

    trainer.train(
        plot_convergence=True,
        adaptive_lambda=lambda_method,
        early_stopping_patience=early_stop_patience,
        lambda_graph=lambda_graph,
    )

    # Evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = {'dataset': dataset_name}

    for split_name, X_split, y_int in [
        ('train', X_train, y_train_int),
        ('val', X_val, y_val_int),
        ('test', X_test, y_test_int),
    ]:
        acc, mse = evaluate_classifier(trainer.best_model, X_split, y_int, n_classes, device)
        metrics[f'{split_name}_accuracy'] = acc
        metrics[f'{split_name}_mse'] = mse

    print(f"  GraphReg: test_acc={metrics['test_accuracy']:.4f}, test_mse={metrics['test_mse']:.6f}")
    gap = metrics['train_accuracy'] - metrics['test_accuracy']
    if gap > 0.15:
        print(f"  WARNING: overfit gap = {gap:.3f}")

    pd.DataFrame([metrics]).to_csv(os.path.join(experiment_folder, 'metrics.csv'), index=False)
    return experiment_folder
