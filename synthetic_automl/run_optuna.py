import os
import sys
import json
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch import float64 as fl64
from torch.optim import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

import optuna
from optuna.samplers import TPESampler


SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'examples', 'gradient_isomap', 'synthetic', 'outputs', 'torus')

N_TRIALS = 50
LAMBDA_MIN = 1e-6
LAMBDA_MAX = 0.5


NUM_EPOCHS = 500
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 100

TEST_MODE = False
if TEST_MODE:
    N_TRIALS = 10
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(folder_path):

    print(f"Loading data from: {folder_path}")
    
    data = {
        'X_train': np.load(os.path.join(folder_path, 'X_train.npy')),
        'X_val': np.load(os.path.join(folder_path, 'X_val.npy')),
        'X_test': np.load(os.path.join(folder_path, 'X_test.npy')),
        'y_train': np.load(os.path.join(folder_path, 'y_train.npy')),
        'y_val': np.load(os.path.join(folder_path, 'y_val.npy')),
        'y_test': np.load(os.path.join(folder_path, 'y_test.npy')),
        'fps_indices': np.load(os.path.join(folder_path, 'fps_indices.npy')),
        'best_distance_matrix': np.load(os.path.join(folder_path, 'best_distance_matrix.npy')),
        'base_projections': np.load(os.path.join(folder_path, 'base_projections.npy')),
        'train_projections': np.load(os.path.join(folder_path, 'train_projections.npy')),
    }
    
    print(f"  X_train: {data['X_train'].shape}, X_val: {data['X_val'].shape}, X_test: {data['X_test'].shape}")
    return data


def reconstruct_distance_matrix(best_distances_matrix, n_basis):
    weights_matrix = np.zeros((n_basis, n_basis))
    idx = 0
    for i in range(n_basis):
        for j in range(i+1, n_basis):
            weights_matrix[i, j] = best_distances_matrix[idx]
            weights_matrix[j, i] = best_distances_matrix[idx]
            idx += 1
    return weights_matrix


class SimpleModel(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 32, dtype=torch.float64)
        self.fc3 = nn.Linear(32, 1, dtype=torch.float64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def compute_graph_loss(predictions, batch_indices, Y_all, device):
    Y_batch = torch.tensor(Y_all[batch_indices], dtype=torch.float64).to(device)
    

    distances_sq = torch.cdist(Y_batch, Y_batch, p=2) ** 2
    nonzero_dists = distances_sq[distances_sq > 0]
    
    if len(nonzero_dists) > 0:
        sigma = torch.median(nonzero_dists).sqrt()
    else:
        sigma = torch.tensor(1.0, device=device)
    

    W_batch = torch.exp(-distances_sq / (2 * sigma ** 2))
    W_batch = W_batch - torch.diag(torch.diag(W_batch))

    D_diag = torch.sum(W_batch, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D_diag + 1e-10, -0.5))
    I = torch.eye(W_batch.shape[0], device=device, dtype=torch.float64)
    L_sym = I - D_inv_sqrt @ W_batch @ D_inv_sqrt

    loss = torch.trace(predictions.T @ L_sym @ predictions) / predictions.shape[0]
    return loss


def train_with_lambda(lambda_graph, X_train, y_train, X_val, y_val, 
                      train_projections, device, seed):

    set_seed(seed)

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
    

    model = SimpleModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        indices = torch.randperm(len(X_train)).numpy()
        
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            
            batch_x = torch.tensor(X_train[batch_idx], dtype=fl64).to(device)
            batch_y = torch.tensor(y_train[batch_idx], dtype=fl64).to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            
            # Model loss
            model_loss = criterion(output, batch_y)
            
            # Graph loss
            graph_loss = compute_graph_loss(output, batch_idx, train_projections, device)
            
            # Combined loss
            total_loss = model_loss + lambda_graph * graph_loss
            
            total_loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=fl64).to(device)
            val_y = torch.tensor(y_val, dtype=fl64).to(device)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    
    return best_val_loss


def objective(trial, X_train, y_train, X_val, y_val, train_projections, device):

    lambda_graph = trial.suggest_float('lambda_graph', LAMBDA_MIN, LAMBDA_MAX, log=True)

    val_loss = train_with_lambda(
        lambda_graph=lambda_graph,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_projections=train_projections,
        device=device,
        seed=SEED
    )
    
    return val_loss


def evaluate_final_model(lambda_graph, X_train, y_train, X_val, y_val, X_test, y_test,
                         train_projections, device):

    set_seed(SEED)
    
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    
    model = SimpleModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    final_epochs = NUM_EPOCHS * 2
    
    for epoch in range(final_epochs):
        model.train()
        indices = torch.randperm(len(X_train)).numpy()
        
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            
            batch_x = torch.tensor(X_train[batch_idx], dtype=fl64).to(device)
            batch_y = torch.tensor(y_train[batch_idx], dtype=fl64).to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            
            model_loss = criterion(output, batch_y)
            graph_loss = compute_graph_loss(output, batch_idx, train_projections, device)
            total_loss = model_loss + lambda_graph * graph_loss
            
            total_loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=fl64).to(device)
            val_y = torch.tensor(y_val, dtype=fl64).to(device)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE * 2:
                break
    

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=fl64).to(device)
        test_pred = model(test_x).cpu().numpy().flatten()
    
    test_mse = mean_squared_error(y_test.flatten(), test_pred)
    test_r2 = r2_score(y_test.flatten(), test_pred)
    
    return test_mse, test_r2, best_val_loss


def main():
    
    print("\n" + "="*60)
    print("OPTUNA LAMBDA OPTIMIZATION")
    print("="*60)
    print(f"Mode: {'TEST' if TEST_MODE else 'FULL'}")
    print(f"N_TRIALS: {N_TRIALS}")
    print(f"NUM_EPOCHS per trial: {NUM_EPOCHS}")
    print(f"Lambda range: [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    print(f"SEED: {SEED}")
    print("="*60 + "\n")

    if not os.path.exists(DATA_FOLDER):
        print(f"ERROR: Data folder not found: {DATA_FOLDER}")
        print("Please update DATA_FOLDER in this script.")
        sys.exit(1)

    data = load_data(DATA_FOLDER)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = os.path.join(SCRIPT_DIR, 'results', f'optuna_{timestamp}')
    os.makedirs(results_folder, exist_ok=True)

    set_seed(SEED)
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='lambda_optimization'
    )

    start_time = time.time()
    
    study.optimize(
        lambda trial: objective(
            trial,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            train_projections=data['train_projections'],
            device=device
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    optimization_time = time.time() - start_time

    best_lambda = study.best_params['lambda_graph']
    best_val_loss = study.best_value
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best lambda_graph: {best_lambda:.6e}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Optimization time: {optimization_time:.1f}s")
    print("="*60 + "\n")

    print("Training final model with best lambda...")
    test_mse, test_r2, final_val_loss = evaluate_final_model(
        lambda_graph=best_lambda,
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        train_projections=data['train_projections'],
        device=device
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best lambda_graph: {best_lambda:.6e}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Total time: {optimization_time:.1f}s")
    print("="*60)

    results = {
        'best_lambda_graph': float(best_lambda),
        'best_val_loss': float(best_val_loss),
        'test_mse': float(test_mse),
        'test_r2': float(test_r2),
        'n_trials': N_TRIALS,
        'num_epochs_per_trial': NUM_EPOCHS,
        'optimization_time_seconds': optimization_time,
        'seed': SEED,
        'all_trials': [
            {'lambda_graph': t.params['lambda_graph'], 'val_loss': t.value}
            for t in study.trials
        ]
    }
    
    with open(os.path.join(results_folder, 'optuna_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Строим графики
    
    # 1. История оптимизации
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Optimization history
    ax1 = axes[0]
    trials_vals = [t.value for t in study.trials]
    ax1.plot(range(1, len(trials_vals) + 1), trials_vals, 'o-', alpha=0.7)
    ax1.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss:.6f}')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Optuna Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lambda vs Loss
    ax2 = axes[1]
    lambdas = [t.params['lambda_graph'] for t in study.trials]
    losses = [t.value for t in study.trials]
    ax2.scatter(lambdas, losses, alpha=0.7)
    ax2.axvline(x=best_lambda, color='r', linestyle='--', label=f'Best λ: {best_lambda:.2e}')
    ax2.set_xlabel('Lambda Graph')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xscale('log')
    ax2.set_title('Lambda vs Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'optuna_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {results_folder}")
    print("\n Optuna optimization completed!")


if __name__ == "__main__":
    main()
