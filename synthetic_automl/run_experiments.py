
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from synthetic_automl.automl_trainer import run_experiment
from synthetic_automl.compare_results import compare_all_results, create_comparison_plots

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Режим тестирования: True = быстрая проверка, False = полный эксперимент
TEST_MODE = False


DATA_FOLDER = os.path.join(PROJECT_ROOT,'examples', 'gradient_isomap','synthetic', 'outputs', 'torus')



if TEST_MODE:
    NUM_EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 500
    VERBOSE = True
else:
    NUM_EPOCHS = 10000
    EARLY_STOPPING_PATIENCE = 7000
    VERBOSE = True

BATCH_SIZE = 512
LEARNING_RATE = 1e-3

METHODS_TO_RUN = [
    'baseline',
    'sobol',
    'gradnorm',
    'uncertainty_weighting'
]

METHOD_CONFIGS = {
    'baseline': {
        'lambda_graph': 0.0
    },
    'sobol': {
        'initial_lambda_graph': 0.01,
        'warmup_fraction': 0.1,
        'n_samples': 5
    },
    'gradnorm': {
        'initial_lambda_graph': 0.01,
        'alpha': 0.0001,
        'lr_weights': 0.0001
    },
    'uncertainty_weighting': {
        'initial_lambda_graph': 0.01,
        'lr_sigma': 0.001
    }
}

def load_data(folder_path: str) -> Dict[str, np.ndarray]:

    print(f"\n{'='*60}")
    print(f"Loading data from: {folder_path}")
    print(f"{'='*60}")
    
    required_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'fps_indices.npy', 'best_distance_matrix.npy',
        'base_projections.npy', 'train_projections.npy',
        'latent_dim.npy'
    ]

    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(folder_path, f)):
            missing_files.append(f)
    
    if missing_files:
        print(f"\nERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease run first_stage.py first or check DATA_FOLDER path.")
        sys.exit(1)

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
        'latent_dim': int(np.load(os.path.join(folder_path, 'latent_dim.npy')))
    }

    val_proj_path = os.path.join(folder_path, 'val_projections.npy')
    if os.path.exists(val_proj_path):
        data['val_projections'] = np.load(val_proj_path)
    
    print(f"\nData loaded successfully:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_val:   {data['X_val'].shape}")
    print(f"  X_test:  {data['X_test'].shape}")
    print(f"  FPS indices: {len(data['fps_indices'])} basis points")
    print(f"  Latent dim: {data['latent_dim']}")
    
    return data


def reconstruct_distance_matrix(best_distances_matrix: np.ndarray, n_basis: int) -> np.ndarray:
    """
    Reconstruct full symmetric distance matrix from upper triangular form
    """
    weights_matrix = np.zeros((n_basis, n_basis))
    idx = 0
    for i in range(n_basis):
        for j in range(i+1, n_basis):
            weights_matrix[i, j] = best_distances_matrix[idx]
            weights_matrix[j, i] = best_distances_matrix[idx]
            idx += 1
    return weights_matrix


def run_all_experiments(
    data_folder: str,
    methods: List[str],
    method_configs: Dict[str, dict],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    verbose: bool = True
) -> Dict[str, Any]:


    data = load_data(data_folder)

    n_basis = len(data['fps_indices'])
    weights_matrix = reconstruct_distance_matrix(data['best_distance_matrix'], n_basis)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_TEST" if TEST_MODE else ""
    results_folder = os.path.join(SCRIPT_DIR, 'results', f'experiment_{timestamp}{mode_suffix}')
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"AUTOML LAMBDA OPTIMIZATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Results will be saved to: {results_folder}")
    print(f"Epochs: {num_epochs}")
    print(f"Methods: {methods}")
    print(f"{'='*60}\n")

    all_results = {}
    
    for method in methods:
        method_folder = os.path.join(results_folder, method)
        config = method_configs.get(method, {})
        
        try:
            result = run_experiment(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                X_test=data['X_test'],
                y_test=data['y_test'],
                weights_matrix=weights_matrix,
                fps_indices=data['fps_indices'],
                base_projections=data['base_projections'],
                train_projections=data['train_projections'],
                optimizer_name=method,
                optimizer_kwargs=config,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                save_folder=method_folder,
                verbose=verbose
            )
            all_results[method] = result
            
        except Exception as e:
            print(f"\nERROR running {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {'error': str(e)}

    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    comparison = compare_all_results(all_results)

    comparison_path = os.path.join(results_folder, 'comparison_summary.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    create_comparison_plots(all_results, results_folder)

    print_results_table(all_results)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {results_folder}")
    print(f"{'='*60}")
    
    return all_results


def print_results_table(all_results: Dict[str, Any]) -> None:
    """Выводит красивую таблицу с результатами"""
    
    print("\n" + "="*80)
    print(f"{'Method':<25} {'Test MSE':<12} {'Test R²':<10} {'Time (s)':<10} {'Best Epoch':<10}")
    print("-"*80)
    
    for method, result in all_results.items():
        if 'error' in result:
            print(f"{method:<25} ERROR: {result['error'][:40]}")
        else:
            print(f"{method:<25} "
                  f"{result['test_mse']:<12.6f} "
                  f"{result['test_r2']:<10.4f} "
                  f"{result['training_time_seconds']:<10.1f} "
                  f"{result['best_epoch']:<10}")
    
    print("="*80)
    
    # Находим лучший метод
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        best_method = min(valid_results, key=lambda x: valid_results[x]['test_mse'])
        baseline_mse = valid_results.get('baseline', {}).get('test_mse', None)
        best_mse = valid_results[best_method]['test_mse']
        
        print(f"\nBest method: {best_method} (Test MSE: {best_mse:.6f})")
        
        if baseline_mse and best_method != 'baseline':
            improvement = (baseline_mse - best_mse) / baseline_mse * 100
            print(f"Improvement over baseline: {improvement:.2f}%")


if __name__ == "__main__":

    if not os.path.exists(DATA_FOLDER):
        print(f"\nERROR: Data folder not found: {DATA_FOLDER}")
        print("\nPlease update DATA_FOLDER in this script to point to your data.")
        print("The data should be created by running first_stage.py")
        print("\nExpected structure:")
        print("  DATA_FOLDER/")
        print("    ├── X_train.npy")
        print("    ├── y_train.npy")
        print("    ├── fps_indices.npy")
        print("    ├── base_projections.npy")
        print("    └── ...")
        sys.exit(1)
    
    # Запускаем эксперименты
    results = run_all_experiments(
        data_folder=DATA_FOLDER,
        methods=METHODS_TO_RUN,
        method_configs=METHOD_CONFIGS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        verbose=VERBOSE
    )
    
    print("\n Experiment completed!")
