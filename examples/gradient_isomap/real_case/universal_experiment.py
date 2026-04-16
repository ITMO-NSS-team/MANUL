import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from universal_manifold_learning import manifold_learning_pipeline
from universal_baseline import baseline_train_test, make_classifier
from universal_manifold_regularization import manifold_regularization
from dataset_loader import get_dataset_info
from utils.utils import set_global_seed


N_RUNS = 5
N_SAMPLES = 4000
N_BASE_POINTS = 2000
EPOCHS = 30000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 200
DROPOUT = 0.3
MANIFOLD_SEED = 42
LAMBDA_METHOD = 'sobol'   # or None for fixed lambda
LAMBDA_GRAPH = 1.0        # used when LAMBDA_METHOD is None
KNN_K = 200

DATASET = 'mnist'
PRETRAINED_FOLDER = None

RUN_BASELINE = True
RUN_REGULARIZATION = True


if __name__ == "__main__":
    info = get_dataset_info(DATASET)
    input_dim = info['input_dim']
    hidden_dim = info['hidden_dim']
    latent_dim = info['latent_dim']
    n_classes = info['n_classes']

    outputs_dir = f'outputs_{DATASET}'
    data_dir = './data'

    print(f"\n{DATASET.upper()}: MANUL Manifold Regularization ({N_RUNS} runs, {N_SAMPLES} samples)")

    os.makedirs(outputs_dir, exist_ok=True)

    if PRETRAINED_FOLDER:
        results_folder = PRETRAINED_FOLDER
        if not os.path.isdir(results_folder):
            raise FileNotFoundError(f"Not found: {results_folder}")
        print(f"Using pretrained: {results_folder}")
    else:
        print(f"Stage 1: manifold learning (seed={MANIFOLD_SEED})")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        working_folder = f'{outputs_dir}/{DATASET}_run_{timestamp}'
        os.makedirs(working_folder, exist_ok=True)

        results_folder = manifold_learning_pipeline(
            dataset_name=DATASET,
            working_folder=working_folder,
            data_dir=data_dir,
            n_samples=N_SAMPLES,
            n_base_points=N_BASE_POINTS,
            latent_dim=latent_dim,
            epochs=EPOCHS,
            random_state=MANIFOLD_SEED,
        )

    all_metrics = []

    for n in range(N_RUNS):
        run_seed = 100 + n

        print(f"\n--- Run {n + 1}/{N_RUNS} (seed={run_seed}) ---")

        if RUN_BASELINE:
            print(f'\nBaseline (run {n + 1})...')
            set_global_seed(run_seed)
            baseline_metrics, _ = baseline_train_test(
                folder_path=results_folder,
                baseline_model=make_classifier(input_dim, hidden_dim, n_classes, DROPOUT),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
            all_metrics.append({**baseline_metrics, 'run': n})

        if RUN_REGULARIZATION:
            print(f'\nRegularized (run {n + 1})...')
            set_global_seed(run_seed)
            reg_folder = manifold_regularization(
                folder_path=results_folder,
                model=make_classifier(input_dim, hidden_dim, n_classes, DROPOUT),
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                early_stop_patience=EARLY_STOPPING_PATIENCE,
                lambda_method=LAMBDA_METHOD,
                lambda_graph=LAMBDA_GRAPH,
                knn_k=KNN_K,
            )
            reg_metrics = pd.read_csv(f'{reg_folder}/metrics.csv').to_dict('records')[0]
            reg_metrics['experiment_type'] = 'Regularized'
            all_metrics.append({**reg_metrics, 'run': n})

    all_metrics = pd.DataFrame(all_metrics)

    acc_cols = [c for c in all_metrics.columns if 'accuracy' in c]
    experiment_types = all_metrics['experiment_type'].unique().tolist()

    if acc_cols:
        fig, axes = plt.subplots(1, len(acc_cols), figsize=(5 * len(acc_cols), 5))
        if len(acc_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(acc_cols):
            ax = axes[idx]
            all_metrics.boxplot(column=col, by='experiment_type', ax=ax)
            for i, exp in enumerate(experiment_types):
                d = all_metrics[all_metrics['experiment_type'] == exp][col]
                ax.scatter([i + 1] * len(d), d, alpha=0.6, s=30)
            ax.set_title(col.replace('_', ' ').upper())
            ax.set_xlabel('')
            ax.get_figure().suptitle('')
        title_parts = ' vs '.join(experiment_types)
        plt.suptitle(f'{DATASET.upper()} — {title_parts}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, f'{DATASET}_accuracy_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    all_metrics.to_csv(os.path.join(outputs_dir, f'{DATASET}_combined_metrics.csv'), index=False)

    print(f"\nRESULTS (mean ± std):")
    for exp_type in experiment_types:
        sub = all_metrics[all_metrics['experiment_type'] == exp_type]
        print(f"  {exp_type}: test_acc={sub['test_accuracy'].mean():.4f}±{sub['test_accuracy'].std():.4f}, "
              f"test_mse={sub['test_mse'].mean():.6f}" if 'test_mse' in sub.columns else "")

    print(f"\nSaved to {outputs_dir}/")