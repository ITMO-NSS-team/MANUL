import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from baseline import baseline_train_test
from regularizator.GraphRegTrainer import GraphRegTrainer
from utils.utils import split_data, set_global_seed

import warnings
warnings.filterwarnings('ignore', category=FutureWarning,
                        message='unique with argument that is not not a Series')

PRETRAINED_FOLDER = 'outputs_stat_0.01noise_5k_sobol_v3\\sphere\\sphere_run_20260318_133356'

N_RUNS = 5
EPOCHS = 5000
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 100

LAMBDA_GRAPH = 1.0


def run_regularization(folder_path, model, exp_prefix="reg"):
    from torch import float64 as fl64
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(folder_path, f'{exp_prefix}_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)

    fps_indices = np.load(f'{folder_path}/fps_indices.npy')
    dist_matrix = np.load(f'{folder_path}/best_distance_matrix.npy')
    features = np.load(f'{folder_path}/all_features.npy')
    targets = np.load(f'{folder_path}/all_targets.npy')

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, targets)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = GraphRegTrainer(
        train_features=X_train,
        train_target=y_train,
        weights_matrix=dist_matrix,
        base_indices=fps_indices,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        val_features=X_val,
        val_targets=y_val,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        cache_folder=experiment_folder,
    )

    trainer.train(
        plot_convergence=(exp_prefix.startswith("final")),
        adaptive_lambda='sobol',
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        lambda_graph=LAMBDA_GRAPH,
    )

    # Evaluate
    geometry_name = os.path.basename(folder_path).split('_')[0]
    metrics = {'geometry': geometry_name, 'experiment_type': 'Regularized'}

    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test),
    ]:
        t_x = torch.tensor(X_split, dtype=fl64).to(device)
        preds = trainer.best_model(t_x).flatten().cpu().detach().numpy()
        y_true = y_split.flatten()
        metrics[f'{split_name}_mse'] = mean_squared_error(y_true, preds)
        metrics[f'{split_name}_mae'] = mean_absolute_error(y_true, preds)
        metrics[f'{split_name}_r2'] = r2_score(y_true, preds)

    pd.DataFrame([metrics]).to_csv(os.path.join(experiment_folder, 'metrics.csv'), index=False)
    return metrics, experiment_folder


if __name__ == "__main__":
    assert os.path.exists(PRETRAINED_FOLDER), f"Not found: {PRETRAINED_FOLDER}"
    geometry_name = os.path.basename(PRETRAINED_FOLDER).split('_')[0]

    print(f"\n{'=' * 60}")
    print(f"{geometry_name.upper()}: Baseline vs GraphReg (sobol)")
    print(f"  Pretrained: {PRETRAINED_FOLDER}")
    print(f"  N_RUNS={N_RUNS}, EPOCHS={EPOCHS}")
    print(f"{'=' * 60}")

    def make_model():
        return nn.Sequential(
            nn.Linear(3, 32, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 1, dtype=torch.float64),
        )

    all_metrics = []

    for n in range(N_RUNS):
        run_seed = 100 + n
        print(f"\n--- Run {n + 1}/{N_RUNS} (seed={run_seed}) ---")

        # Baseline
        print(f'  Baseline...')
        set_global_seed(run_seed)
        baseline_train_test(
            folder_path=PRETRAINED_FOLDER,
            baseline_model=make_model(),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )

        # Regularized
        print(f'  Regularized...')
        set_global_seed(run_seed)
        metrics_reg, _ = run_regularization(
            folder_path=PRETRAINED_FOLDER,
            model=make_model(),
            exp_prefix="final_reg",
        )
        all_metrics.append(metrics_reg)

    print(f"\nAggregating results...")

    for exp_f in sorted(os.listdir(PRETRAINED_FOLDER)):
        if not exp_f.startswith('baseline_'):
            continue
        metrics_path = os.path.join(PRETRAINED_FOLDER, exp_f, 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            if 'experiment_type' not in df.columns:
                df['experiment_type'] = 'Baseline'
            all_metrics.append(df.to_dict('records')[0])

    combined_df = pd.DataFrame(all_metrics)

    baseline_df = combined_df[combined_df['experiment_type'] == 'Baseline'].tail(N_RUNS)
    reg_df = combined_df[combined_df['experiment_type'] == 'Regularized']
    combined_df = pd.concat([baseline_df, reg_df], ignore_index=True)

    # Save
    config_tag = f'lam_{LAMBDA_GRAPH}'
    outputs_dir = os.path.dirname(PRETRAINED_FOLDER)
    combined_df.to_csv(os.path.join(outputs_dir, f'{geometry_name}_{config_tag}_metrics.csv'), index=False)

    metrics_to_plot = ['test_mse', 'test_mae', 'test_r2',
                       'val_mse', 'val_mae', 'val_r2',
                       'train_mse', 'train_mae', 'train_r2']
    experiment_types = ['Baseline', 'Regularized']
    colors = {'Baseline': '#1f77b4', 'Regularized': '#2ca02c'}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        data_to_plot = [combined_df[combined_df['experiment_type'] == et][metric].values
                        for et in experiment_types]
        bp = ax.boxplot(data_to_plot, labels=experiment_types, patch_artist=True)
        for patch, et in zip(bp['boxes'], experiment_types):
            patch.set_facecolor(colors[et])
            patch.set_alpha(0.4)
        for i, et in enumerate(experiment_types):
            d = combined_df[combined_df['experiment_type'] == et][metric]
            ax.scatter(np.random.normal(i + 1, 0.04, size=len(d)), d,
                       alpha=0.7, s=40, edgecolors='black', color=colors[et], zorder=5)
        ax.set_title(metric.replace('_', ' ').upper(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{geometry_name.upper()}: Baseline vs Regularized (Sobol)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, f'{geometry_name}_{config_tag}_comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS (mean)")
    print(f"{'=' * 60}")
    summary = combined_df.groupby('experiment_type')[
        ['test_mse', 'test_mae', 'test_r2']].mean()
    summary = summary.reindex(['Baseline', 'Regularized'])
    print(summary.to_string(float_format=lambda x: f'{x:.6f}'))
    print(f"\nSaved to {outputs_dir}/")
