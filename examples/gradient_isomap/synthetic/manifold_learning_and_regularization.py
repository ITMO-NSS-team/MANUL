import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from examples.gradient_isomap.synthetic.baseline import baseline_train_test
from examples.gradient_isomap.synthetic.manifold_learning import synthetic_manifold_learning_pipeline
from examples.gradient_isomap.synthetic.manifold_regularization import manifold_regularization

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("Running Manifold Learning")
    print(f"{'=' * 60}\n")

    # Here output directory for all runs can be specified
    outputs_dir = 'outputs_stat_0.01noise_5k_sobol_v3'
    n_runs = 5

    geometries_to_process = [
        'sphere',
        'torus',
        'swiss_roll',
        'swiss_hole',
        'pseudosphere',
        'hyperboloid',
        'helicoid',
        'multi_scale_torus',
        'nonuniform_sphere',
        'cone_surface',
        'genus_2_surface',
        's_curve'
    ]

    for geom in geometries_to_process:
        geom_folder = f'{outputs_dir}/{geom}'
        baselines = []
        regularized = []
        for n in range(n_runs):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            working_folder = f'{geom_folder}/{geom}_run_{timestamp}'
            os.makedirs(working_folder, exist_ok=True)
            results_folder = synthetic_manifold_learning_pipeline(geom, working_folder)

            model_architecture = [nn.Linear(3, 32, dtype=torch.float64),
                                  nn.ReLU(),
                                  nn.Linear(32, 1, dtype=torch.float64)]
            epochs = 5000
            batch_size = 2048
            learning_rate = 1e-3
            early_stopping_patience = 100

            print(f'Run baseline model on {geom}...')
            baseline_folder = baseline_train_test(folder_path=results_folder,
                                                  baseline_model=nn.Sequential(*model_architecture),
                                                  epochs=epochs,
                                                  batch_size=batch_size,
                                                  learning_rate=learning_rate,
                                                  early_stopping_patience=early_stopping_patience)
            print(f'Run regularization on {geom}...')
            reg_folder = manifold_regularization(folder_path=results_folder,
                                                 model=nn.Sequential(*model_architecture),
                                                 num_epochs=epochs,
                                                 batch_size=batch_size,
                                                 learning_rate=learning_rate,
                                                 early_stop_patience=early_stopping_patience,
                                                 lambda_method='sobol')

        for folder in os.listdir(geom_folder):
            if 'run' in folder:
                for exp_f in os.listdir(f'{geom_folder}/{folder}'):
                    if 'baseline' in exp_f:
                        baselines.append(f'{geom_folder}/{folder}/{exp_f}/metrics.csv')
                    if 'regularization' in exp_f:
                        regularized.append(f'{geom_folder}/{folder}/{exp_f}/metrics.csv')

        all_metrics_list = []
        for n in range(len(baselines)):
            baseline_metrics = pd.read_csv(baselines[n])
            regularized_metrics = pd.read_csv(regularized[n])

            baseline_metrics['experiment_type'] = 'Baseline'
            baseline_metrics['run'] = n
            regularized_metrics['experiment_type'] = 'Regularized'
            regularized_metrics['run'] = n
            all_metrics_list.append(baseline_metrics)
            all_metrics_list.append(regularized_metrics)
        all_metrics = pd.concat(all_metrics_list, ignore_index=True)

        metrics_to_plot = ['train_mse', 'train_mae', 'train_r2',
                           'val_mse', 'val_mae', 'val_r2',
                           'test_mse', 'test_mae', 'test_r2']

        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            all_metrics.boxplot(column=metric, by='experiment_type', ax=ax)
            for i, exp_type in enumerate(['Baseline', 'Regularized']):
                exp_data = all_metrics[all_metrics['experiment_type'] == exp_type][metric]
                x_pos = i + 1
                ax.scatter([x_pos] * len(exp_data), exp_data, alpha=0.6, s=30)
            ax.set_title(f'{metric.upper()}')
            ax.set_xlabel('')
            ax.set_ylabel('Value')
            ax.get_figure().suptitle('')
        plt.suptitle(f'{geom.upper()} - Baseline vs Regularized Comparison',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        comparison_path = os.path.join(geom_folder, f'{geom}_comparison_boxplots.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Comparison plots saved to: {comparison_path}")
        combined_metrics_path = os.path.join(geom_folder, f'{geom}_combined_metrics.csv')
        all_metrics.to_csv(combined_metrics_path, index=False)
        print(f"✓ Combined metrics saved to: {combined_metrics_path}")
