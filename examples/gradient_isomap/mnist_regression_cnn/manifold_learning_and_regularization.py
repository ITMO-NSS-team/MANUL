import os
import matplotlib.pyplot as plt
import pandas as pd

from examples.gradient_isomap.mnist_regression_cnn.baseline import baseline_train_test, CNNEncoderRegressor
from examples.gradient_isomap.mnist_regression_cnn.manifold_learning import mnist_manifold_learning
from examples.gradient_isomap.mnist_regression_cnn.manifold_regularization import manifold_regularization

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("MNIST CNN - Running Manifold Learning")
    print(f"{'=' * 60}\n")

    # Here output directory for all runs can be specified
    outputs_dir = 'outputs_60000'
    n_runs = 5

    for n in range(n_runs):
        results_folder = mnist_manifold_learning(outputs_dir)

        print(f"\n{'=' * 60}")
        print("MNIST CNN - Manifold Learning Finished")
        print(f"{'=' * 60}\n")

        # Baseline and regularized models share the same CNN encoder + regression
        # head architecture and operate on raw 28x28 pixels (reshaped internally
        # from the flat 784-dim batches) - the manifold only supplies a
        # regularization signal during training, it is not used as an input
        # projection.
        epochs = 5000
        batch_size = 1000
        learning_rate = 0.01
        early_stopping_patience = 100

        print(f'Run baseline CNN model on MNIST...')
        baseline_train_test(folder_path=results_folder,
                            baseline_model=CNNEncoderRegressor(),
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            early_stopping_patience=early_stopping_patience)
        print(f'Run regularization on MNIST CNN...')
        manifold_regularization(folder_path=results_folder,
                                model=CNNEncoderRegressor(),
                                num_epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                early_stop_patience=early_stopping_patience,
                                lambda_method='sobol')

    baselines = []
    regularized = []
    for folder in os.listdir(outputs_dir):
        if 'run' in folder:
            for exp_f in os.listdir(f'{outputs_dir}/{folder}'):
                if 'baseline' in exp_f:
                    baselines.append(f'{outputs_dir}/{folder}/{exp_f}/metrics.csv')
                if 'regularization' in exp_f:
                    regularized.append(f'{outputs_dir}/{folder}/{exp_f}/metrics.csv')

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
    plt.suptitle('MNIST CNN - Baseline vs Regularized Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    comparison_path = os.path.join(outputs_dir, 'mnist_cnn_comparison_boxplots.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Comparison plots saved to: {comparison_path}")
    combined_metrics_path = os.path.join(outputs_dir, 'mnist_cnn_combined_metrics.csv')
    all_metrics.to_csv(combined_metrics_path, index=False)
    print(f"✓ Combined metrics saved to: {combined_metrics_path}")
