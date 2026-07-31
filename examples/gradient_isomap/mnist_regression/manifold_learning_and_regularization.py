import pandas as pd
import torch
from torch import nn

from examples.gradient_isomap.mnist_regression.manifold_learning import mnist_manifold_learning

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("MNIST - Running Manifold Learning")
    print(f"{'=' * 60}\n")
    # Here output directory for all runs can be specified
    outputs_dir = 'outputs_60000'
    n_runs = 5
    results_folder = mnist_manifold_learning(outputs_dir)

    print(f"\n{'=' * 60}")
    print("MNIST - Manifold Learning Finished")
    print(f"{'=' * 60}\n")
    metadata = pd.read_csv(f'{results_folder}/metadata.csv')['Value']
    metadata = metadata['Value'][metadata['Parameter'] == 'Latent dimension']
    latent_dim = metadata.values[0]

    model_architecture = [nn.Linear(latent_dim, 32, dtype=torch.float64),
                          nn.ReLU(),
                          nn.Linear(32, 1, dtype=torch.float64)]
    epochs = 5000
    batch_size = 2048
    learning_rate = 1e-3
    early_stopping_patience = 100

