import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from torchvision import datasets
from utils.utils import set_global_seed


DATASET_INFO = {
    'mnist': {
        'input_dim': 784,
        'n_classes': 10,
        'hidden_dim': 128,
        'latent_dim': 127,
        'loader': lambda data_dir: datasets.MNIST(root=data_dir, train=True, download=True),
        'shape': (28, 28),
    },
    'fmnist': {
        'input_dim': 784,
        'n_classes': 10,
        'hidden_dim': 128,
        'latent_dim': 199,
        'loader': lambda data_dir: datasets.FashionMNIST(root=data_dir, train=True, download=True),
        'shape': (28, 28),
    },
    'cifar10': {
        'input_dim': 3072,
        'n_classes': 10,
        'hidden_dim': 256,
        'latent_dim': 284,
        'loader': lambda data_dir: datasets.CIFAR10(root=data_dir, train=True, download=True),
        'shape': (32, 32, 3),
    },
}


def load_dataset(dataset_name, data_dir='./data', n_samples=10000, random_state=42):
    """
    Load and subsample a dataset.

    Args:
        dataset_name: 'mnist', 'fmnist', or 'cifar10'
        data_dir: root directory for data download
        n_samples: number of samples to subsample (stratified)
        random_state: random seed

    Returns:
        X: [n_samples, D] float64, normalized to [0, 1]
        y: [n_samples] int64, labels 0-9
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[dataset_name]
    set_global_seed(random_state)

    ds = info['loader'](data_dir)

    # Extract features and labels
    if dataset_name == 'cifar10':
        X_full = ds.data.reshape(len(ds.data), -1).astype(np.float64) / 255.0
        y_full = np.array(ds.targets, dtype=np.int64)
    else:
        X_full = ds.data.numpy().reshape(len(ds), -1).astype(np.float64) / 255.0
        y_full = ds.targets.numpy().astype(np.int64)

    # Stratified subsample
    if n_samples < len(X_full):
        X, _, y, _ = train_test_split(
            X_full, y_full,
            train_size=n_samples,
            random_state=random_state,
            stratify=y_full
        )
    else:
        X, y = X_full, y_full

    print(f"  {dataset_name.upper()} loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Feature range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


def get_dataset_info(dataset_name):
    """Get metadata for a dataset."""
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_INFO.keys())}")
    return DATASET_INFO[dataset_name]


def labels_to_onehot(labels, n_classes=10):
    """Convert integer labels [N] to one-hot float64 [N, n_classes]."""
    return np.eye(n_classes, dtype=np.float64)[labels.astype(int)]


def evaluate_classifier(model, X, y_int, n_classes, device):
    """
    Evaluate a classification model. Returns (accuracy, mse).
    MSE = mean_squared_error(one_hot_targets, raw_model_outputs).
    """
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float64).to(device)).cpu().numpy()
    pred_labels = np.argmax(preds, axis=1)
    y_oh = labels_to_onehot(y_int, n_classes)
    acc = accuracy_score(y_int, pred_labels)
    mse = mean_squared_error(y_oh, preds)
    return acc, mse
