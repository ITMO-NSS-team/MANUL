import copy
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.utils import split_data


class CNNEncoderRegressor(nn.Module):
    """
    Compact CNN encoder (conv layers down to a small bottleneck code) followed
    by a small regression head - the CNN counterpart of the fully-connected
    baseline used in mnist_regression, so it can be dropped into the same
    baseline/regularization pipeline without changing any of the shared
    training code (it consumes flat [N, 784] batches and reshapes internally).
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, dtype=torch.float64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1, dtype=torch.float64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64, dtype=torch.float64),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 1, dtype=torch.float64),
            # no activation on the regression output - a terminal ReLU is a
            # dead-end: once its pre-activation goes negative, the local
            # gradient is exactly 0, permanently blocking gradient to every
            # earlier layer. Confirmed empirically on the MLP counterpart
            # of this pipeline (regularization-investigation Дополнение 7):
            # 6/15 runs collapsed to exactly R2=-2.391755, the analytic
            # score for "always predict 0" on this data split.
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        code = self.encoder(x)
        return self.regressor(code)


def calc_accuracy(targets, predictions):
    """
    Calculate accuracy for regression by rounding predictions to nearest integer.
    """
    predictions_rounded = np.round(predictions).astype(int)
    targets_int = targets.astype(int)
    correct = np.sum(predictions_rounded == targets_int)
    accuracy = correct / len(targets_int)

    return accuracy

def baseline_train_test(folder_path, baseline_model, epochs, batch_size, learning_rate, early_stopping_patience):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    baseline_folder = os.path.join(folder_path, f'baseline_{timestamp}')
    os.makedirs(baseline_folder, exist_ok=True)

    print(f"✓ Experiment folder: {baseline_folder}")
    print(f"✓ Device: {device}")

    print("\n📂 Loading data...")
    try:
        n_samples = int(folder_path.split('/')[-2].split('_')[-1])
        mnist_dataset = datasets.MNIST(root='../data', train=True, download=True)
        X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32) / 255.0
        y = mnist_dataset.targets.numpy()
        X = X[:n_samples]
        y = y[:n_samples]
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    except Exception as e:
        print(f'Error loading data from {folder_path}: {e}')
        print(f'N samples found wrong: {n_samples}')
        raise

    # Convert to PyTorch tensors and create datasets
    print("Creating PyTorch datasets...")

    # Training dataset
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Validation dataset
    X_val_tensor = torch.tensor(X_val, dtype=torch.float64)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float64).reshape(-1, 1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Test dataset
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64).reshape(-1, 1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=False: matches GraphRegTrainer.train()'s fixed-order
        # batches (it can't shuffle - it needs batch_indices to stay real
        # dataset indices for landmark lookups), so the baseline vs
        # regularized comparison isn't confounded by batch-order.
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Batches per epoch: {len(train_loader)}")

    print(f"\n{'=' * 40}")
    print("TRAINING BASELINE MODEL")
    print(f"{'=' * 40}")

    # Model, loss, optimizer
    baseline_model = baseline_model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)

    # Training setup
    baseline_train_losses, baseline_val_losses = [], []
    time_list = []
    best_val_loss, patience_counter = float('inf'), 0
    best_epoch, best_model_state = 0, None

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        baseline_model.train()
        epoch_train_loss = 0.0

        # Training phase
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = baseline_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        baseline_train_losses.append(avg_train_loss)
        current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        time_list.append(current_time)

        # Validation phase
        baseline_model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = baseline_model(batch_x)
                loss = criterion(output, batch_y)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        baseline_val_losses.append(avg_val_loss)

        # Progress logging
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss={avg_train_loss:.6f}, '
                  f'Val Loss={avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # .copy() only shallow-copies the dict - the tensors inside still
            # alias the live model's storage, which optimizer.step() mutates
            # in place. deepcopy is required for a real, independent snapshot.
            best_model_state = copy.deepcopy(baseline_model.state_dict())
            best_epoch = epoch + 1
        else:
            patience_counter += 1

            if epoch > 0 and avg_val_loss > baseline_val_losses[-1]:
                patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            print(f'Validation stopped improving {patience_counter} epochs ago')
            print(f'Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}')
            break


    if best_model_state is not None:
        baseline_model.load_state_dict(best_model_state)
        print(f"✓ Loaded best model from epoch {best_epoch}")

    baseline_model.eval()

    all_train_preds, all_train_targets = [], []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        output = baseline_model(batch_x)
        all_train_preds.append(output.cpu().detach().numpy())
        all_train_targets.append(batch_y.numpy())

    train_preds = np.concatenate(all_train_preds, axis=0).flatten()
    train_targets = np.concatenate(all_train_targets, axis=0).flatten()

    baseline_train_mse = mean_squared_error(train_targets, train_preds)
    baseline_train_mae = mean_absolute_error(train_targets, train_preds)
    baseline_train_r2 = r2_score(train_targets, train_preds)
    baseline_train_accuracy = calc_accuracy(train_targets, train_preds)

    print("\nFinal evaluation...")
    with torch.no_grad():
        all_val_preds, all_val_targets = [], []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            output = baseline_model(batch_x)
            all_val_preds.append(output.cpu().numpy())
            all_val_targets.append(batch_y.numpy())

        val_preds = np.concatenate(all_val_preds, axis=0).flatten()
        val_targets = np.concatenate(all_val_targets, axis=0).flatten()

        baseline_val_mse = mean_squared_error(val_targets, val_preds)
        baseline_val_mae = mean_absolute_error(val_targets, val_preds)
        baseline_val_r2 = r2_score(val_targets, val_preds)
        baseline_val_accuracy = calc_accuracy(val_targets, val_preds)

        # Test set
        all_test_preds, all_test_targets = [], []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = baseline_model(batch_x)
            all_test_preds.append(output.cpu().numpy())
            all_test_targets.append(batch_y.numpy())

        test_preds = np.concatenate(all_test_preds, axis=0).flatten()
        test_targets = np.concatenate(all_test_targets, axis=0).flatten()

        baseline_test_mse = mean_squared_error(test_targets, test_preds)
        baseline_test_mae = mean_absolute_error(test_targets, test_preds)
        baseline_test_r2 = r2_score(test_targets, test_preds)
        baseline_test_accuracy = calc_accuracy(test_targets, test_preds)


    print(f"\n{'=' * 40}")
    print("FINAL RESULTS")
    print(f"{'=' * 40}")
    print(f"Training - MSE: {baseline_train_mse:.6f}, MAE: {baseline_train_mae:.6f}, R²: {baseline_train_r2:.6f}, accuracy: {baseline_train_accuracy}")
    print(f"Validation - MSE: {baseline_val_mse:.6f}, MAE: {baseline_val_mae:.6f}, R²: {baseline_val_r2:.6f}, accuracy: {baseline_val_accuracy}")
    print(f"Test - MSE: {baseline_test_mse:.6f}, MAE: {baseline_test_mae:.6f}, R²: {baseline_test_r2:.6f}, accuracy: {baseline_test_accuracy}")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(baseline_train_losses) + 1)),
        'time_spent': time_list,
        'train_loss': baseline_train_losses,
        'val_loss': baseline_val_losses
    })
    history_df.to_csv(os.path.join(baseline_folder, 'convergence_log.csv'), index=False)

    # Save metrics
    metrics_df = pd.DataFrame([{
        'train_mse': baseline_train_mse,
        'train_mae': baseline_train_mae,
        'train_r2': baseline_train_r2,
        'train_accuracy': baseline_train_accuracy,
        'val_mse': baseline_val_mse,
        'val_mae': baseline_val_mae,
        'val_r2': baseline_val_r2,
        'val_accuracy': baseline_val_accuracy,
        'test_mse': baseline_test_mse,
        'test_mae': baseline_test_mae,
        'test_r2': baseline_test_r2,
        'test_accuracy': baseline_test_accuracy
    }])
    metrics_df.to_csv(os.path.join(baseline_folder, 'metrics.csv'), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_train_losses, label='Training Loss', linewidth=2)
    plt.plot(baseline_val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'MNIST - CNN Baseline Model Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(baseline_folder, 'training_plot.png'), dpi=150)
    plt.close()

    print(f"\n✓ Results saved to: {baseline_folder}")
    print(f"  - Model: best_model.pth")
    print(f"  - Metrics: metrics.csv")
    print(f"  - History: training_history.csv")
    print(f"  - Plot: training_plot.png")


    return baseline_folder


if __name__ == "__main__":
    folder_path = 'outputs_60000/mnist_run_20260115_234826'

    baseline_train_test(folder_path=folder_path,
                        baseline_model=CNNEncoderRegressor(),
                        epochs=5000,
                        batch_size=1000,
                        learning_rate=0.001,  # 0.01 collapsed training on the MLP counterpart, see baseline note above
                        early_stopping_patience=100)
