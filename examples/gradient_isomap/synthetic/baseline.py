import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.utils import split_data


def baseline_train_test(folder_path, baseline_model, epochs, batch_size, learning_rate, early_stopping_patience):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    geometry_name = os.path.basename(folder_path).split('_')[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    baseline_folder = os.path.join(folder_path, f'baseline_{timestamp}')
    os.makedirs(baseline_folder, exist_ok=True)

    print(f"✓ Experiment folder: {baseline_folder}")
    print(f"✓ Geometry: {geometry_name}")
    print(f"✓ Device: {device}")

    print("\n📂 Loading data...")
    try:
        features = np.load(f'{folder_path}/all_features.npy')
        targets = np.load(f'{folder_path}/all_targets.npy')
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, targets)
    except Exception as e:
        print(f'Error loading data from {folder_path}: {e}')
        print('Required files:\n1) all_features.npy\n2) all_targets.npy')
        print("\n❌ Please run manifold learning or specify the correct folder!")
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
        shuffle=True,
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
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss={avg_train_loss:.6f}, '
                  f'Val Loss={avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = baseline_model.state_dict().copy()
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

    print(f"\n{'=' * 40}")
    print("FINAL RESULTS")
    print(f"{'=' * 40}")
    print(f"Training - MSE: {baseline_train_mse:.6f}, MAE: {baseline_train_mae:.6f}, R²: {baseline_train_r2:.6f}")
    print(f"Validation - MSE: {baseline_val_mse:.6f}, MAE: {baseline_val_mae:.6f}, R²: {baseline_val_r2:.6f}")
    print(f"Test - MSE: {baseline_test_mse:.6f}, MAE: {baseline_test_mae:.6f}, R²: {baseline_test_r2:.6f}")

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
        'geometry': geometry_name,
        'train_mse': baseline_train_mse,
        'train_mae': baseline_train_mae,
        'train_r2': baseline_train_r2,
        'val_mse': baseline_val_mse,
        'val_mae': baseline_val_mae,
        'val_r2': baseline_val_r2,
        'test_mse': baseline_test_mse,
        'test_mae': baseline_test_mae,
        'test_r2': baseline_test_r2
    }])
    metrics_df.to_csv(os.path.join(baseline_folder, 'metrics.csv'), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_train_losses, label='Training Loss', linewidth=2)
    plt.plot(baseline_val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{geometry_name.upper()} - Baseline Model Training')
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

    fig = plt.figure(figsize=(16, 6))

    # Left: True values
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                      c=test_targets, alpha=0.8)
    ax1.set_title('True Values', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(sc1, ax=ax1, shrink=0.6, label='Value')

    # Middle: Predicted values
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                      c=test_preds, alpha=0.8)
    ax2.set_title('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Value')

    # Right: Errors
    ax3 = fig.add_subplot(133, projection='3d')
    errors = np.abs(test_targets - test_preds)
    sc3 = ax3.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                      c=errors, cmap='hot_r', alpha=0.8)
    ax3.set_title('Absolute Errors', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(sc3, ax=ax3, shrink=0.6, label='|Error|')

    plt.suptitle(f'{geometry_name.upper()} - Baseline Model Predictions',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    side_by_side_path = os.path.join(baseline_folder, '3d_side_by_side.png')
    plt.savefig(side_by_side_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Side-by-side comparison saved to {side_by_side_path}")

    return baseline_folder


if __name__ == "__main__":
    folder_path = 'outputs/torus_run_20260103_182935'
    model_architecture = nn.Sequential(
        nn.Linear(3, 32, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(32, 1, dtype=torch.float64)
    )
    baseline_train_test(folder_path, model_architecture)
