import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Adam.GradientIsomap import GradientIsomap
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.utils import split_data, set_global_seed

from dataset_loader import load_dataset, get_dataset_info


def visualize_projections(projections, labels, title, save_path, n_classes=10):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc = ax.scatter(projections[:, 0], projections[:, 1],
                    c=labels, cmap='tab10', alpha=0.6, s=8,
                    vmin=0, vmax=n_classes - 1, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, ticks=range(n_classes))
    cbar.set_label('Class')
    ax.set_xlabel('Latent dim 0')
    ax.set_ylabel('Latent dim 1')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {save_path}")


def manifold_learning_pipeline(dataset_name, working_folder, data_dir='./data',
                                n_samples=10000, n_base_points=1000,
                                latent_dim=None, epochs=5000,
                                random_state=42):
    """
    Universal manifold learning pipeline.

    Args:
        dataset_name: 'mnist', 'fmnist', or 'cifar10'
        working_folder: output directory
        data_dir: root for data download
        n_samples: subset size
        n_base_points: FPS points
        latent_dim: intrinsic dim (None = use default from dataset_loader)
        epochs: GradientIsomap epochs
        random_state: seed
    """
    info = get_dataset_info(dataset_name)
    if latent_dim is None:
        latent_dim = info['latent_dim']
    n_classes = info['n_classes']
    proj_method = 'random_forest'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"{dataset_name.upper()} MANIFOLD LEARNING PIPELINE")
    print("=" * 80)

    set_global_seed(random_state)
    X, y = load_dataset(dataset_name, data_dir, n_samples, random_state)

    np.save(f'{working_folder}/all_features.npy', X)
    np.save(f'{working_folder}/all_targets.npy', y)

    print(f"\nSplitting data into train/val/test (70%/15%/15%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, (0.7, 0.15, 0.15))
    print(f"  Training: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # === FPS ===
    print("\n=== FPS SAMPLING ===")
    fps_path = f'{working_folder}/fps_indices.npy'
    if os.path.exists(fps_path):
        fps_indices = np.load(fps_path)
        fps_time = '0'
    else:
        start_time = time.time()
        fps_indices = memory_efficient_fps(features=X_train, n_samples=n_base_points, batch_size=500)
        np.save(fps_path, fps_indices)
        fps_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"  Base points: {len(fps_indices)} out of {X_train.shape[0]}")

    # === MANIFOLD LEARNING ===
    print("\n=== MANIFOLD LEARNING ===")
    train_features_fps = torch.tensor(X_train[fps_indices], dtype=torch.float32).to(device)

    # One-hot targets for FPS base points
    fps_labels = y_train[fps_indices].astype(int)
    fps_onehot = np.eye(n_classes, dtype=np.float32)[fps_labels]
    train_target_fps = torch.tensor(fps_onehot, dtype=torch.float32).to(device)

    print(f"  FPS features: {train_features_fps.shape}")
    print(f"  FPS targets (one-hot): {train_target_fps.shape}")
    print(f"  Training GradientIsomap (latent_dim={latent_dim}, epochs={epochs})...")

    start_time = time.time()
    isomap = GradientIsomap(
        train_feature=train_features_fps,
        train_target=train_target_fps,
        latent_len=latent_dim,
        checkpoint_each=100,
        save_checkpoint_matrix=False,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=epochs,
        stop_criteria_value=0.001
    )
    isomap.train()
    isomap_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

    base_projection = isomap.best_isomap_model().detach().cpu().numpy()
    np.save(f'{working_folder}/base_projection.npy', base_projection)

    # === PROJECTIONS ===
    print("\n=== COMPUTING PROJECTIONS ===")
    X_base = X_train[fps_indices]

    for split_name, X_split in [('train', X_train), ('val', X_val), ('test', X_test)]:
        print(f"  Computing {split_name} projections...")
        start_time = time.time()

        if split_name == 'train':
            projector = Projector(source_data=X_split, base_indices=fps_indices,
                                  batch_size=1024, base_projection=base_projection)
            proj = projector.compute_projection(method=proj_method)
        else:
            X_combined = np.vstack([X_base, X_split])
            base_idx = np.arange(len(fps_indices))
            projector = Projector(source_data=X_combined, base_indices=base_idx,
                                  batch_size=1024, base_projection=base_projection)
            proj_full = projector.compute_projection(method=proj_method)
            proj = proj_full[len(fps_indices):]

        np.save(os.path.join(working_folder, f'{split_name}_projections.npy'), proj)
        t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"    {split_name}: {proj.shape} ({t})")

    # === VISUALIZATIONS ===
    print("\n=== VISUALIZING PROJECTIONS ===")
    train_proj = np.load(os.path.join(working_folder, 'train_projections.npy'))
    test_proj = np.load(os.path.join(working_folder, 'test_projections.npy'))

    visualize_projections(train_proj, y_train,
                          f'{dataset_name.upper()} Train — Manifold Projection (n={len(y_train)})',
                          os.path.join(working_folder, 'manifold_projection_train.png'), n_classes)
    visualize_projections(test_proj, y_test,
                          f'{dataset_name.upper()} Test — Manifold Projection (n={len(y_test)})',
                          os.path.join(working_folder, 'manifold_projection_test.png'), n_classes)

    # Combined
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, proj, lab, name in [(axes[0], train_proj, y_train, 'Train'),
                                 (axes[1], test_proj, y_test, 'Test')]:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=lab, cmap='tab10',
                         alpha=0.6, s=8, vmin=0, vmax=n_classes - 1)
        ax.set_title(f'{name} (n={len(lab)})')
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=axes, ticks=range(n_classes), shrink=0.8)
    plt.suptitle(f'{dataset_name.upper()} — Learned Manifold Projections', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(working_folder, 'manifold_projections_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # === METADATA ===
    metadata = {
        'dataset_type': dataset_name,
        'n_samples': n_samples,
        'n_base_points': n_base_points,
        'latent_dim': latent_dim,
        'random_seed': random_state,
        'input_dim': info['input_dim'],
        'n_classes': n_classes,
    }
    with open(f'{working_folder}/experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame({
        'Parameter': list(metadata.keys()) + ['fps_time', 'isomap_time'],
        'Value': list(metadata.values()) + [fps_time, isomap_time]
    }).to_csv(f'{working_folder}/metadata.csv', index=False)

    print(f"\n{dataset_name.upper()} manifold learning complete!")
    return working_folder
