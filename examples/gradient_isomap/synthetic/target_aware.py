import os
import json
import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Adam.GradientIsomapTargetAware import GradientIsomapTargetAware
from Adam.visualization_utils import original_visualization_simple
from structure_approximation.IntrinsicNN import IntrinsicNN
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.utils import split_data
from data.synthetic_geometries import geometries, noisy_manifold

n_samples = 5000
n_base_points = 1000
noise_percent = 0.01
latent_dim = 2
epochs = 5000
n_neighbors = 25
device = 'cuda'

target_weight = 0.1
epsilon = 0.01

GEOMETRY_NAMES = ['torus']
MODES = ['baseline','l2', 'add', 'multiply']


def evaluate_with_intrinsic_nn(train_proj, train_targets, test_proj, test_targets,
                               latent_dim, device='cuda'):
    train_proj_t = torch.tensor(train_proj, dtype=torch.float32).to(device)
    train_targets_t = torch.tensor(train_targets, dtype=torch.float32).to(device)
    test_proj_t = torch.tensor(test_proj, dtype=torch.float32).to(device)

    task_model = IntrinsicNN(
        train_proj_t,
        train_targets_t,
        latent_dim,
        plot_convergence=False,
        epochs=500
    )
    task_model.train()

    train_pred = task_model.model(train_proj_t).detach().cpu().numpy().flatten()
    train_loss = np.mean((train_pred - train_targets.flatten()) ** 2)

    test_pred = task_model.model(test_proj_t).detach().cpu().numpy().flatten()
    test_loss = np.mean((test_pred - test_targets.flatten()) ** 2)

    return train_loss, test_loss, train_pred, test_pred


def run_single_experiment(geometry_name, mode, working_folder):
    print("=" * 80)
    print(f"GEOMETRY: {geometry_name} | MODE: {mode}")
    print("=" * 80)

    print(f"Working folder: {working_folder}")
    print(f"Generating {n_samples} points for {geometry_name} with {noise_percent * 100}% noise...")

    geometry_function = geometries[geometry_name][0]
    X, y = noisy_manifold(geometry_function, noise_percent=noise_percent, n_samples=n_samples)
    np.save(f'{working_folder}/all_features.npy', X)
    np.save(f'{working_folder}/all_targets.npy', y)

    print(f"  Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"  Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    print("\nSplitting data into train/val/test (70%/15%/15%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, (0.7, 0.15, 0.15))

    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\n=== FPS SAMPLING ===")
    fps_path = f'{working_folder}/fps_indices.npy'
    if os.path.exists(fps_path):
        fps_indices = np.load(fps_path)
        fps_extract_time = "loaded"
        print(f'FPS indices loaded from {fps_path}')
    else:
        start_time = time.time()
        fps_indices = memory_efficient_fps(features=X_train, n_samples=n_base_points, batch_size=500)
        np.save(fps_path, fps_indices)
        fps_extract_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'FPS indices saved to {fps_path}')

    print("\n=== MANIFOLD LEARNING ===")
    train_features = torch.tensor(X_train[fps_indices], dtype=torch.float32).to(device)
    train_target = torch.tensor(y_train[fps_indices], dtype=torch.float32).to(device)

    print(f"Training on {device}: {train_features.shape[0]} points, latent_dim={latent_dim}")

    start_time = time.time()

    isomap = GradientIsomapTargetAware(
        train_feature=train_features,
        train_target=train_target,
        latent_len=latent_dim,
        n_neighbors=n_neighbors,
        checkpoint_each=100,
        save_checkpoint_matrix=False,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=epochs,
        stop_criteria_value=0.001,
        use_target_modification=(mode != 'baseline'),
        target_weight=target_weight,
        epsilon=epsilon,
        modification_mode=mode if mode != 'baseline' else 'add'
    )

    isomap.train(use_init_assumption=False)
    isomap_train_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    isomap.visualize_trained()

    base_projection = isomap.best_isomap_model().detach().cpu().numpy()
    np.save(f'{working_folder}/base_projection.npy', base_projection)
    print(f"Saved base projections {working_folder}/base_projection.npy")

    print("\n=== COMPUTING PROJECTIONS ===")

    # Combine train and test data for single model fitting
    n_train = len(X_train)
    X_combined = np.vstack([X_train, X_test])

    print(f"  Projecting all data ({X_combined.shape[0]} points) using Projector...")
    projector = Projector(
        source_data=X_combined,
        base_indices=fps_indices,
        base_projection=base_projection,
        batch_size=1024
    )
    all_projections = projector.compute_projection(method='random_forest')

    # Split projections back
    train_projections = all_projections[:n_train]
    test_projections = all_projections[n_train:]

    np.save(f'{working_folder}/train_projections.npy', train_projections)
    np.save(f'{working_folder}/test_projections.npy', test_projections)
    print(f"  Train projections: {train_projections.shape}")
    print(f"  Test projections: {test_projections.shape}")

    print("\n=== EVALUATION (IntrinsicNN) ===")

    train_loss, test_loss, train_pred, test_pred = evaluate_with_intrinsic_nn(
        train_projections, y_train,
        test_projections, y_test,
        latent_dim, device
    )

    print(f"  Train MSE: {train_loss:.6f}")
    print(f"  Test MSE:  {test_loss:.6f}")

    print("\n=== VISUALIZATION ===")
    original_visualization_simple(
        X_test,
        y_test,
        test_pred,
        save_path=f'{working_folder}/prediction_test.png'
    )
    print(f"  Saved: {working_folder}/prediction_test.png")

    isomap_train_loss = isomap.best_loss
    metadata = pd.DataFrame({
        'Parameter': ['Geometry', 'Mode', 'Total samples', 'Base points',
                      'Noise level', 'Latent dim', 'Device', 'FPS time',
                      'Train time', 'Isomap train loss', 'Train MSE', 'Test MSE'],
        'Value': [geometry_name, mode, n_samples, n_base_points,
                  noise_percent, latent_dim, device, fps_extract_time,
                  isomap_train_time, f"{isomap_train_loss:.6f}",
                  f"{train_loss:.6f}", f"{test_loss:.6f}"]
    })
    metadata.to_csv(f'{working_folder}/metadata.csv', index=False)

    print("\nConfiguration:")
    for _, row in metadata.iterrows():
        print(f"  {row['Parameter']}: {row['Value']}")

    print(f"\n>>> ISOMAP TRAIN LOSS: {isomap_train_loss:.6f}")
    print(f">>> TRAIN MSE: {train_loss:.6f}")
    print(f">>> TEST MSE: {test_loss:.6f}")
    print(f">>> Geometry {geometry_name} / Mode {mode} complete!")

    return {
        'geometry': geometry_name,
        'mode': mode,
        'isomap_train_loss': float(isomap_train_loss),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'train_time': isomap_train_time,
        'folder': working_folder
    }


def create_summary(all_results, output_dir):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n=== ISOMAP TRAIN LOSS ===")
    print(f"{'Geometry':<15} {'Baseline':<12} {'Multiply':<12} {'L2':<12} {'Add':<12}")
    print("-" * 65)

    for geom in GEOMETRY_NAMES:
        row = f"{geom:<15}"
        for mode in MODES:
            for r in all_results:
                if r['geometry'] == geom and r['mode'] == mode:
                    row += f" {r['isomap_train_loss']:<12.6f}"
                    break
        print(row)

    print(f"\n=== TEST MSE ===")
    print(f"{'Geometry':<15} {'Baseline':<12} {'Multiply':<12} {'L2':<12} {'Add':<12}")
    print("-" * 65)

    for geom in GEOMETRY_NAMES:
        row = f"{geom:<15}"
        for mode in MODES:
            for r in all_results:
                if r['geometry'] == geom and r['mode'] == mode:
                    row += f" {r['test_loss']:<12.6f}"
                    break
        print(row)

    print("-" * 65)

    print("\nBest method per geometry (by TEST MSE):")
    for geom in GEOMETRY_NAMES:
        geom_results = [r for r in all_results if r['geometry'] == geom]
        best = min(geom_results, key=lambda x: x['test_loss'])
        baseline = next((r for r in geom_results if r['mode'] == 'baseline'), None)

        if baseline and best['mode'] != 'baseline':
            improv = (baseline['test_loss'] - best['test_loss']) / baseline['test_loss'] * 100
            print(f"  {geom}: {best['mode']} (test={best['test_loss']:.6f}, {improv:+.1f}% vs baseline)")
        else:
            print(f"  {geom}: {best['mode']} (test={best['test_loss']:.6f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(GEOMETRY_NAMES))
    width = 0.2
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    ax = axes[0]
    for i, mode in enumerate(MODES):
        losses = [next(r['isomap_train_loss'] for r in all_results
                       if r['geometry'] == g and r['mode'] == mode)
                  for g in GEOMETRY_NAMES]
        offset = (i - len(MODES) / 2 + 0.5) * width
        bars = ax.bar(x + offset, losses, width, label=mode, color=colors[i])
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_ylabel('Isomap Train Loss')
    ax.set_title('Isomap Train Loss (during optimization)')
    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRY_NAMES)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    for i, mode in enumerate(MODES):
        losses = [next(r['test_loss'] for r in all_results
                       if r['geometry'] == g and r['mode'] == mode)
                  for g in GEOMETRY_NAMES]
        offset = (i - len(MODES) / 2 + 0.5) * width
        bars = ax.bar(x + offset, losses, width, label=mode, color=colors[i])
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_ylabel('Test MSE')
    ax.set_title('TEST MSE (generalization)')
    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRY_NAMES)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved: {output_dir}/comparison.png")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print("EXPERIMENT: GradientIsomap Target-Aware Modifications")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"experiments_target_aware_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    all_results = []

    for geometry_name in GEOMETRY_NAMES:
        for mode in MODES:
            working_folder = os.path.join(output_dir, f'{geometry_name}_{mode}')
            os.makedirs(working_folder, exist_ok=True)

            try:
                result = run_single_experiment(geometry_name, mode, working_folder)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {geometry_name}/{mode}: {e}")
                import traceback
                traceback.print_exc()

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        create_summary(all_results, output_dir)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETED")
    print(f"Results: {output_dir}")
    print(f"{'=' * 80}")

    return all_results


if __name__ == "__main__":
    results = main()