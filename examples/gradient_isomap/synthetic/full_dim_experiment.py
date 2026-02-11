import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import time
import math
from sklearn.datasets import make_swiss_roll, make_s_curve

from Adam.GradientIsomap import GradientIsomap
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.utils import split_data

def normalize_points(points, new_min=0, new_max=1):
    """Normalize points to [new_min, new_max]."""
    return new_min + (points - np.min(points)) * (new_max - new_min) / (np.max(points) - np.min(points))


def combine_coordinates(u, v):
    """Combine u and v using y = u + v ."""
    u_norm = (u - u.min()) / (u.max() - u.min()) if u.max() > u.min() else u
    v_norm = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v
    combined = u_norm + v_norm
    return (combined - combined.min()) / (combined.max() - combined.min())


def sphere_full_dim(n_samples=1000, normalize=True):
    r = 3
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-1, 1, s)
    v = np.linspace(0, 2 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_, v_ = np.ravel(u_), np.ravel(v_)

    x = r * (1 - u_ ** 2) ** 0.5 * np.cos(v_)
    y = r * (1 - u_ ** 2) ** 0.5 * np.sin(v_)
    z = r * u_

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(u_[:n_samples], v_[:n_samples])


def torus_full_dim(n_samples=1000, normalize=True):
    r, R = 1, 3
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(0.1, 2 * np.pi, s)
    v = np.linspace(0, 2 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_, v_ = np.ravel(u_), np.ravel(v_)

    x = (R + r * np.cos(v_)) * np.cos(u_)
    y = (R + r * np.cos(v_)) * np.sin(u_)
    z = r * np.sin(v_)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(u_[:n_samples], v_[:n_samples])


def swiss_roll_full_dim(n_samples=1000, normalize=True):
    X, t = make_swiss_roll(n_samples=n_samples, noise=0, random_state=42)
    if normalize:
        X = normalize_points(X, 0, 1)
    return X, combine_coordinates(t, X[:, 1])


def swiss_hole_full_dim(n_samples=1000, normalize=True):
    X, t = make_swiss_roll(n_samples=n_samples, noise=0, hole=True, random_state=42)
    if normalize:
        X = normalize_points(X, 0, 1)
    return X, combine_coordinates(t, X[:, 1])


def s_curve_full_dim(n_samples=1000, normalize=True):
    X, t = make_s_curve(n_samples, noise=0, random_state=0)
    if normalize:
        X = normalize_points(X, 0, 1)
    return X, combine_coordinates(t, X[:, 1])


def pseudosphere_full_dim(n_samples=1000, normalize=True):
    r = 2
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-5, 5, s)
    v = np.linspace(0, 2 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_, v_ = np.ravel(u_), np.ravel(v_)

    x = r * (1 / np.cosh(u_)) * np.cos(v_)
    y = r * (1 / np.cosh(u_)) * np.sin(v_)
    z = r * u_ - r * np.tanh(u_)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(u_[:n_samples], v_[:n_samples])


def hyperboloid_full_dim(n_samples=1000, normalize=True):
    a, b, c = 2, 2, 2
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-5, 5, s)
    v = np.linspace(0, 2 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_, v_ = np.ravel(u_), np.ravel(v_)

    x = a * np.cosh(u_) * np.cos(v_)
    y = b * np.cosh(u_) * np.sin(v_)
    z = c * np.sinh(u_)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(u_[:n_samples], v_[:n_samples])


def helicoid_full_dim(n_samples=1000, normalize=True):
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(0, 10, s)
    v = np.linspace(-1.5 * np.pi, 1.5 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_, v_ = np.ravel(u_), np.ravel(v_)

    x = u_ * np.cos(v_)
    y = u_ * np.sin(v_)
    z = v_

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(u_[:n_samples], v_[:n_samples])


def multi_scale_torus_full_dim(n_samples=1000, normalize=True):
    s = math.ceil(n_samples ** 0.5)
    theta = np.linspace(0, 2 * np.pi, s)
    phi = np.linspace(0, 2 * np.pi, s)
    theta_, phi_ = np.meshgrid(theta, phi)
    theta_, phi_ = np.ravel(theta_), np.ravel(phi_)

    R, r = 3, 1
    x = (R + r * np.cos(theta_)) * np.cos(phi_) + 0.3 * np.cos(8 * theta_)
    y = (R + r * np.cos(theta_)) * np.sin(phi_) + 0.3 * np.sin(8 * theta_)
    z = r * np.sin(theta_) + 0.3 * np.cos(8 * phi_)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(theta_[:n_samples], phi_[:n_samples])


def nonuniform_sphere_full_dim(n_samples=1000, normalize=True):
    np.random.seed(42)
    u = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 2 * np.pi, n_samples)

    z = np.sign(u - 0.5) * (np.abs(u - 0.5) * 2) ** 0.3
    r = np.sqrt(1 - z ** 2)
    x = r * np.cos(v)
    y = r * np.sin(v)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points, combine_coordinates(u, v)


def cone_surface_full_dim(n_samples=1000, normalize=True):
    np.random.seed(42)
    r = np.random.uniform(0, 2, n_samples)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = r

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points, combine_coordinates(r, theta)


def genus_2_surface_full_dim(n_samples=1000, normalize=True):
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(0, 2 * np.pi, s)
    v = np.linspace(0, 2 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_, v_ = np.ravel(u_), np.ravel(v_)

    x = np.cos(u_) * (2 + np.cos(v_))
    y = np.sin(u_) * (2 + np.cos(v_))
    z = np.sin(v_) + 0.5 * np.sin(2 * v_) * np.cos(u_)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    return points[:n_samples], combine_coordinates(u_[:n_samples], v_[:n_samples])


def connected_multiscale_manifold_full_dim(n_samples=1000, normalize=True):
    """1D manifold — returns standard target."""
    t = np.linspace(0, 4 * np.pi, n_samples)

    x = (2 + np.cos(t)) * np.cos(t)
    y = (2 + np.cos(t)) * np.sin(t)
    z = np.sin(2 * t)

    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points, 0, 1)

    # 1D manifold: target is just normalized t
    target = (t - t.min()) / (t.max() - t.min())
    return points, target


full_dim_geometries = {
    'sphere': sphere_full_dim,
    'torus': torus_full_dim,
    'swiss_roll': swiss_roll_full_dim,
    'swiss_hole': swiss_hole_full_dim,
    's_curve': s_curve_full_dim,
    'pseudosphere': pseudosphere_full_dim,
    'hyperboloid': hyperboloid_full_dim,
    'helicoid': helicoid_full_dim,
    'multi_scale_torus': multi_scale_torus_full_dim,
    'nonuniform_sphere': nonuniform_sphere_full_dim,
    'cone_surface': cone_surface_full_dim,
    'genus_2_surface': genus_2_surface_full_dim,
    'connected_multiscale_manifold': connected_multiscale_manifold_full_dim
}



def synthetic_full_dim_pipeline(geometry_name, working_folder):
    """
    Manifold learning pipeline with full-dimensional target.

    Parameters:
    -----------
    geometry_name : str
        Name of geometry
    working_folder : str
        Path to outputs directory
    """

    n_samples = 5000
    n_base_points = 1000
    latent_dim = 2
    epochs = 5000
    proj_method = 'random_forest'
    device = 'cuda'

    print("=" * 80)
    print("FULL-DIMENSIONAL TARGET PIPELINE")
    print("=" * 80)

    print(f"Working folder: {working_folder}")
    print(f"Generating {n_samples} points for {geometry_name}")
    print(f"Target: y = u² + v² ")

    geometry_function = full_dim_geometries[geometry_name]
    X, y = geometry_function(n_samples=n_samples)

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
    if os.path.exists(f'{working_folder}/fps_indices.npy'):
        fps_indices = np.load(f'{working_folder}/fps_indices.npy')
        fps_extract_time = 0
        print(f'FPS indices loaded from {working_folder}/fps_indices.npy')
    else:
        start_time = time.time()
        fps_indices = memory_efficient_fps(features=X_train, n_samples=n_base_points, batch_size=500)
        np.save(f'{working_folder}/fps_indices.npy', fps_indices)
        fps_extract_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'FPS indices saved to {working_folder}/fps_indices.npy')

    print("\n=== MANIFOLD LEARNING ===")
    train_features = torch.tensor(X_train[fps_indices], dtype=torch.float32).to(device)
    train_target = torch.tensor(y_train[fps_indices], dtype=torch.float32).to(device)

    print(f"Training GradientIsomap (latent_dim={latent_dim}, epochs={epochs})...")
    start_time = time.time()
    isomap = GradientIsomap(
        train_feature=train_features,
        train_target=train_target,
        latent_len=latent_dim,
        checkpoint_each=100,
        save_checkpoint_matrix=False,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=epochs,
        stop_criteria_value=0.001
    )
    isomap.train()
    isomap_train_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    isomap.visualize_trained()

    base_projection = isomap.best_isomap_model().detach().cpu().numpy()
    np.save(f'{working_folder}/base_projection.npy', base_projection)
    print(f"Saved base projections {working_folder}/base_projection.npy")

    print("\n=== COMPUTING PROJECTIONS ===")
    start_time = time.time()
    projector = Projector(
        source_data=X_train,
        base_indices=fps_indices,
        batch_size=1024,
        base_projection=base_projection,
    )
    train_projections = projector.compute_projection(method=proj_method)
    projection_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    np.save(os.path.join(working_folder, 'train_projections.npy'), train_projections)
    print(f"Saved train projections")

    metadata = pd.DataFrame({
        'Parameter': ['Geometry name', 'Total samples', 'Base points (FPS)',
                      'Target type', 'Target formula', 'Latent dimension', 'Device',
                      'FPS time', 'Isomap train time', 'Projection method', 'Projection time'],
        'Value': [geometry_name, n_samples, n_base_points,
                  'full_dimensional', 'u + v', latent_dim, device,
                  fps_extract_time, isomap_train_time, proj_method, projection_time]
    })
    metadata.to_csv(f'{working_folder}/metadata.csv', index=False)

    print("\nConfiguration:")
    for _, row in metadata.iterrows():
        print(f"{row['Parameter']} - {row['Value']}")
    print(f"Geometry {geometry_name} processing complete!")

    return working_folder


if __name__ == "__main__":
    outputs_dir = 'outputs_full_dim_target/'

    geometries_to_test = list(full_dim_geometries.keys())
    n_runs = 5

    print(f"\n{'=' * 80}")
    print("FULL-DIMENSIONAL TARGET EXPERIMENT")
    print(f"{'=' * 80}")
    print(f"Geometries: {len(geometries_to_test)}")
    print(f"Target: y = u + v")
    print(f"Runs per geometry: {n_runs}")
    print(f"{'=' * 80}\n")

    for geom in geometries_to_test:
        for run in range(n_runs):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            working_folder = f'{outputs_dir}/{geom}/{geom}_run{run}_{timestamp}'
            os.makedirs(working_folder, exist_ok=True)

            print(f"\n Geometry: {geom}, Run: {run + 1}/{n_runs}")

            try:
                synthetic_full_dim_pipeline(geom, working_folder)
                print(f"✓ Completed")
            except Exception as e:
                print(f"✗ Error: {e}")
                continue

    print(f"\n{'=' * 80}")
    print("All experiments saved to outputs_full_dim_target folder")
    print(f"{'=' * 80}")
