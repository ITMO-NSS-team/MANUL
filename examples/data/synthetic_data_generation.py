import math
from sklearn.datasets import make_swiss_roll, make_s_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(42)


def plot_data(points, colors, title='', save_path=None):
    pca_points = PCA(n_components=2).fit_transform(points)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.scatter(points[:, 1], points[:, 0], points[:, 2], c=colors)
    cb = ax2.scatter(pca_points[:, 1], pca_points[:, 0], c=colors)
    plt.colorbar(cb)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/{title}.png')
        plt.close()
    else:
        plt.show()


def helicoid(n_samples=1000, normalize=True):
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(0, 10, s)
    v = np.linspace(-1.5 * np.pi, 1.5 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T
    x = points[:, 1] * np.cos(points[:, 0])
    y = points[:, 1] * np.sin(points[:, 0])
    z = points[:, 0]
    points_euq = np.vstack([x, y, z]).T
    if normalize:
        points_euq = normalize_points(points_euq)
    colors = u_
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    return points_euq[:n_samples], colors[:n_samples]


def torus(n_samples=1000, normalize=True):
    r = 1
    R = 3

    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(0.1, 2 * np.pi, s)
    v = np.linspace(0, 2 * np.pi, s)

    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = (R + r * np.cos(points[:, 0])) * np.cos(points[:, 1])
    y = (R + r * np.cos(points[:, 0])) * np.sin(points[:, 1])
    z = r * np.sin(points[:, 0])

    points_euq = np.vstack([x, y, z]).T
    if normalize:
        points_euq = normalize_points(points_euq)
    colors = u_

    colors = (colors - colors.min()) / (colors.max() - colors.min())
    return points_euq[:n_samples], colors[:n_samples]


def sphere(n_samples=1000, normalize=True):
    r = 3
    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-1, 1, s)
    v = np.linspace(0, 2 * np.pi, s)
    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = r * (1 - points[:, 1] ** 2) ** 0.5 * np.cos(points[:, 0])
    y = r * (1 - points[:, 1] ** 2) ** 0.5 * np.sin(points[:, 0])
    z = r * points[:, 1]

    points_euq = np.vstack([x, y, z]).T
    if normalize:
        points_euq = normalize_points(points_euq)
    colors = v_

    colors = (colors - colors.min()) / (colors.max() - colors.min())
    #plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]


def pseudosphere(n_samples=1000, normalize=True):
    r = 2

    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-5, 5, s)
    v = np.linspace(0, 2 * np.pi, s)

    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = r * (1 / np.cosh(points[:, 1])) * np.cos(points[:, 0])
    y = r * (1 / np.cosh(points[:, 1])) * np.sin(points[:, 0])
    z = r * points[:, 1] - r * np.tanh(points[:, 1])

    points_euq = np.vstack([x, y, z]).T
    if normalize:
        points_euq = normalize_points(points_euq)
    colors = v_
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    #plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]


def hyperboloid_of_one_sheet(n_samples=1000, normalize=True):
    a = 2
    b = 2
    c = 2

    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-5, 5, s)
    v = np.linspace(0, 2 * np.pi, s)

    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = a * np.cosh(points[:, 1]) * np.cos(points[:, 0])
    y = b * np.cosh(points[:, 1]) * np.sin(points[:, 0])
    z = c * np.sinh(points[:, 1])

    points_euq = np.vstack([x, y, z]).T
    if normalize:
        points_euq = normalize_points(points_euq)
    colors = v_

    colors = (colors - colors.min()) / (colors.max() - colors.min())
    #plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]


def swiss_roll(n_samples=1000, normalize=True):
    X, colors = make_swiss_roll(n_samples=n_samples, noise=0, random_state=42)
    if normalize:
        X = normalize_points(X)
    colors = colors / max(colors)
    return X, colors


def swiss_hole(n_samples=1000, normalize=True):
    X, colors = make_swiss_roll(n_samples=n_samples, noise=0, hole=True, random_state=42)
    if normalize:
        X = normalize_points(X)
    colors = colors / max(colors)
    return X, colors


def s_curve(n_samples=1000, normalize=True):
    X, colors = make_s_curve(n_samples, noise=0, random_state=0)
    if normalize:
        X = normalize_points(X)
    colors = colors / max(colors)
    return X, colors


def multi_scale_torus(n_samples=1000, normalize=True):
    """Torus with varying local curvature."""
    theta = np.linspace(0, 2 * np.pi, n_samples)
    phi = np.linspace(0, 2 * np.pi, n_samples)
    R, r = 3, 1
    x = (R + r * np.cos(theta)) * np.cos(phi) + 0.3 * np.cos(8 * theta)
    y = (R + r * np.cos(theta)) * np.sin(phi) + 0.3 * np.sin(8 * theta)
    z = r * np.sin(theta) + 0.3 * np.cos(8 * phi)
    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points)
    colors = theta
    colors = colors / colors.max()
    return points, colors


def nonuniform_sphere(n_samples=1000, normalize=True):
    """Sphere with non-uniform sampling density."""
    np.random.seed(42)
    u = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 2 * np.pi, n_samples)
    z = np.sign(u - 0.5) * (np.abs(u - 0.5) * 2) ** 0.3
    r = np.sqrt(1 - z ** 2)
    x = r * np.cos(v)
    y = r * np.sin(v)
    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points)
    colors = v
    colors = colors / colors.max()
    return points, colors


def cone_surface(n_samples=1000, normalize=True):
    """Cone surface with singularity at apex."""
    np.random.seed(42)
    r = np.random.uniform(0, 2, n_samples)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = r  # Cone equation: z = r
    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points)
    colors = theta
    colors = colors / colors.max()
    return points, colors


def genus_2_surface(n_samples=1000, normalize=True):
    """Surface with genus 2 (double torus)."""
    u = np.linspace(0, 2 * np.pi, int(np.sqrt(n_samples)))
    v = np.linspace(0, 2 * np.pi, int(np.sqrt(n_samples)))
    u, v = np.meshgrid(u, v)
    u, v = u.ravel(), v.ravel()
    x = np.cos(u) * (2 + np.cos(v))
    y = np.sin(u) * (2 + np.cos(v))
    z = np.sin(v) + 0.5 * np.sin(2 * v) * np.cos(u)
    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points)
    colors = u
    colors = colors / colors.max()
    return points, colors


def connected_multiscale_manifold(n_samples=1000, normalize=True):
    """Single connected manifold with varying properties."""
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = (2 + np.cos(t)) * np.cos(t)
    y = (2 + np.cos(t)) * np.sin(t)
    z = np.sin(2 * t)
    points = np.vstack([x, y, z]).T
    if normalize:
        points = normalize_points(points)
    colors = t
    colors = colors / colors.max()
    return points, colors


def noisy_manifold(base_func, noise_percent=0.1, n_samples=1000):
    """Add topological noise to any base manifold."""
    points, colors = base_func(n_samples)
    noise = np.random.normal(0, np.max(points) * noise_percent, points.shape)
    return points + noise, colors


def normalize_points(points, new_min=0, new_max=20):
    norm_points = new_min + (points - np.min(points)) * (new_max - new_min) / (np.max(points) - np.min(points))
    return norm_points


geometries = {'sphere': [sphere, 2],
              'swiss_roll': [swiss_roll, 2],
              'swiss_hole': [swiss_hole, 2],
              's_curve': [s_curve, 2],
              'torus': [torus, 2],
              'pseudosphere': [pseudosphere, 2],
              'hyperboloid': [hyperboloid_of_one_sheet, 2],
              'helicoid': [helicoid, 2],
              'multi_scale_torus': [multi_scale_torus, 2],
              'nonuniform_sphere': [nonuniform_sphere, 2],
              'cone_surface': [cone_surface, 2],
              'genus_2_surface': [genus_2_surface, 2],
              'connected_multiscale_manifold': [connected_multiscale_manifold, 1]
              }

'''for g in geometries:
    points, colors = geometries[g][0]()
    plot_data(points, colors, f'{g}')'''
