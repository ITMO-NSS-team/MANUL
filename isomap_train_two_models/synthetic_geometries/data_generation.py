import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def circles_2d(n_samples=1000):
    xs = np.random.uniform(low=-1, high=1, size=n_samples)
    ys = np.random.uniform(low=-1, high=1, size=n_samples)
    points = np.vstack((xs, ys)).T
    colors = np.array([(abs(point[0]) + abs(point[1])) / 2 for point in points])
    return points, colors

def plot_data(points, colors):
    pca_points = PCA(n_components=2).fit_transform(points)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.scatter(points[:, 1], points[:, 0], points[:, 2], c=colors)
    cb = ax2.scatter(pca_points[:, 1], pca_points[:, 0], c=colors)
    plt.colorbar(cb)
    plt.tight_layout()
    plt.show()


def helicoid(n_samples=1000):
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
    colors = u_

    colors = (colors - colors.min()) / (colors.max() - colors.min())
    plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]


def torus(n_samples=1000):
    r = 1
    R = 3

    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(0.1, 2*np.pi, s)
    v = np.linspace(0, 2*np.pi, s)

    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = (R+r*np.cos(points[:, 0])) * np.cos(points[:, 1])
    y = (R + r * np.cos(points[:, 0])) * np.sin(points[:, 1])
    z = r * np.sin(points[:, 0])

    points_euq = np.vstack([x, y, z]).T
    colors = u_

    colors = (colors - colors.min()) / (colors.max() - colors.min())
    '''colors[colors>0.5] = 1
    colors[colors < 0.5] = 0'''
    plot_data(points_euq, colors)
    return points_euq[:n_samples], colors[:n_samples]


def sphere(n_samples=3000):
    r = 3

    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-1, 1, s)
    v = np.linspace(0, 2 * np.pi, s)

    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = r*(1-points[:, 1]**2)**0.5 * np.cos(points[:, 0])
    y = r * (1 - points[:, 1] ** 2) ** 0.5 * np.sin(points[:, 0])
    z = r*points[:, 1]

    points_euq = np.vstack([x, y, z]).T
    colors = v_

    colors = (colors - colors.min()) / (colors.max() - colors.min())
    plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]


def pseudosphere(n_samples=3000):
    r = 2

    s = math.ceil(n_samples ** 0.5)
    u = np.linspace(-5, 5, s)
    v = np.linspace(0, 2 * np.pi, s)

    u_, v_ = np.meshgrid(u, v)
    u_ = np.ravel(u_)
    v_ = np.ravel(v_)
    points = np.vstack([v_, u_]).T

    x = r*(1/np.cosh(points[:, 1]))*np.cos(points[:, 0])
    y = r * (1 / np.cosh(points[:, 1])) * np.sin(points[:, 0])
    z = r*points[:, 1] - r*np.tanh(points[:, 1])

    points_euq = np.vstack([x, y, z]).T
    colors = v_
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]

def hyperboloid_of_one_sheet(n_samples=3000):
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

    x = a*np.cosh(points[:, 1]) * np.cos(points[:, 0])
    y = b * np.cosh(points[:, 1]) * np.sin(points[:, 0])
    z = c*np.sinh(points[:, 1])

    points_euq = np.vstack([x, y, z]).T
    colors = v_
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    plot_data(points_euq[:n_samples], colors[:n_samples])
    return points_euq[:n_samples], colors[:n_samples]


geometries = {'torus': torus,
              'sphere': sphere,
              'pseudosphere': pseudosphere,
              'hyperboloid_of_one_sheet': hyperboloid_of_one_sheet,
              'helicoid': helicoid}

'''for g in geometries.keys():
    geometries[g]()'''