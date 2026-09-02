import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def compute_movie_stats(train_events, num_items):

    sum_r = np.zeros(num_items, dtype=np.float32)
    cnt_r = np.zeros(num_items, dtype=np.int32)

    for (u, m, r) in train_events:
        m = int(m)
        sum_r[m] += float(r)
        cnt_r[m] += 1

    mean_rating = np.zeros(num_items, dtype=np.float32)
    mask = cnt_r > 0
    mean_rating[mask] = sum_r[mask] / cnt_r[mask]

    popularity = cnt_r.astype(np.float32)
    return mean_rating, popularity


def plot_gi_convergence(history, save_path, top_k=10):

    epochs = history['epoch']
    train_loss = history['train_loss']
    val_hr = history['val_hr']
    val_ndcg = history['val_ndcg']

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train BCE loss')
    plt.title('GI+NCF: Train loss vs epoch')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    if any(v is not None for v in val_hr):
        plt.plot(epochs, val_hr, marker='o', color='tab:blue')
        plt.xlabel('Epoch')
        plt.ylabel(f'HR@{top_k} (val)')
        plt.title('GI+NCF: HR@K on validation')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No val HR recorded',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.subplot(1, 3, 3)
    if any(v is not None for v in val_ndcg):
        plt.plot(epochs, val_ndcg, marker='o', color='tab:orange')
        plt.xlabel('Epoch')
        plt.ylabel(f'NDCG@{top_k} (val)')
        plt.title('GI+NCF: NDCG@K on validation')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No val NDCG recorded',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_manifold_isomap(isomap_model,
                         train_events,
                         num_items,
                         color_by='rating',
                         title_prefix='Isomap manifold',
                         images_dir='.',
                         run_name='run'):

    os.makedirs(images_dir, exist_ok=True)

    isomap_model.eval()
    with torch.no_grad():
        item_Z = isomap_model().to(torch.float32).cpu().numpy()  # [num_items, latent_dim]

    latent_dim = item_Z.shape[1]

    mean_rating, popularity = compute_movie_stats(train_events, num_items)

    if color_by == 'rating':
        colors = mean_rating
        cbar_label = 'Mean rating (train)'
    elif color_by == 'popularity':
        colors = popularity
        cbar_label = 'Popularity (num train ratings)'
    else:
        raise ValueError("color_by must be 'rating' or 'popularity'")

    if latent_dim == 2:
        Z_plot = item_Z
        fig = plt.figure(figsize=(8, 6))
        sc = plt.scatter(Z_plot[:, 0], Z_plot[:, 1],
                         c=colors, cmap='viridis',
                         s=20, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, label=cbar_label)
        subtitle = f'(latent_dim=2, без PCA)'
        plt.xlabel('Z1')
        plt.ylabel('Z2')

    elif latent_dim >= 3:
        if latent_dim == 3:
            Z3 = item_Z
            subtitle = f'(latent_dim=3, без PCA)'
        else:
            # если размерность > 3, сжимаем до 3D PCA
            pca = PCA(n_components=3)
            Z3 = pca.fit_transform(item_Z)
            explained = pca.explained_variance_ratio_.sum()
            subtitle = f'(PCA from {latent_dim}D to 3D, explained var={explained:.2%})'

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2],
                        c=colors, cmap='viridis',
                        s=15, alpha=0.8, edgecolors='none')
        fig.colorbar(sc, ax=ax, label=cbar_label)
        ax.set_xlabel('Z1')
        ax.set_ylabel('Z2')
        ax.set_zlabel('Z3')
    else:
        raise ValueError(f"Ожидалось latent_dim >= 2, получено {latent_dim}")

    plt.title(f"{title_prefix} colored by {color_by}\n{subtitle}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    dim_tag = f"{latent_dim}d" if latent_dim <= 3 else "3d_pca"
    out_path = os.path.join(images_dir,
                            f"{run_name}_isomap_{color_by}_{dim_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Визуализация Isomap manifold ({color_by}) сохранена в {out_path}")


def plot_movie_pca(movie_user_mat,
                   train_events,
                   color_by='rating',
                   title_prefix='PCA of movie-user matrix'):

    num_items = movie_user_mat.shape[0]

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(movie_user_mat)
    explained = pca.explained_variance_ratio_.sum()

    mean_rating, popularity = compute_movie_stats(train_events, num_items)

    if color_by == 'rating':
        colors = mean_rating
        cbar_label = 'Mean rating (train)'
    elif color_by == 'popularity':
        colors = popularity
        cbar_label = 'Popularity (num train ratings)'
    else:
        raise ValueError("color_by must be 'rating' or 'popularity'")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1],
                     c=colors, cmap='viridis',
                     s=20, alpha=0.7, edgecolors='none')
    plt.colorbar(sc, label=cbar_label)
    plt.title(f"{title_prefix} colored by {color_by}\n(PCA, explained var={explained:.2%})")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cf_inner_losses(cf_history,
                         images_dir: str,
                         run_name: str = "run",
                         loss_name: str = "BCE"):

    os.makedirs(images_dir, exist_ok=True)

    train_losses = cf_history.get('train_loss', [])
    val_losses = cf_history.get('val_loss', [])

    num_outer = len(train_losses)
    if num_outer == 0:
        print("cf_history is empty, nothing to plot.")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for outer_idx, inner_losses in enumerate(train_losses):
        if inner_losses is None or len(inner_losses) == 0:
            continue
        x = list(range(len(inner_losses)))
        plt.plot(x, inner_losses, marker='o', label=f'outer {outer_idx}')

    plt.xlabel('Inner epoch (cf_ep)')
    plt.ylabel(f'{loss_name} train loss')
    plt.title('NeuMF train loss по inner-эпохам\nдля разных outer-эпох')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    has_val = False
    for outer_idx, inner_losses in enumerate(val_losses):
        if inner_losses is None or len(inner_losses) == 0:
            continue

        y = [v for v in inner_losses if v is not None]
        x = list(range(len(y)))
        if len(y) == 0:
            continue
        has_val = True
        plt.plot(x, y, marker='s', label=f'outer {outer_idx}')

    if has_val:
        plt.xlabel('Inner epoch (cf_ep)')
        plt.ylabel(f'{loss_name} val loss')
        plt.title('NeuMF val loss по inner-эпохам\nдля разных outer-эпох')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет val-loss (val_loader=None)',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')

    plt.tight_layout()
    out_path = os.path.join(images_dir, f"{run_name}_cf_inner_losses.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"График inner-лоссов NeuMF сохранён в {out_path}")


def plot_pure_ncf_metrics(history, save_dir, run_name="pure_ncf_run621"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['epoch'], history['hr_val'], label='HR@10', color='blue', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('HR@10')
    ax1.set_title('Hit Ratio (HR)')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history['epoch'], history['ndcg_val'], label='NDCG@10', color='green', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('NDCG')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{run_name}_metrics.png'), dpi=150)
    plt.close()


def plot_gi_losses(history, cf_history, images_dir, run_name="run"):

    os.makedirs(images_dir, exist_ok=True)

    epochs = history.get('epoch', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label='Isomap train BCE loss')
    if any(v is not None for v in val_loss):
        plt.plot(epochs, [v if v is not None else float('nan') for v in val_loss],
                 marker='s', label='Isomap val BCE loss')
    plt.xlabel('Outer epoch')
    plt.ylabel('BCE loss')
    plt.title('Isomap (outer) train/val loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path_outer = os.path.join(images_dir, f"{run_name}_outer_isomap_loss.png")
    plt.savefig(out_path_outer, dpi=150)
    plt.close()
    # print(f"Isomap outer-loss график сохранён в {out_path_outer}")

    cf_train = cf_history.get('train_loss', [])
    cf_val   = cf_history.get('val_loss', [])
    num_outer = len(cf_train)

    for outer_idx in range(num_outer):
        inner_train = cf_train[outer_idx] if outer_idx < len(cf_train) else None
        inner_val   = cf_val[outer_idx]   if outer_idx < len(cf_val)   else None

        if inner_train is None or len(inner_train) == 0:
            continue

        x = list(range(len(inner_train)))

        plt.figure(figsize=(8, 5))
        plt.plot(x, inner_train, marker='o', label='NeuMF train BCE loss')

        if inner_val is not None and any(v is not None for v in inner_val):
            y_val = [v if v is not None else float('nan') for v in inner_val]
            plt.plot(x, y_val, marker='s', label='NeuMF val BCE loss')

        plt.xlabel('Inner epoch (cf_ep)')
        plt.ylabel('BCE loss')
        plt.title(f'NeuMF inner train/val loss (outer epoch {outer_idx+1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path_inner = os.path.join(
            images_dir,
            f"{run_name}_inner_neumf_losses_outer{outer_idx}.png"
        )
        plt.savefig(out_path_inner, dpi=150)
        plt.close()
        # print(f"NeuMF inner-loss график для outer={outer_idx} сохранён в {out_path_inner}")


def plot_pure_ncf_losses(history, images_dir, run_name="pure_ncf"):
    """
    history: словарь с ключами 'epoch', 'train_loss', 'val_loss'
    images_dir: папка для сохранения PNG
    run_name: префикс имени файла
    """
    os.makedirs(images_dir, exist_ok=True)

    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label='Train BCE loss')
    plt.plot(epochs, val_loss, marker='s', label='Val BCE loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE loss')
    plt.title('Pure NeuMF: train/val loss vs epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(images_dir, f"{run_name}_loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"График лоссов NCF сохранён в {out_path}")


def plot_isomap_deltas(dist_history, images_dir, run_name="ginmf_isomap"):
    """
    dist_history: словарь с ключами 'epoch', 'delta_prev', 'delta_init'
    images_dir: папка для сохранения PNG
    run_name: префикс имени файлов
    """
    os.makedirs(images_dir, exist_ok=True)

    epochs = dist_history.get('epoch', [])
    delta_prev = dist_history.get('delta_prev', [])
    delta_init = dist_history.get('delta_init', [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, delta_prev, marker='o')
    ax1.set_xlabel('Outer epoch')
    ax1.set_ylabel('Δ_prev (||D_t - D_{t-1}|| / ||D_{t-1}||)')
    ax1.set_title('Isomap: относительное изменение (к предыдущей эпохе)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, delta_init, marker='s', color='tab:orange')
    ax2.set_xlabel('Outer epoch')
    ax2.set_ylabel('Δ_init (||D_t - D_0|| / ||D_0||)')
    ax2.set_title('Isomap: относительное изменение (к начальному состоянию)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_combined = os.path.join(images_dir, f"{run_name}_delta_combined.png")
    plt.savefig(out_combined, dpi=150)
    plt.close()
