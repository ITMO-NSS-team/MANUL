import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def analyze_isomap_matrix(file_path: str):
    print(f"{'='*60}")
    print(f" Анализ матрицы: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")

    # 1. Загрузка данных
    try:
        data = np.load(file_path)
        D_input = data['D_input']
        D_geodesic = data['D_geodesic']
        knn_adj = data['knn_adj']
        Z = data['Z']
        D_latent = data['D_latent']
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        return

    n_items = D_input.shape[0]
    latent_dim = Z.shape[1]
    print(f"Количество объектов (items): {n_items}")
    print(f"Размерность эмбеддингов (Z): {latent_dim}\n")

    # Вспомогательная функция для статистики
    def print_stats(name, mat):
        is_inf = np.isinf(mat).sum()
        is_nan = np.isnan(mat).sum()
        zeros = (mat == 0).sum() - n_items # вычитаем диагональ
        
        valid_mat = mat[~np.isinf(mat) & ~np.isnan(mat)]
        
        print(f"--- {name} ---")
        print(f"  Min: {valid_mat.min():.6f} | Max: {valid_mat.max():.6f}")
        print(f"  Mean: {valid_mat.mean():.6f} | Std: {valid_mat.std():.6f}")
        if is_inf > 0: print(f"  [!] ВНИМАНИЕ: Найдено {is_inf} значений INF!")
        if is_nan > 0: print(f"  [!] ВНИМАНИЕ: Найдено {is_nan} значений NaN!")
        if zeros > 0:  print(f"  [!] Нулей (вне диагонали): {zeros} ({zeros/(n_items*(n_items-1))*100:.2f}%)")
        print()

    # 2. Базовая статистика
    print_stats("1. D_input (Входные расстояния от градиентов)", D_input)
    print_stats("2. D_geodesic (Геодезические расстояния)", D_geodesic)
    print_stats("3. D_latent (Расстояния в латентном пространстве Z)", D_latent)

    # 3. Анализ связности графа (КРИТИЧЕСКИ ВАЖНО)
    print(f"--- Анализ KNN Графа ---")
    adj_sparse = sp.csr_matrix(knn_adj)
    n_components, labels = csgraph.connected_components(csgraph=adj_sparse, directed=False)
    
    print(f"  Количество компонент связности: {n_components}")
    if n_components == 1:
        print("  [OK] Граф полностью связен. Проблем с топологией нет.")
    else:
        print(f"  [!] КРИТИЧЕСКАЯ ПРОБЛЕМА: Граф разорван на {n_components} частей!")
        print("      Геодезические расстояния между частями не определены (inf).")
        print("      Это частая причина 'зависания' матрицы (градиенты не проходят).")
        
        # Считаем размер компонент
        unique, counts = np.unique(labels, return_counts=True)
        counts = sorted(counts, reverse=True)
        print(f"      Размеры 5 крупнейших компонент: {counts[:5]}")
    print()

    # 4. Анализ эмбеддингов Z (Проверка на Mode Collapse)
    print(f"--- Анализ Эмбеддингов (Z) ---")
    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0)
    print(f"  Среднее значение по координатам: {z_mean.mean():.6f}")
    print(f"  Среднее СКО (разброс) по координатам: {z_std.mean():.6f}")
    
    if z_std.mean() < 1e-4:
        print("  [!] КРИТИЧЕСКАЯ ПРОБЛЕМА: Коллапс латентного пространства (Mode Collapse)!")
        print("      Все объекты слиплись в одну точку. Isomap умер.")
    else:
        print("  [OK] Объекты распределены в пространстве.")
    
    # Норма эмбеддингов
    norms = np.linalg.norm(Z, axis=1)
    print(f"  Норма векторов (L2): Min={norms.min():.4f}, Max={norms.max():.4f}, Mean={norms.mean():.4f}\n")

    # 5. Визуализация (сохранение картинок)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Гистограммы расстояний
    axes[0].hist(D_input.flatten(), bins=50, alpha=0.5, label='D_input', density=True)
    axes[0].set_title("Распределение входных расстояний")
    
    valid_geodesic = D_geodesic[~np.isinf(D_geodesic)]
    axes[1].hist(valid_geodesic.flatten(), bins=50, color='orange', density=True)
    axes[1].set_title("Распределение D_geodesic")

    # PCA проекция Z
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)
    axes[2].scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.3, s=10)
    axes[2].set_title(f"PCA проекция Z (2D)\nExpl. variance: {pca.explained_variance_ratio_.sum():.2f}")

    #plt.tight_layout()
    #plt.show()

# Запуск анализа (подставьте свой путь)
if __name__ == "__main__":
    analyze_isomap_matrix("logs_tecd_marketplace_isomap_cf/810/matrices_epoch7.npz")