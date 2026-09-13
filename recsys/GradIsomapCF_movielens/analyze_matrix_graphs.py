import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def analyze_and_plot_folder(folder_path: str, save_path: str = None):
    """Находит все матрицы в папке, сортирует их по эпохам и строит один общий график:

    каждая эпоха — это отдельная строка из 3 диаграмм.
    """
    # 1. Поиск всех файлов matrices_epoch*.npz в папке
    search_pattern = os.path.join(folder_path, "matrices_epoch*.npz")
    files = glob.glob(search_pattern)

    if not files:
        print(f"[-] В папке '{folder_path}' не найдено файлов 'matrices_epoch*.npz'")
        return

    # Вспомогательная функция для извлечения номера эпохи из имени файла
    def get_epoch_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"epoch(\d+)", filename)
        return int(match.group(1)) if match else -1

    # 2. Сортировка файлов по возрастанию номера эпохи
    files.sort(key=get_epoch_number)

    num_files = len(files)
    print(f"[+] Найдено файлов для анализа: {num_files}")
    for f in files:
        print(f"  - {os.path.basename(f)} (Эпоха: {get_epoch_number(f)})")

    # 3. Создание общего полотна для графиков
    # Ширина 18 (как у вас), высота пропорциональна количеству файлов (5 на каждую строку)
    fig, axes = plt.subplots(
        nrows=num_files,
        ncols=3,
        figsize=(18, 5 * num_files),
        squeeze=False,  # Чтобы axes всегда был двумерным массивом [row, col]
    )

    # 4. Цикл по всем файлам и отрисовка
    for idx, file_path in enumerate(files):
        epoch = get_epoch_number(file_path)
        print(f"\nОбработка эпохи {epoch}...")

        try:
            data = np.load(file_path)
            D_input = data["D_input"]
            D_geodesic = data["D_geodesic"]
            Z = data["Z"]
        except Exception as e:
            print(f"Ошибка загрузки файла {file_path}: {e}")
            continue

        # --- Колонка 1: Распределение входных расстояний ---
        axes[idx, 0].hist(
            D_input.flatten(), bins=50, alpha=0.5, density=True, color="blue"
        )
        axes[idx, 0].set_title(f"Эпоха {epoch}: Входные расстояния")
        axes[idx, 0].set_ylabel(
            f"EPOCH {epoch}", fontsize=14, fontweight="bold"
        )  # Выделяем эпоху слева

        # --- Колонка 2: Геодезические расстояния ---
        valid_geodesic = D_geodesic[~np.isinf(D_geodesic) & ~np.isnan(D_geodesic)]
        if len(valid_geodesic) > 0:
            axes[idx, 1].hist(
                valid_geodesic.flatten(), bins=50, color="orange", density=True
            )
        axes[idx, 1].set_title(f"Эпоха {epoch}: Геодезические")

        # --- Колонка 3: PCA проекция Z ---
        pca = PCA(n_components=2)
        try:
            Z_pca = pca.fit_transform(Z)
            axes[idx, 2].scatter(
                Z_pca[:, 0], Z_pca[:, 1], alpha=0.3, s=10, color="green"
            )
            var_ratio = pca.explained_variance_ratio_.sum()
            axes[idx, 2].set_title(
                f"Эпоха {epoch}: PCA Z (Expl. var: {var_ratio:.2f})"
            )
        except Exception as e:
            axes[idx, 2].text(
                0.5,
                0.5,
                f"Ошибка PCA: {e}",
                ha="center",
                va="center",
                color="red",
            )
            axes[idx, 2].set_title(f"Эпоха {epoch}: PCA Ошибка")

    # Корректируем расположение элементов, чтобы заголовки не наезжали друг на друга
    plt.tight_layout()

    # Сохранение, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[+] Общий график успешно сохранен в: {save_path}")

    #
    # plt.show()


# Пример запуска:
if __name__ == "__main__":
    # Укажите путь к вашей папке с матрицами (например, 'logs_tecd_marketplace_isomap_cf/802/')
    # folder_to_analyze = "logs_tecd_marketplace_isomap_cf/905"
    folder_to_analyze = "recsys/GradIsomapCF_movielens/logs_movielens_isomap_cf/run61"

    # Запуск. График также сохранится на диск в этой же папке
    analyze_and_plot_folder(
        folder_path=folder_to_analyze,
        save_path=os.path.join(folder_to_analyze, "images/isomap_evolution.png"),
    )