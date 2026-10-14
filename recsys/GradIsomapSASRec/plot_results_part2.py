import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# ==========================================
# НАСТРОЙКИ
# ==========================================
GEOMETRY_FOLDER = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/logs_sasrec_isomap/"  # <-- Ваш путь
SIZES = ["300x800", "1000x1200", "3000x2500", "6000x3000"]
SEEDS = [0, 1, 2]
EPOCH_INTERVAL = 25  # Шаг для ridgeline и стрелок траектории

# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================

import ast

def parse_orc_curvatures(x):
    """
    Из строки вида '[(u, v, k), (u, v, k), ...]' возвращает список k (float).
    """
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        seq = x
    elif isinstance(x, str):
        s = x.strip()
        if s == '' or s.lower() == 'nan':
            return []
        try:
            seq = ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            print(f"[warn] cannot parse orc_edges: {s[:80]!r} ({e})")
            return []
    else:
        return []

    out = []
    for item in seq:
        # item = (u, v, k) либо просто k
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append(float(item[2]))
        else:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
    return out


def load_geometry_data(folder, sizes, seeds):
    """Загружает все CSV и парсит orc_edges из JSON."""
    all_data = []
    for size in sizes:
        for seed in seeds:
            filename = f"geometry_results_{size}_seed{seed}.csv"
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                print(f"⚠️  Файл не найден: {filepath}")
                continue
            
            df = pd.read_csv(filepath)
            df['size'] = size
            df['seed'] = seed
            
            # Парсим orc_edges из JSON-строки в список float
            #if 'orc_edges' in df.columns:
            #    df['orc_edges_list'] = df['orc_edges'].apply(
            #        lambda x: json.loads(x) if isinstance(x, str) else x
            #    )
            if 'orc_edges' in df.columns:
                df['orc_edges_list'] = df['orc_edges'].apply(parse_orc_curvatures)
            
            all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

df_all = load_geometry_data(GEOMETRY_FOLDER, SIZES, SEEDS)
print(f"✅ Загружено {len(df_all)} строк")

# ==========================================
# ГРАФИК 1: RIDGELINE РАСПРЕДЕЛЕНИЯ ORC ПО ЭПОХАМ
# ==========================================
def plot_orc_ridgeline(df, size, seed, epoch_interval=25):
    """
    Ridgeline plot распределения ORC по рёбрам для выбранных эпох.
    """
    # Фильтруем данные
    subset = df[(df['size'] == size) & (df['seed'] == seed)].copy()
    
    # Выбираем эпохи с заданным интервалом
    epochs = sorted(subset['epoch'].unique())
    selected_epochs = [e for e in epochs if e % epoch_interval == 0]
    if epochs[-1] not in selected_epochs:
        selected_epochs.append(epochs[-1])  # Всегда включаем последнюю эпоху
    
    # Создаём длинный DataFrame для ridgeline
    rows = []
    for epoch in selected_epochs:
        row_data = subset[subset['epoch'] == epoch]
        if row_data.empty:
            continue
        orc_values = row_data['orc_edges_list'].values[0]
        for val in orc_values:
            rows.append({'epoch': epoch, 'orc_value': val})
    
    plot_df = pd.DataFrame(rows)
    
    # Построение ridgeline
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Используем joypy-стиль через seaborn
    unique_epochs = sorted(plot_df['epoch'].unique())
    n_epochs = len(unique_epochs)
    
    # Цветовая схема (от синего к красному по эпохам)
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (n_epochs - 1)) for i in range(n_epochs)]
    
    # Рисуем KDE для каждой эпохи
    max_density = 0
    kde_data = {}
    for epoch in unique_epochs:
        values = plot_df[plot_df['epoch'] == epoch]['orc_value']
        kde = sns.kdeplot(values, ax=ax, color='gray', alpha=0, 
                          bw_adjust=0.8, warn_singular=False)
        # Получаем данные KDE
        line = kde.lines[-1]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        kde_data[epoch] = (x_data, y_data)
        max_density = max(max_density, y_data.max())
        plt.close(kde.figure)  # Закрываем временный график
    
    # Рисуем заполненные ridgelines
    for i, epoch in enumerate(unique_epochs):
        x_data, y_data = kde_data[epoch]
        # Нормализуем высоту и смещаем по Y
        y_offset = i * 1.2
        y_scaled = (y_data / max_density) * 0.9
        
        ax.fill_between(x_data, y_offset, y_offset + y_scaled, 
                        color=colors[i], alpha=0.7, zorder=2)
        ax.plot(x_data, y_offset + y_scaled, color=colors[i], 
                linewidth=1.5, zorder=3)
        
        # Подпись эпохи слева
        ax.text(x_data.min() - 0.02, y_offset + 0.45, 
                f'Epoch {epoch}', va='center', ha='right', 
                fontsize=10, fontweight='bold', color=colors[i])
    
    # Настройка осей
    ax.set_xlim(x_data.min() - 0.05, x_data.max() + 0.05)
    ax.set_ylim(-0.5, n_epochs * 1.2)
    ax.set_xlabel('Ollivier-Ricci Curvature (per edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title(f'Ridgeline: Distribution of ORC across edges\n{size}, seed={seed}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Добавляем вертикальную линию на 0 (граница гиперболичности)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    ax.text(0, n_epochs * 1.2 + 0.1, 'ORC=0\n(tree-like)', 
            ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    out_name = f"ridgeline_orc_{size}_seed{seed}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_name.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()
    print(f"✅ Сохранено: {out_name}")

# Построить для всех выборок (пример для первой)
for size in SIZES[:1]:  # Уберите [:1] чтобы построить для всех
    for seed in SEEDS[:1]:  # Уберите [:1] чтобы построить для всех сидов
        plot_orc_ridgeline(df_all, size, seed, EPOCH_INTERVAL)

# ==========================================
# ГРАФИК 2: 2D SCATTER "H1_COUNT vs H1_MAX_PERSISTENCE" С ТРАЕКТОРИЕЙ
# ==========================================
def plot_h1_trajectory(df, size, seed, geo_or_lat='lat', epoch_interval=25):
    """
    2D scatter траектории топологии: H1_count vs H1_max_persistence.
    Цвет = эпоха, стрелки показывают направление эволюции.
    """
    subset = df[(df['size'] == size) & (df['seed'] == seed)].copy()
    subset = subset.sort_values('epoch')
    
    # Выбираем колонки в зависимости от geo/lat
    if geo_or_lat == 'geo':
        count_col = 'geo_h1_count'
        pers_col = 'geo_h1_max_persistence'
        title_prefix = 'Original Graph (geo)'
    else:
        count_col = 'lat_h1_count'
        pers_col = 'lat_h1_max_persistence'
        title_prefix = 'Latent Space (lat)'
    
    epochs = subset['epoch'].values
    counts = subset[count_col].values
    persistences = subset[pers_col].values
    
    # Фильтруем эпохи для стрелок
    arrow_epochs_mask = [e % epoch_interval == 0 for e in epochs]
    # Всегда включаем первую и последнюю
    arrow_epochs_mask[0] = True
    arrow_epochs_mask[-1] = True
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Цветовая схема по эпохам
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=epochs.min(), vmax=epochs.max())
    colors = cmap(norm(epochs))
    
    # Рисуем точки всех эпох
    scatter = ax.scatter(counts, persistences, c=epochs, cmap=cmap, 
                         s=80, edgecolors='black', linewidth=1.5, 
                         zorder=3, alpha=0.9)
    
    # Рисуем стрелки между выбранными эпохами
    for i in range(len(epochs) - 1):
        if arrow_epochs_mask[i]:
            x1, y1 = counts[i], persistences[i]
            x2, y2 = counts[i+1], persistences[i+1]
            
            # Вычисляем направление стрелки
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0.01:  # Пропускаем слишком короткие
                # Нормализуем и укорачиваем стрелку, чтобы не перекрывала точки
                scale = 0.85
                ax.annotate('', 
                           xy=(x1 + dx*scale, y1 + dy*scale),
                           xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', 
                                          color=colors[i], 
                                          lw=2, alpha=0.7),
                           zorder=2)
    
    # Аннотируем ключевые эпохи
    for i in range(len(epochs)):
        if arrow_epochs_mask[i]:
            ax.annotate(f'{epochs[i]}', 
                       xy=(counts[i], persistences[i]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color=colors[i],
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Цветовая шкала
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Epoch', fontsize=11, fontweight='bold')
    
    # Настройка осей
    ax.set_xlabel('H1 Count (number of cycles/holes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('H1 Max Persistence (lifetime of longest cycle)', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix}: Topology Trajectory\n{size}, seed={seed}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Добавляем стрелку направления времени
    ax.annotate('Time →', xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=11, fontweight='bold', color='darkred',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    out_name = f"C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/plots/h1_trajectory_{geo_or_lat}_{size}_seed{seed}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_name.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()
    print(f"✅ Сохранено: {out_name}")

# Построить для всех выборок и geo/lat
for size in SIZES:
    for seed in SEEDS[:1]:  # Уберите [:1] для всех сидов
        for geo_or_lat in ['geo', 'lat']:
            plot_h1_trajectory(df_all, size, seed, geo_or_lat, EPOCH_INTERVAL)

# ==========================================
# БОНУС: СРАВНЕНИЕ GEO vs LAT НА ОДНОМ ГРАФИКЕ
# ==========================================
def plot_geo_vs_lat_comparison(df, size, seed, epoch_interval=25):
    """
    Сравнивает траектории geo и lat на одном графике.
    """
    subset = df[(df['size'] == size) & (df['seed'] == seed)].copy()
    subset = subset.sort_values('epoch')
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=subset['epoch'].min(), vmax=subset['epoch'].max())
    
    for geo_or_lat, color_base, marker in [('geo', '#1f77b4', 'o'), ('lat', '#ff7f0e', 's')]:
        if geo_or_lat == 'geo':
            count_col = 'geo_h1_count'
            pers_col = 'geo_h1_max_persistence'
            label = 'Original Graph (geo)'
        else:
            count_col = 'lat_h1_count'
            pers_col = 'lat_h1_max_persistence'
            label = 'Latent Space (lat)'
        
        epochs = subset['epoch'].values
        counts = subset[count_col].values
        persistences = subset[pers_col].values
        
        # Линия траектории
        ax.plot(counts, persistences, color=color_base, linewidth=2, 
                alpha=0.6, zorder=1, label=label)
        
        # Точки
        ax.scatter(counts, persistences, c=[cmap(norm(e)) for e in epochs],
                   s=100, marker=marker, edgecolors='black', linewidth=1.5,
                   zorder=3, alpha=0.9)
        
        # Стрелки
        for i in range(len(epochs) - 1):
            if epochs[i] % epoch_interval == 0 or i == 0:
                ax.annotate('', 
                           xy=(counts[i+1], persistences[i+1]),
                           xytext=(counts[i], persistences[i]),
                           arrowprops=dict(arrowstyle='->', 
                                          color=color_base, 
                                          lw=1.5, alpha=0.5),
                           zorder=2)
    
    # Цветовая шкала
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Epoch', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('H1 Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('H1 Max Persistence', fontsize=12, fontweight='bold')
    ax.set_title(f'Geo vs Lat Topology Trajectory\n{size}, seed={seed}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    out_name = f"C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/plots/h1_geo_vs_lat_{size}_seed{seed}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Сохранено: {out_name}")

# Пример использования
for size in SIZES[:1]:
    for seed in SEEDS[:1]:
        plot_geo_vs_lat_comparison(df_all, size, seed, EPOCH_INTERVAL)