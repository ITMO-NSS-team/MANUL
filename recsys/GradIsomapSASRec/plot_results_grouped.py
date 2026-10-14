import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
GEOMETRY_FOLDER = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/logs_sasrec_isomap/"  # <-- Ваш путь к папке с CSV
SIZES = ["300x800", "1000x1200", "3000x2500", "6000x3000"]
SEEDS = [0, 1, 2]

# 5 равномерных этапов по 300 эпохам
TARGET_EPOCHS = [0, 60, 120, 180, 240, 290] 

# Список мер, для которых нужно построить графики
METRICS_LIST = [
    "orc_mean", 
    "orc_f_neg",
    "residual_variance",
    "spearman_rho",
    "geo_h0_max_persistence",
    "spectral_spectral_gap", 
    "spectral_lambda_max",
    "spectral_lambda2",
    "lat_h0_max_persistence", 
    "kruskal_stress"
]

# Цветовая палитра для этапов (градиент от светлого к темному показывает прогресс обучения)
# Можно заменить на plt.cm.viridis, plt.cm.plasma и т.д.
EPOCH_COLORS = plt.cm.Blues(np.linspace(0.3, 0.9, len(TARGET_EPOCHS)))

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_geometry_data(folder, sizes, seeds):
    all_data = []
    for size in sizes:
        for seed in seeds:
            filepath = os.path.join(folder, f"geometry_results_{size}_seed{seed}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['size'] = size
                df['seed'] = seed
                all_data.append(df)
            else:
                print(f"⚠️ Файл не найден: {filepath}")
    return pd.concat(all_data, ignore_index=True)

print("Загрузка данных...")
df_all = load_geometry_data(GEOMETRY_FOLDER, SIZES, SEEDS)

# Оставляем только нужные эпохи
df_filtered = df_all[df_all['epoch'].isin(TARGET_EPOCHS)]
print(f"✅ Загружено и отфильтровано {len(df_filtered)} строк.")

# ==========================================
# 3. ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================
for metric in METRICS_LIST:
    if metric not in df_filtered.columns:
        print(f"️ Метрика '{metric}' не найдена в CSV. Пропускаем.")
        continue

    # Агрегация: считаем mean и std по сидам для каждого размера и эпохи
    agg_df = df_filtered.groupby(['size', 'epoch'])[metric].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(SIZES))  # Позиции групп на оси X (0, 1, 2, 3)
    n_bars = len(TARGET_EPOCHS)
    bar_width = 0.15  # Ширина одного столбца
    
    # Рисуем столбцы для каждого этапа
    for i, epoch in enumerate(TARGET_EPOCHS):
        # Берем данные для текущей эпохи
        mask = agg_df['epoch'] == epoch
        means = agg_df.loc[mask, 'mean'].values
        stds = agg_df.loc[mask, 'std'].values
        
        # Вычисляем позицию столбца внутри группы
        # Центрируем группу: сдвигаем влево на половину общей ширины группы
        offset = (n_bars / 2 - i - 0.5) * bar_width
        positions = x + offset
        
        ax.bar(positions, means, bar_width, 
               yerr=stds, 
               label=f'Epoch {epoch}', 
               color=EPOCH_COLORS[i], 
               edgecolor='black', 
               linewidth=0.8, 
               capsize=5, # Шляпки на error bars
               zorder=3)
        
        # Добавляем точные значения над столбцами (опционально, можно закомментировать)
        # for pos, mean in zip(positions, means):
        #     ax.text(pos, mean + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

    # Настройка осей и оформления
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES, fontsize=12, fontweight='bold')
    ax.set_xlabel("Размер выборки (Матрица)", fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=13, fontweight='bold')
    ax.set_title(f"Динамика метрики '{metric}' по этапам обучения", 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Сетка и легенда
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="Этап обучения", loc='upper left', fontsize=11, title_fontsize=12, framealpha=0.9)
    
    # Сохранение
    out_name = f"C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/plots/grouped_sasrec_amazon/bar_chart_{metric}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    #plt.savefig(out_name.replace('.png', '.pdf'), bbox_inches='tight') # Для статьи
    #plt.show()
    plt.close()
    print(f"✅ График для '{metric}' сохранен: {out_name}")