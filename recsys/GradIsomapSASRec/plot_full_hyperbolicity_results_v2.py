import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Загрузка данных
df = pd.read_csv('C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/full_hyperbolicity_table.csv')

# --- Диагностика (можно потом удалить) ---
print("geometry values:", df['geometry'].unique())
print("size values:", df['size'].unique())

# Агрегация по сидам (mean ± std)
df_agg = df.groupby(['size', 'geometry'])[['delta_rel', 'HR@10', 'NDCG@10']].agg(['mean', 'std']).reset_index()
df_agg.columns = ['size', 'geometry', 'delta_rel_mean', 'delta_rel_std', 'HR_mean', 'HR_std', 'NDCG_mean', 'NDCG_std']

# --- Порядок и палитры ---
sizes_order = ['300x800', '1000x1200', '3000x2500', '6000x3000']
geometries  = ['GISASRec', 'Poincare', 'Euclidean']   # ← приведите к реальным именам из CSV

# Цвета для размеров (как на оригинальном графике)
size_colors = {
    '300x800':   '#4daf4a',   # Зелёный
    '1000x1200': '#377eb8',   # Синий
    '3000x2500': '#ff7f00',   # Оранжевый
    '6000x3000': '#e41a1c',   # Красный
}

# Маркеры для геометрий
geometry_markers = {
    'GISASRec':  'o',
    'Poincare':  '^',
    'Euclidean': 's',
}

# --- Фигура ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ==========================================
# ЛЕВЫЙ ГРАФИК: Hyperbolicity vs Quality
# ==========================================
for size in sizes_order:
    size_data = df_agg[df_agg['size'] == size]
    color = size_colors.get(size, 'gray')

    for geom in geometries:
        geom_data = size_data[size_data['geometry'] == geom]
        if geom_data.empty:
            continue

        x     = geom_data['delta_rel_mean'].values[0]
        y     = geom_data['HR_mean'].values[0]
        x_err = geom_data['delta_rel_std'].values[0]
        y_err = geom_data['HR_std'].values[0]

        ax1.errorbar(
            x, y, xerr=x_err, yerr=y_err,
            fmt=geometry_markers.get(geom, 'o'),
            color=color,
            markersize=12, markeredgecolor='black',
            markeredgewidth=1.5, capsize=4,
            zorder=3,   # label ставим вручную ниже
        )

ax1.set_xlabel('delta_rel (Gromov hyperbolicity, normalized — LOWER = more tree-like/hyperbolic)', fontsize=11)
ax1.set_ylabel('test HR@10', fontsize=12, fontweight='bold')
ax1.set_title('Hyperbolicity vs downstream quality, all 3 geometries x all 4 scales\nstill no clean hyperbolicity relationship, at any scale',
              fontsize=12, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# --- Две легенды на ax1: цвет = scale, маркер = geometry ---
scale_handles = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=size_colors[s], markeredgecolor='black',
           markersize=12, linestyle='None', label=s)
    for s in sizes_order
]
geom_handles = [
    Line2D([0], [0], marker=geometry_markers[g], color='gray',
           linestyle='None', markersize=10, label=g)
    for g in geometries
]

leg1 = ax1.legend(handles=scale_handles, title='scale (color)',
                  loc='upper left', framealpha=0.9)
ax1.add_artist(leg1)   # ← чтобы вторая легенда не стёрла первую
ax1.legend(handles=geom_handles, title='geometry (marker)',
           loc='lower right', framealpha=0.9)


# ==========================================
# ПРАВЫЙ ГРАФИК: Quality vs Scale (ЦВЕТНОЙ)
# ==========================================
# Идея: цвет = scale, маркер = geometry.
# Внутри каждой геометрии точки идут по возрастанию scale,
# поэтому удобно рисовать их по одной точке с нужным цветом.

x_positions = {s: i for i, s in enumerate(sizes_order)}

for geom in geometries:
    geom_data = df_agg[df_agg['geometry'] == geom].copy()
    if geom_data.empty:
        print(f"[skip] geometry='{geom}' not found in data")
        continue

    # Сортируем по порядку scale
    geom_data['size_num'] = geom_data['size'].map(x_positions)
    geom_data = geom_data.dropna(subset=['size_num']).sort_values('size_num')

    xs    = geom_data['size_num'].values
    ys    = geom_data['HR_mean'].values
    yerrs = geom_data['HR_std'].values
    cols  = [size_colors.get(s, 'gray') for s in geom_data['size'].values]

    # Пунктирная соединительная линия по этой геометрии
    ax2.plot(xs, ys, linestyle='--', color='gray',
             alpha=0.5, linewidth=1.5, zorder=1)

    # Точки с error bars, каждая своего цвета (по scale)
    for x, y, ye, c in zip(xs, ys, yerrs, cols):
        ax2.errorbar(
            x, y, yerr=ye,
            fmt=geometry_markers[geom],
            color=c,
            markersize=14,
            markeredgecolor='black',
            markeredgewidth=1.5,
            capsize=5,
            zorder=3,
        )

ax2.set_xticks(range(len(sizes_order)))
ax2.set_xticklabels(sizes_order, fontsize=10)
ax2.set_xlabel('scale', fontsize=12)
ax2.set_ylabel('test HR@10', fontsize=12, fontweight='bold')
ax2.set_title("Quality vs scale, all 3 geometries under equal conditions\n"
              "(dashed line = mean per scale; GINCF's edge over Euclidean narrows as scale grows)",
              fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# --- Легенда 1: цвет = scale ---
scale_handles2 = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=size_colors[s], markeredgecolor='black',
           markersize=12, linestyle='None', label=s)
    for s in sizes_order
]
# --- Легенда 2: маркер = geometry ---
geom_handles2 = [
    Line2D([0], [0], marker=geometry_markers[g], color='gray',
           linestyle='None', markersize=10, label=g)
    for g in geometries
]

leg2a = ax2.legend(handles=scale_handles2, title='scale (color)',
                   loc='upper left', framealpha=0.9)
ax2.add_artist(leg2a)
ax2.legend(handles=geom_handles2, title='geometry (marker)',
           loc='lower right', framealpha=0.9)


# --- Общий заголовок ---
fig.suptitle('Amazon Beauty: GINCF vs Poincare vs Euclidean baseline, across scale\n'
             '(eta=0.001, frozen projection, dropout=0.2 for all three) — same comparison, equal conditions, at every scale',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('c:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/hyperbolicity_vs_quality_4sizes.png', dpi=300, bbox_inches='tight')
#plt.savefig('hyperbolicity_vs_quality_4sizes.pdf', bbox_inches='tight')
plt.show()

print("График сохранен: hyperbolicity_vs_quality_4sizes.png и .pdf")