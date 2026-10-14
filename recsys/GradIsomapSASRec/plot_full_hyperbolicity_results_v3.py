import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Загрузка данных
df = pd.read_csv('C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/full_hyperbolicity_table.csv')

# --- Диагностика ---
print("geometry values:", df['geometry'].unique())
print("size values:",     df['size'].unique())
print("columns:",         list(df.columns))

# Агрегация по сидам (только mean — std больше не нужен)
df_agg = df.groupby(['size', 'geometry'])[['delta_rel', 'HR@10', 'NDCG@10']].mean().reset_index()
df_agg = df_agg.rename(columns={
    'delta_rel': 'delta_rel_mean',
    'HR@10':     'HR_mean',
    'NDCG@10':   'NDCG_mean',
})

# --- Порядок и палитры ---
sizes_order = ['300x800', '1000x1200', '3000x2500', '6000x3000']
geometries  = ['GISASRec', 'Poincare', 'Euclidean']   # ← подставьте реальные имена из CSV

size_colors = {
    '300x800':   '#4daf4a',
    '1000x1200': '#377eb8',
    '3000x2500': '#ff7f00',
    '6000x3000': '#e41a1c',
}
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
    color = size_colors.get(size, 'gray')

    for geom in geometries:
        marker = geometry_markers.get(geom, 'o')

        # --- 1) отдельные сиды: мелкие полупрозрачные точки ---
        raw = df[(df['size'] == size) & (df['geometry'] == geom)]
        if not raw.empty:
            ax1.scatter(
                raw['delta_rel'], raw['HR@10'],
                s=40, marker=marker, color=color,
                alpha=0.35, edgecolors='none',
                zorder=2,
            )

        # --- 2) среднее: крупный маркер с чёрной обводкой ---
        agg = df_agg[(df_agg['size'] == size) & (df_agg['geometry'] == geom)]
        if not agg.empty:
            ax1.scatter(
                agg['delta_rel_mean'], agg['HR_mean'],
                s=180, marker=marker, color=color,
                edgecolors='black', linewidths=1.5,
                zorder=3,
            )

ax1.set_xlabel('delta_rel (Gromov hyperbolicity, normalized — LOWER = more tree-like/hyperbolic)', fontsize=11)
ax1.set_ylabel('test HR@10', fontsize=12, fontweight='bold')
ax1.set_title('Hyperbolicity vs downstream quality, all 3 geometries x all 4 scales\n'
              'still no clean hyperbolicity relationship, at any scale',
              fontsize=12, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# --- Легенды ax1 ---
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
style_handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
           markeredgecolor='black', markersize=12, linestyle='None',
           label='mean'),
    Line2D([0], [0], marker='o', color='gray', alpha=0.4,
           markersize=7, linestyle='None', label='individual seed'),
]

leg1 = ax1.legend(handles=scale_handles, title='scale (color)',
                  loc='upper left', framealpha=0.9)
ax1.add_artist(leg1)

leg1b = ax1.legend(handles=geom_handles, title='geometry (marker)',
                   loc='lower right', framealpha=0.9)
ax1.add_artist(leg1b)

ax1.legend(handles=style_handles, title='point type',
           loc='upper right', framealpha=0.9)

# ==========================================
# ПРАВЫЙ ГРАФИК: Quality vs Scale
# ==========================================
x_positions = {s: i for i, s in enumerate(sizes_order)}

# Небольшой горизонтальный "джиттер" для сидов одной группы,
# чтобы точки не накладывались друг на друга
def jitter(n, width=0.12):
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)

for geom in geometries:
    marker = geometry_markers.get(geom, 'o')

    # --- отдельные сиды ---
    for size in sizes_order:
        raw = df[(df['size'] == size) & (df['geometry'] == geom)].copy()
        if raw.empty:
            continue
        color = size_colors.get(size, 'gray')
        x0 = x_positions[size]
        xs = x0 + jitter(len(raw))
        ax2.scatter(
            xs, raw['HR@10'],
            s=45, marker=marker, color=color,
            alpha=0.35, edgecolors='none',
            zorder=2,
        )

    # --- пунктирная линия по средним ---
    agg = df_agg[df_agg['geometry'] == geom].copy()
    agg['x'] = agg['size'].map(x_positions)
    agg = agg.dropna(subset=['x']).sort_values('x')
    if not agg.empty:
        ax2.plot(agg['x'], agg['HR_mean'],
                 linestyle='--', color='gray',
                 alpha=0.6, linewidth=1.5, zorder=1)

    # --- средние: крупные маркеры ---
    for _, row in agg.iterrows():
        size  = row['size']
        color = size_colors.get(size, 'gray')
        ax2.scatter(
            row['x'], row['HR_mean'],
            s=200, marker=marker, color=color,
            edgecolors='black', linewidths=1.5,
            zorder=3,
        )

ax2.set_xticks(range(len(sizes_order)))
ax2.set_xticklabels(sizes_order, fontsize=10)
ax2.set_xlabel('scale', fontsize=12)
ax2.set_ylabel('test HR@10', fontsize=12, fontweight='bold')
ax2.set_title("Quality vs scale, all 3 geometries under equal conditions\n"
              "(dashed line = mean per scale; GISASRec's edge over Euclidean narrows as scale grows)",
              fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# --- Легенды ax2 ---
scale_handles2 = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=size_colors[s], markeredgecolor='black',
           markersize=12, linestyle='None', label=s)
    for s in sizes_order
]
geom_handles2 = [
    Line2D([0], [0], marker=geometry_markers[g], color='gray',
           linestyle='None', markersize=10, label=g)
    for g in geometries
]
style_handles2 = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
           markeredgecolor='black', markersize=12, linestyle='None',
           label='mean'),
    Line2D([0], [0], marker='o', color='gray', alpha=0.4,
           markersize=7, linestyle='None', label='individual seed'),
]

leg2a = ax2.legend(handles=scale_handles2, title='scale (color)',
                   loc='upper left', framealpha=0.9)
ax2.add_artist(leg2a)

leg2b = ax2.legend(handles=geom_handles2, title='geometry (marker)',
                   loc='lower right', framealpha=0.9)
ax2.add_artist(leg2b)

ax2.legend(handles=style_handles2, title='point type',
           loc='upper right', framealpha=0.9)

# --- Общий заголовок ---
fig.suptitle('Amazon Beauty: GISASRec vs Poincare vs Euclidean baseline, across scale\n'
             '(eta=0.001, frozen projection, dropout=0.2 for all three) — same comparison, equal conditions, at every scale',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/hyperbolicity_vs_quality_4sizes_v3.png', dpi=300, bbox_inches='tight')
#plt.savefig('hyperbolicity_vs_quality_4sizes.pdf', bbox_inches='tight')
plt.show()

print("График сохранен: hyperbolicity_vs_quality_4sizes.png и .pdf")