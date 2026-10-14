import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================
# Укажите путь к папке, где лежат ваши CSV файлы
GEOMETRY_FOLDER_PATH = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/logs_sasrec_isomap" 

# Какую геометрическую метрику показать сверху?
GEOMETRY_METRIC = "kruskal_stress"  

# Размеры выборок в порядке возрастания (для красивого графика)
SIZES_ORDER = ["300x800", "1000x1200", "3000x2500", "6000x3000"]
METHODS_ORDER = ["pure_init (Euclidean)", "epoch0 (1 step)", "epochN (converged)"]

# Цвета для размеров (от светлого к темному или контрастные)
COLORS = {
    "300x800": "#4daf4a",    # Зеленый
    "1000x1200": "#377eb8",  # Синий
    "3000x2500": "#ff7f00",  # Оранжевый
    "6000x3000": "#e41a1c"   # Красный
}

# ==========================================
# 2. ПАРСИНГ ТЕКСТОВЫХ ДАННЫХ (КАЧЕСТВО)
# ==========================================
raw_text = """1000x1200_seed0	1000x1200_seed1	1000x1200_seed2	300x800_seed0	300x800_seed1	300x800_seed2	3000x2500_seed0	3000x2500_seed1	3000x2500_seed2	6000x3000_seed0	6000x3000_seed1	6000x3000_seed2
pure_init (Euclidean)           HR@10=0.3507  NDCG@10=0.1825	pure_init (Euclidean)           HR@10=0.3538  NDCG@10=0.1830	pure_init (Euclidean)           HR@10=0.3517  NDCG@10=0.1821	pure_init (Euclidean)           HR@10=0.3188  NDCG@10=0.1537	pure_init (Euclidean)           HR@10=0.3255  NDCG@10=0.1543	pure_init (Euclidean)           HR@10=0.3255  NDCG@10=0.1542	pure_init (Euclidean)           HR@10=0.3256  NDCG@10=0.1795	pure_init (Euclidean)           HR@10=0.3246  NDCG@10=0.1798	pure_init (Euclidean)           HR@10=0.3235  NDCG@10=0.1783	pure_init (Euclidean)           HR@10=0.3422  NDCG@10=0.1892	pure_init (Euclidean)           HR@10=0.3438  NDCG@10=0.1901	pure_init (Euclidean)           HR@10=0.3473  NDCG@10=0.1926
epoch0 (1 step)                 HR@10=0.3374  NDCG@10=0.1733	epoch0 (1 step)                 HR@10=0.3528  NDCG@10=0.1791	epoch0 (1 step)                 HR@10=0.3538  NDCG@10=0.1817	epoch0 (1 step)                 HR@10=0.3591  NDCG@10=0.1716	epoch0 (1 step)                 HR@10=0.3289  NDCG@10=0.1568	epoch0 (1 step)                 HR@10=0.3221  NDCG@10=0.1569	epoch0 (1 step)                 HR@10=0.3558  NDCG@10=0.1942	epoch0 (1 step)                 HR@10=0.3407  NDCG@10=0.1884	epoch0 (1 step)                 HR@10=0.3537  NDCG@10=0.1931	epoch0 (1 step)                 HR@10=0.3496  NDCG@10=0.1929	epoch0 (1 step)                 HR@10=0.3583  NDCG@10=0.1974	epoch0 (1 step)                 HR@10=0.3569  NDCG@10=0.1954
epochN (converged)              HR@10=0.4141  NDCG@10=0.2142	epochN (converged)              HR@10=0.4018  NDCG@10=0.2095	epochN (converged)              HR@10=0.4100  NDCG@10=0.2147	epochN (converged)              HR@10=0.3960  NDCG@10=0.2057	epochN (converged)              HR@10=0.4027  NDCG@10=0.1944	epochN (converged)              HR@10=0.3591  NDCG@10=0.1803	epochN (converged)              HR@10=0.4647  NDCG@10=0.2485	epochN (converged)              HR@10=0.5047  NDCG@10=0.2690	epochN (converged)              HR@10=0.4854  NDCG@10=0.2630	epochN (converged)              HR@10=0.4790  NDCG@10=0.2622	epochN (converged)              HR@10=0.4947  NDCG@10=0.2648	epochN (converged)              HR@10=0.4744  NDCG@10=0.2539"""

# Парсим сырой текст в DataFrame
rows = []
headers = raw_text.strip().split('\n')[0].split('\t')
for line in raw_text.strip().split('\n')[1:]:
    cells = line.split('\t')
    for cell in cells:
        # Извлекаем метод, HR и NDCG с помощью regex
        method_match = re.match(r"([a-zA-Z0-9\(\) ]+?)\s+HR@10=", cell)
        metrics_match = re.search(r"HR@10=([\d\.]+)\s+NDCG@10=([\d\.]+)", cell)
        if method_match and metrics_match:
            method = method_match.group(1).strip()
            hr = float(metrics_match.group(1))
            ndcg = float(metrics_match.group(2))
            
            # Находим размер и сид из заголовка (он дублируется, берем первый)
            # В вашем тексте заголовки идут в порядке 1000, 300, 3000, 6000. 
            # Нам нужно сопоставить ячейку с заголовком.
            # Для простоты, так как структура жесткая, парсим размер из заголовка соответствующей колонки
            # (В реальном коде лучше мапить по индексу, но здесь мы восстановим из текста)
            pass

# Более надежный парсер, учитывающий структуру вашего текста:
data_quality = []
lines = raw_text.strip().split('\n')
headers = lines[0].split('\t')

for line in lines[1:]:
    cells = line.split('\t')
    for i, cell in enumerate(cells):
        header = headers[i]
        size, seed = header.split('_seed')
        seed = int(seed)
        
        method = re.match(r"([^\t]+?)\s+HR@10=", cell).group(1).strip()
        hr = float(re.search(r"HR@10=([\d\.]+)", cell).group(1))
        ndcg = float(re.search(r"NDCG@10=([\d\.]+)", cell).group(1))
        
        data_quality.append({"size": size, "seed": seed, "method": method, "HR@10": hr, "NDCG@10": ndcg})

df_quality = pd.DataFrame(data_quality)

# ==========================================
# 3. ЗАГРУЗКА ГЕОМЕТРИИ ИЗ CSV
# ==========================================
def load_geometry_data(folder, metric):
    files = glob.glob(os.path.join(folder, "geometry_results_*.csv"))
    if not files:
        raise FileNotFoundError(f"В папке {folder} не найдено файлов geometry_results_*.csv")
    
    df_list = []
    for f in files:
        basename = os.path.basename(f).replace("geometry_results_", "").replace(".csv", "")
        parts = basename.split("_seed")
        size = parts[0]
        seed = int(parts[1])
        
        df = pd.read_csv(f)
        df['size'] = size
        df['seed'] = seed
        df_list.append(df[['epoch', 'size', 'seed', metric]])
        
    return pd.concat(df_list, ignore_index=True)

df_geometry = load_geometry_data(GEOMETRY_FOLDER_PATH, GEOMETRY_METRIC)

# Агрегация геометрии (Mean ± Std по сидам для каждого размера)
geo_agg = df_geometry.groupby(['size', 'epoch'])[GEOMETRY_METRIC].agg(['mean', 'std']).reset_index()

# Агрегация качества (Mean ± Std по сидам для каждого размера и метода)
# Маппим методы на "псевдо-эпохи" для визуального выравнивания с верхним графиком
stage_to_epoch = {"pure_init (Euclidean)": 0, "epoch0 (1 step)": 1, "epochN (converged)": 300}
df_quality['stage_epoch'] = df_quality['method'].map(stage_to_epoch)

qual_agg = df_quality.groupby(['size', 'method', 'stage_epoch'])['NDCG@10'].agg(['mean', 'std']).reset_index()

# ==========================================
# 4. ПОСТРОЕНИЕ ГРАФИКА
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=False)
fig.subplots_adjust(hspace=0.05) # Минимальный отступ между графиками

# --- ВЕРХНИЙ ГРАФИК: Геометрия (Непрерывный) ---
for size in SIZES_ORDER:
    subset = geo_agg[geo_agg['size'] == size].sort_values('epoch')
    color = COLORS[size]
    
    # Линия среднего
    ax1.plot(subset['epoch'], subset['mean'], color=color, linewidth=2.5, label=size, zorder=3)
    # Заштрихованная область (стандартное отклонение)
    ax1.fill_between(subset['epoch'], 
                     subset['mean'] - subset['std'], 
                     subset['mean'] + subset['std'], 
                     color=color, alpha=0.2, zorder=2)

ax1.set_ylabel(f"Топология: {GEOMETRY_METRIC}", fontsize=12, fontweight='bold')
ax1.set_title("Влияние размера выборки на геометрию латентного пространства и качество рекомендаций", 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax1.legend(title="Размер матрицы", loc='upper left', framealpha=0.9)
ax1.set_xlim(0, 300)

# Добавляем вертикальные пунктиры для ориентира (Init, Epoch0, EpochN)
for ep in [0, 1, 300]:
    ax1.axvline(x=ep, color='gray', linestyle=':', alpha=0.5, zorder=1)

# --- НИЖНИЙ ГРАФИК: Качество (Дискретные точки) ---
# Используем те же "псевдо-эпохи" (0, 1, 300) для X, чтобы точки были выровданы с пунктирами сверху
for size in SIZES_ORDER:
    subset = qual_agg[qual_agg['size'] == size].sort_values('stage_epoch')
    color = COLORS[size]
    
    # Линия, соединяющая точки (для ощущения траектории)
    ax2.plot(subset['stage_epoch'], subset['mean'], color=color, linewidth=2, alpha=0.5, zorder=2)
    # Сами точки с error bars
    ax2.errorbar(subset['stage_epoch'], subset['mean'], yerr=subset['std'], 
                 fmt='o', color=color, markersize=9, markeredgecolor='white', 
                 markeredgewidth=1.5, capsize=4, label=size, zorder=3)

ax2.set_ylabel("Качество: NDCG@10", fontsize=12, fontweight='bold')
ax2.set_xlabel("Этап обучения (Эпоха)", fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Настраиваем X-ось нижнего графика (категории вместо чисел)
ax2.set_xticks([0, 1, 300])
ax2.set_xticklabels(["Init\n(Euclidean)", "Epoch 0\n(1 step)", "Epoch N\n(Converged)"], fontsize=10)
ax2.set_xlim(-20, 320)

# --- АННОТАЦИЯ (Гвоздь программы) ---
# Добавляем стрелку/текст, указывающий на корреляцию
#ax1.annotate("Рост циклов (h1)\nна больших данных", 
#             xy=(250, geo_agg[geo_agg['size']=='6000x3000']['mean'].iloc[-1]), 
#             xytext=(150, geo_agg['mean'].max() * 0.9),
#             arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
#             fontsize=10, color='red', fontweight='bold', ha='center')
#
#ax2.annotate("Преимущество гиперболичности\nнивелируется шумом", 
#             xy=(300, qual_agg[(qual_agg['size']=='6000x3000') & (qual_agg['method']=='epochN (converged)')]['mean'].iloc[0]), 
#             xytext=(150, 0.18),
#             arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
#             fontsize=10, color='red', fontweight='bold', ha='center')

# --- СОХРАНЕНИЕ ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_name = f"{GEOMETRY_FOLDER_PATH}/geometry_vs_quality_{GEOMETRY_METRIC}.png"
plt.savefig(out_name, dpi=300, bbox_inches='tight')
# Для статьи лучше сохранить в PDF:
# plt.savefig(out_name.replace('.png', '.pdf'), bbox_inches='tight')
plt.show()
print(f"График успешно сохранен: {out_name}")