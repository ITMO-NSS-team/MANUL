import os
import glob
import numpy as np

# Базовая директория проекта (замените на вашу, если нужно)
BASE_DIR = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec"

# Функция расчета delta_hyperbolicity
def delta_hyperbolicity(D: np.ndarray, basepoint: int = 0) -> float:
    n = D.shape[0]
    w = basepoint
    dw = D[w, :]
    A = 0.5 * (dw[:, None] + dw[None, :] - D)
    M = np.full((n, n), -np.inf, dtype=D.dtype)
    for y in range(n):
        cand = np.minimum(A[:, y:y + 1], A[y:y + 1, :])
        np.maximum(M, cand, out=M)
    delta = float(np.max(M - A))
    return delta

def calc_delta_rel(D):
    diam = float(D.max())
    delta = delta_hyperbolicity(D, basepoint=0)
    delta_rel = (2.0 * delta / diam) if diam > 0 else float("nan")
    return delta, diam, delta_rel

def main():
    print("="*70)
    print("Анализ гиперболичности геометрий (delta_rel) для 6 подходов")
    print("="*70)
    
    # Конфигурация: (Название, Путь к файлу, Ключ для извлечения матрицы)
    experiments = [
        (
            "Poincare frozen", 
            os.path.join(BASE_DIR, "poincare_sasrec_baseline_geometry.npz"), 
            "D"  # Гиперболические расстояния
        ),
        (
            "Eucl frozen", 
            os.path.join(BASE_DIR, "euclidean_sasrec_geometry.npz"), 
            "D"  # Евклидовы расстояния MDS
        ),
        (
            "Eucl baseline", 
            os.path.join(BASE_DIR, "euclidean_sasrec_baseline_geometry3.npz"), 
            "D"  # Евклидовы расстояния выученных эмбеддингов
        ),
        (
            "SASRec classic", 
            os.path.join(BASE_DIR, "classic_sasrec_geometry", "exp_1", "classic_sasrec_geometry.npz"), 
            "D"  # Евклидовы расстояния выученных эмбеддингов
        ),
        (
            "Ablation 24 best", 
            os.path.join(BASE_DIR, "ablation", "ablation_exp_1", "epochn_converged.npz"), 
            "D"  # Евклидовы расстояния в Z (сошедшаяся абляция)
        ),
        (
            "Ablation 24 last", 
            os.path.join(BASE_DIR, "ablation", "ablation_exp_1_last", "epochn_converged.npz"), 
            "D"  # Евклидовы расстояния в Z (сошедшаяся абляция)
        ),
        #(
        #    "Ablation 24 last", 
        #    None, # Путь зададим динамически ниже, так как это файл из логов GINCF
        #    "D_latent" # Евклидовы расстояния в Z (последняя эпоха GINCF)
        #)
    ]

    ## Динамически находим последний файл для "Ablation 24 last" (GINCF logs)
    #gincf_folder = os.path.join(BASE_DIR, "logs_sasrec_isomap", "sasrec_amazon_beauty_24")
    #epoch_files = glob.glob(os.path.join(gincf_folder, "matrices_epoch*.npz"))
    #if epoch_files:
    #    epoch_files.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
    #    # Берем последнюю эпоху (например, 81)
    #    experiments[5] = ("Ablation 24 last", epoch_files[-1], "D_latent")
    #else:
    #    print(f"[WARNING] Папка с логами GINCF не найдена или пуста: {gincf_folder}")

    results = []
    
    for name, path, key in experiments:
        print(f"\n[{name}]")
        if path is None or not os.path.exists(path):
            print(f"  ❌ Файл не найден: {path}")
            results.append((name, np.nan, np.nan, np.nan, key))
            continue
            
        try:
            data = np.load(path)
            
            if key not in data:
                print(f"  ❌ Ключ '{key}' не найден в файле. Доступные ключи: {list(data.keys())}")
                results.append((name, np.nan, np.nan, np.nan, key))
                continue
                
            D = data[key].astype(np.float64)
            delta, diam, delta_rel = calc_delta_rel(D)
            
            print(f"  ✅ Берем матрицу: {key}")
            print(f"  📏 delta={delta:.4f}, diam={diam:.4f}, delta_rel={delta_rel:.4f}")
            
            results.append((name, delta, diam, delta_rel, key))
            
        except Exception as e:
            print(f"  ❌ Ошибка при чтении файла: {e}")
            results.append((name, np.nan, np.nan, np.nan, key))

    # --- ИТОГОВАЯ ТАБЛИЦА ---
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*70)
    print(f"{'Geometry':<20} | {'Matrix Key':<10} | {'delta':<10} | {'diam':<10} | {'delta_rel':<10}")
    print("-" * 70)
    for name, delta, diam, delta_rel, key in results:
        d_str = f"{delta:.4f}" if not np.isnan(delta) else "N/A"
        diam_str = f"{diam:.4f}" if not np.isnan(diam) else "N/A"
        dr_str = f"{delta_rel:.4f}" if not np.isnan(delta_rel) else "N/A"
        print(f"{name:<20} | {key:<10} | {d_str:<10} | {diam_str:<10} | {dr_str:<10}")

    print("\n💡 Интерпретация delta_rel:")
    print("  ~ 0.00 - 0.15 : Идеально гиперболическая (древовидная) структура.")
    print("  ~ 0.20 - 0.30 : Умеренная гиперболичность (хорошо для иерархий).")
    print("  ~ 0.40 - 0.50 : Евклидова или случайная структура (плоское пространство).")
    print("\n🎯 Ожидаемый порядок (от самого гиперболического к самому евклидову):")
    print("  Poincare < GINCF (Ablation last/best) < Euclidean / Classic")

if __name__ == "__main__":
    main()