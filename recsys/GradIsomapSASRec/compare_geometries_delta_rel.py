"""
Сравнение delta_rel (Gromov hyperbolicity) для трех геометрий:
1. Euclidean (из save_euclidean_baseline_geometry_sasrec.py)
2. Poincare (из poincare_baseline_sasrec.py)
3. GINCF/SASRecIsomap (из matrices_epoch*.npz основной папки обучения)
"""
import os
import numpy as np

#HERE = os.path.dirname(os.path.abspath(__file__))

# Функция расчета delta_hyperbolicity (из analyze_hyperbolicity.py)
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
    # --- ПУТИ К ФАЙЛАМ ---
    # 1. Euclidean
    euclidean_path = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/euclidean_sasrec_geometry.npz"
    
    # 2. Poincare
    poincare_path = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/poincare_sasrec_baseline_geometry.npz"
    
    # 3. GINCF (SASRecIsomap) - берем последнюю эпоху из основной папки обучения
    # Замени 'sasrec_amazon_beauty_24' на твою папку и номер эпохи
    gincf_folder = "C:/NSS_lab_recsys/2_sem/nss_manul_mnist_reg_example/MANUL/recsys/GradIsomapSASRec/logs_sasrec_isomap/sasrec_amazon_beauty_24" 
    # Найдем последний файл матриц
    import glob
    epoch_files = glob.glob(os.path.join(gincf_folder, "matrices_epoch*.npz"))
    # Сортируем по номеру эпохи
    epoch_files.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
    #gincf_path = epoch_files[-1] if epoch_files else None
    gincf_path = epoch_files[81] if epoch_files else None
    
    print("="*60)
    print("Анализ гиперболичности геометрий (delta_rel)")
    print("="*60)
    
    results = {}
    
    # 1. Euclidean
    if os.path.exists(euclidean_path):
        data = np.load(euclidean_path)
        D = data['D'].astype(np.float64)
        delta, diam, delta_rel = calc_delta_rel(D)
        results['Euclidean'] = delta_rel
        print(f"[Euclidean] delta={delta:.4f}, diam={diam:.4f}, delta_rel={delta_rel:.4f}")
    else:
        print(f"[Euclidean] Файл не найден: {euclidean_path}")
        
    # 2. Poincare
    if os.path.exists(poincare_path):
        data = np.load(poincare_path)
        D = data['D'].astype(np.float64)
        delta, diam, delta_rel = calc_delta_rel(D)
        results['Poincare'] = delta_rel
        print(f"[Poincare ] delta={delta:.4f}, diam={diam:.4f}, delta_rel={delta_rel:.4f}")
    else:
        print(f"[Poincare] Файл не найден: {poincare_path}")
        
    # 3. GINCF
    if gincf_path and os.path.exists(gincf_path):
        data = np.load(gincf_path)
        # Для GINCF берем D_latent (евклидовы расстояния в Z пространстве)
        # Если хочешь D_input (то, что оптимизирует Isomap), замени ключ на 'D_input'
        if 'D_latent' in data:
            D = data['D_latent'].astype(np.float64)
            key_name = 'D_latent'
        elif 'D_input' in data:
            D = data['D_input'].astype(np.float64)
            key_name = 'D_input'
        else:
            print(f"[GINCF] Не найдены ключи D_latent или D_input в {gincf_path}")
            return
            
        delta, diam, delta_rel = calc_delta_rel(D)
        results['GINCF'] = delta_rel
        print(f"[GINCF    ] ({key_name}) delta={delta:.4f}, diam={diam:.4f}, delta_rel={delta_rel:.4f}")
    else:
        print(f"[GINCF] Файл не найден: {gincf_path}")
        
    print("\n--- ИТОГОВАЯ ТАБЛИЦА ---")
    print(f"{'Geometry':<12} | {'delta_rel':<10}")
    print("-" * 25)
    for name, val in results.items():
        print(f"{name:<12} | {val:<10.4f}")
        
    print("\nИнтерпретация:")
    print("- delta_rel ~ 0.0: Идеально гиперболическая (древовидная) структура.")
    print("- delta_rel ~ 0.4-0.5: Евклидова/случайная структура.")
    print("Ожидается: Poincare < GINCF < Euclidean (если GINCF выучил гиперболичность).")

if __name__ == "__main__":
    main()