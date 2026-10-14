"""
Сравнение delta_rel (Gromov hyperbolicity) для матриц из одной папки:
- В папке лежат файлы matrices_epochN.npz (N от 0 до 299)
- Скрипт принимает путь к папке и частоту проверяемых эпох
- Для каждой выбранной эпохи считается delta_rel
"""
import os
import glob
import json
import numpy as np


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


def parse_epoch_number(filepath: str):
    """Извлекает номер эпохи из имени файла matrices_epochN.npz"""
    basename = os.path.basename(filepath)
    # ожидаем ...matrices_epochN.npz
    try:
        after = basename.split("epoch")[-1]
        num = int(after.split(".")[0])
        return num
    except (ValueError, IndexError):
        return None


def main():
    # --- ПАРАМЕТРЫ ---
    # Путь к папке с матрицами
    folder = "Z:/Borisova/MANUL/recsys/SASRec/sasrec_amazon_beauty_noreg_nowarm_6000x3000_seed0"
    #folder = "Z:/Borisova/MANUL/recsys/SASRec/logs_sasrec_isomap/sasrec_amazon_beauty_33"
    json_filename = "delta_rel_results.json"
    init_filename = "D_input_init.npy"

    # Частота проверяемых эпох (например, 30 -> эпохи 0, 30, 60, ...)
    freq = 30

    # Какой ключ использовать из npz: 'D_latent' или 'D_input'
    # Если None — берется 'D_latent', при отсутствии — 'D_input'
    key_name = "D_latent"

    # --- СБОР ФАЙЛОВ ---
    all_files = glob.glob(os.path.join(folder, "matrices_epoch*.npz"))
    if not all_files:
        print(f"В папке не найдено файлов matrices_epoch*.npz: {folder}")
        return

    # Сортируем по номеру эпохи
    files_with_epoch = []
    for f in all_files:
        n = parse_epoch_number(f)
        if n is not None:
            files_with_epoch.append((n, f))
    files_with_epoch.sort(key=lambda x: x[0])

    # Фильтруем по частоте: эпохи 0, freq, 2*freq, ...
    selected = [(n, f) for (n, f) in files_with_epoch if n % freq == 0]
    if not selected:
        print(f"Нет эпох, кратных {freq}. Доступные эпохи: "
              f"{files_with_epoch[0][0]}..{files_with_epoch[-1][0]}")
        return

    print("=" * 60)
    print("Анализ гиперболичности геометрий (delta_rel)")
    print(f"Папка: {folder}")
    print(f"Частота проверки: каждая {freq}-я эпоха")
    print(f"Ключ матрицы: {key_name}")
    print("=" * 60)

    results = {}
    records = []

    init_path = os.path.join(folder, init_filename)
    if os.path.exists(init_path):
        D = np.load(init_path).astype(np.float64)
        delta, diam, delta_rel = calc_delta_rel(D)

        records.append({
            "epoch": -1,          # -1 как маркер "init"
            "label": "init",
            "delta": float(delta),
            "diam": float(diam),
            "delta_rel": float(delta_rel),
            "key": "D_input_init",
            "file": init_filename,
        })

        print(f"[init       ] (D_input_init) "
              f"delta={delta:.4f}, diam={diam:.4f}, delta_rel={delta_rel:.4f}")
    else:
        print(f"[init] Файл не найден: {init_path}")

    for epoch, path in selected:
        data = np.load(path)

        if key_name in data:
            D = data[key_name].astype(np.float64)
            used_key = key_name
        elif "D_latent" in data:
            D = data["D_latent"].astype(np.float64)
            used_key = "D_latent"
        elif "D_input" in data:
            D = data["D_input"].astype(np.float64)
            used_key = "D_input"
        else:
            print(f"[epoch {epoch:>3}] Не найдены ключи D_latent/D_input в {path}")
            continue

        delta, diam, delta_rel = calc_delta_rel(D)
        records.append({
            "epoch": int(epoch),
            "delta": float(delta),
            "diam": float(diam),
            "delta_rel": float(delta_rel),
            "key": used_key,
            "file": os.path.basename(path),
        })

        results[epoch] = delta_rel
        print(f"[epoch {epoch:>3}] ({used_key}) "
              f"delta={delta:.4f}, diam={diam:.4f}, delta_rel={delta_rel:.4f}")

    json_path = os.path.join(folder, json_filename)
    output = {
        "folder": folder,
        "freq": int(freq),
        "key_name": key_name,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n--- ИТОГОВАЯ ТАБЛИЦА ---")
    print(f"{'epoch':<8} | {'delta_rel':<10}")
    print("-" * 25)
    for rec in records:
        print(f"{rec['epoch']:<8} | {rec['delta_rel']:<10.4f}")

    print(f"\nJSON сохранен: {json_path}")

    print("\nИнтерпретация:")
    print("- delta_rel ~ 0.0: Идеально гиперболическая (древовидная) структура.")
    print("- delta_rel ~ 0.4-0.5: Евклидова/случайная структура.")


if __name__ == "__main__":
    main()