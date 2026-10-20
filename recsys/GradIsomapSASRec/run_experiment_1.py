"""
run_experiment_1.py — точка входа без терминала.

Запускает оба скрипта ablation/save для SASRec, подменяя sys.argv.
Просто нажми Run в IDE (или python run_experiment_1.py).

Порядок:
  1. save_euclidean_baseline_geometry_sasrec.py  — обучает чистый SASRec,
     сохраняет sasrec_euclidean_baseline_geometry_<tag>.npz
  2. ablation_geometry_vs_optimization_sasrec.py — сравнивает pure_init /
     epoch0 / epochN на уже сохранённых Z-снимках GradientIsomapSASRec

ВАЖНО: ablation требует, чтобы ДО него уже был запущен sweep
GradientIsomapSASRec и существовала папка logs_<...>/<run>/ с
D_input_init.npy и matrices_epoch*.npz. Если её нет — оба шага ниже
упадут на этапе загрузки Z. Сначала запусти GradientIsomapSASRec
(или свой sweep-скрипт), потом уже этот файл.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ──────────────────────────────────────────────────────────────────
#  НАСТРОЙКИ — меняй только здесь
# ──────────────────────────────────────────────────────────────────

# Общие параметры датасета
DATASET_DIR_NAME = "amazon_beauty"
DATASET_TYPE = "amazon"
AMAZON_CATEGORY = "Beauty_and_Personal_Care"
MAX_USERS = 300
MAX_MOVIES = 800

# Параметры SASRec
LATENT_DIM = 64
HIDDEN_UNITS = 64
MAXLEN = 50
NUM_BLOCKS = 2
NUM_HEADS = 1
DROPOUT_RATE = 0.2
SELECT_BY = "hr"          # "hr" или "loss"
PATIENCE = 3
EPOCHS = 30
BATCH_SIZE = 256
SEED = 0

# Тег для имён файлов (чтобы разные датасеты не перезаписывали друг друга)
TAG = "beauty"

# Папка с Z-снимками от уже завершённого sweep GradientIsomapSASRec.
# Должна содержать: D_input_init.npy и matrices_epoch*.npz
LOGS_FOLDER = os.path.join(
    HERE, "logs_amazon_beauty_sasrec", "sasrec_beauty_01"
)


# ──────────────────────────────────────────────────────────────────
#  Обёртки запуска
# ──────────────────────────────────────────────────────────────────

def _run_module(module_name: str, argv: list):
    """Импортирует модуль, подменяет sys.argv и вызывает main()."""
    import importlib
    print(f"\n{'=' * 70}\n>>> {module_name}\n{'=' * 70}", flush=True)

    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)   # чтобы повторный запуск подхватил новый argv
        mod.main()
    finally:
        sys.argv = old_argv


def run_save_euclidean():
    argv = [
        "--dataset_dir_name", DATASET_DIR_NAME,
        "--dataset_type", DATASET_TYPE,
        "--amazon_category", AMAZON_CATEGORY,
        "--max_users", str(MAX_USERS),
        "--max_movies", str(MAX_MOVIES),
        "--tag", TAG,
        "--select_by", SELECT_BY,
        "--patience", str(PATIENCE),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--hidden_units", str(HIDDEN_UNITS),
        "--maxlen", str(MAXLEN),
        "--num_blocks", str(NUM_BLOCKS),
        "--num_heads", str(NUM_HEADS),
        "--dropout_rate", str(DROPOUT_RATE),
        "--seed", str(SEED),
    ]
    _run_module("save_euclidean_baseline_geometry_sasrec", argv)


def run_ablation():
    if not os.path.isdir(LOGS_FOLDER):
        raise FileNotFoundError(
            f"LOGS_FOLDER не существует: {LOGS_FOLDER}\n"
            f"Сначала запусти GradientIsomapSASRec sweep, чтобы получить "
            f"D_input_init.npy и matrices_epoch*.npz."
        )
    argv = [
        "--logs_folder", LOGS_FOLDER,
        "--dataset_dir_name", DATASET_DIR_NAME,
        "--dataset_type", DATASET_TYPE,
        "--amazon_category", AMAZON_CATEGORY,
        "--max_users", str(MAX_USERS),
        "--max_movies", str(MAX_MOVIES),
        "--latent_dim", str(LATENT_DIM),
        "--hidden_units", str(HIDDEN_UNITS),
        "--maxlen", str(MAXLEN),
        "--num_blocks", str(NUM_BLOCKS),
        "--num_heads", str(NUM_HEADS),
        "--dropout_rate", str(DROPOUT_RATE),
        "--select_by", SELECT_BY,
        "--patience", str(PATIENCE),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
    ]
    _run_module("ablation_geometry_vs_optimization_sasrec", argv)


# ──────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────

def main(type):
    if type == "RUN_SAVE_EUCLIDEAN":
        run_save_euclidean()

    if type == "RUN_ABLATION":
        run_ablation()

    print("\n[run_experiment_1] done.", flush=True)


if __name__ == "__main__":
    type = "RUN_SAVE_EUCLIDEAN"
    main(type)
    