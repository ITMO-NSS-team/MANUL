"""
Собирает два CSV, которые читает hyperbolicity_scaling_comparison.py,
используя только данные масштаба 300u/800i.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# ВАЖНО: скрипт-рисовалка фильтрует строки по НАЧАЛУ строки 'geometry':
#   GINCF     -> startswith("GINCF seed") И содержит "restored-best"
#   Poincare  -> startswith("Poincare")
#   Euclidean -> startswith("Euclidean")
# delta_rel — нормализованная гиперболичность (Gromov), нужна для левой
# панели. Если её нет — точку на левой панели не увидите, но на правой
# (HR@10 vs scale) она появится.
# ----------------------------------------------------------------------

rows = [
    # GINCF, сходящийся чекпойнт (epochN = restored-best)
    {"geometry": "GINCF seed0 restored-best",
     "delta_rel": None,   # <-- подставьте своё значение
     "hr10": 0.3792},

    # (опционально) промежуточный чекпойнт — в текущем фильтре он
    # НЕ попадёт на график, т.к. нет "restored-best" в имени
    # {"geometry": "GINCF seed0 epoch0", "delta_rel": None, "hr10": 0.3289},

    # Euclidean baseline, pure_init
    {"geometry": "Euclidean (pure_init)",
     "delta_rel": None,   # <-- подставьте своё значение
     "hr10": 0.3255},

    # Poincare pretrained — добавьте, если считали его
    # {"geometry": "Poincare", "delta_rel": None, "hr10": 0.XXXX},
]

base = pd.DataFrame(rows, columns=["geometry", "delta_rel", "hr10"])
base.to_csv(os.path.join(HERE, "full_hyperbolicity_table_frozenproj.csv"), index=False)

# Пустая таблица с правильными заголовками — чтобы рисовалка не падала
scaling = pd.DataFrame(columns=["geometry", "delta_rel", "hr10"])
scaling.to_csv(os.path.join(HERE, "full_hyperbolicity_table_scaling.csv"), index=False)

print("OK: full_hyperbolicity_table_frozenproj.csv")
print("OK: full_hyperbolicity_table_scaling.csv (empty)")