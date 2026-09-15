"""
Same figure type as hyperbolicity_frozenproj_comparison.png (hyperbolicity
vs quality scatter + HR@10-vs-H1 bar chart), extended with the two larger
scale-test points (1000u/1200i, 3000u/2500i) under the same adopted config
(eta_outer=0.001, freeze_item_projection=True, dropout=0.2), to show
whether quality trends up with scale and whether same-scale seeds cluster.
See docs/recsys_paper_diary.md and the "GINCF outer-loop noise & warm-start"
memory, 2026-09-15.

Caveat plotted explicitly: only 300u/800i has seed repeats (3); the two
larger scale points are n=1 each.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"C:\Users\Julia\Documents\NSS_lab\документы\2027 WWW Recsys\process_docs"

base = pd.read_csv(os.path.join(HERE, "full_hyperbolicity_table_frozenproj.csv"))
scaling = pd.read_csv(os.path.join(HERE, "full_hyperbolicity_table_scaling.csv"))

gincf_300 = base[base["geometry"].str.contains("GINCF seed") & base["geometry"].str.contains("restored-best")].copy()
poincare = base[base["geometry"].str.contains("Poincare")].iloc[0]
euclidean = base[base["geometry"].str.contains("Euclidean")].iloc[0]

SCALE_COLORS = {"300u/800i": "tab:green", "1000u/1200i": "tab:blue", "3000u/2500i": "tab:orange"}
SCALE_ORDER = ["300u/800i", "1000u/1200i", "3000u/2500i"]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Left: hyperbolicity (delta_rel) vs downstream quality (HR@10) ---
ax = axes[0]
for _, row in gincf_300.iterrows():
    ax.scatter(row["delta_rel"], row["hr10"], s=140, color=SCALE_COLORS["300u/800i"],
               edgecolor="black", zorder=3)
    seed = row["geometry"].split("seed=")[1].split(",")[0]
    ax.annotate(f"300u/800i seed={seed}", (row["delta_rel"], row["hr10"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)

for scale_label in ["1000u/1200i", "3000u/2500i"]:
    rows = scaling[scaling["geometry"].str.contains(scale_label)]
    if rows.empty:
        continue
    row = rows.iloc[0]
    ax.scatter(row["delta_rel"], row["hr10"], s=200, marker="D", color=SCALE_COLORS[scale_label],
               edgecolor="black", zorder=3)
    ax.annotate(f"{scale_label} (n=1)", (row["delta_rel"], row["hr10"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight="bold")

ax.scatter(poincare["delta_rel"], poincare["hr10"], s=140, color="mediumpurple", edgecolor="black", zorder=3)
ax.annotate("Poincare-pretrained", (poincare["delta_rel"], poincare["hr10"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9)

ax.scatter(euclidean["delta_rel"], euclidean["hr10"], s=140, color="firebrick", edgecolor="black", zorder=3)
ax.annotate("Euclidean NeuMF", (euclidean["delta_rel"], euclidean["hr10"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9)

ax.set_xlabel("delta_rel (Gromov hyperbolicity, normalized — LOWER = more tree-like/hyperbolic)")
ax.set_ylabel("test HR@10")
ax.set_title("Hyperbolicity vs downstream quality, across scale\nquality rises with scale (diamonds); still no clean hyperbolicity relationship")
ax.grid(alpha=0.3)

# --- Right: HR@10 vs scale, colored/grouped, showing the trend directly ---
ax = axes[1]
plot_rows = []
for _, row in gincf_300.iterrows():
    plot_rows.append(("300u/800i", row["hr10"]))
for scale_label in ["1000u/1200i", "3000u/2500i"]:
    rows = scaling[scaling["geometry"].str.contains(scale_label)]
    if rows.empty:
        continue
    plot_rows.append((scale_label, rows.iloc[0]["hr10"]))

for scale_label in SCALE_ORDER:
    xs = [i for i, (s, _) in enumerate(plot_rows) if s == scale_label]
    ys = [plot_rows[i][1] for i in xs]
    if not xs:
        continue
    ax.scatter([scale_label] * len(ys), ys, s=180, color=SCALE_COLORS[scale_label],
               edgecolor="black", zorder=3, label=f"{scale_label} (n={len(ys)})")

ax.axhline(euclidean["hr10"], color="firebrick", linestyle="--", linewidth=1.5,
           label=f"Euclidean NeuMF baseline (0.123, 300u/800i)")
ax.set_xlabel("scale")
ax.set_ylabel("test HR@10")
ax.set_title("GINCF quality vs scale\n(300u/800i has 3 seeds clustering closely; larger scales are n=1 so far)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("Amazon Beauty: geometry diagnostics for the converging method across scale\n"
             "(eta=0.001, frozen projection, dropout=0.2) — does quality/geometry hold as scale grows?",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT_DIR, "hyperbolicity_scaling_comparison.png")
fig.savefig(out_path, dpi=130)
print(f"[Save] {out_path}")
