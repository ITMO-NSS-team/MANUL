"""
Same figure type as hyperbolicity_frozenproj_comparison.png (hyperbolicity
vs quality scatter + a second panel), extended across all three scales
tested (300u/800i, 1000u/1200i, 3000u/2500i) under the same adopted config
(eta_outer=0.001, freeze_item_projection=True, dropout=0.2) - and now with
Poincare-pretrained and the fair (dropout=0.2) Euclidean baseline computed
at every scale too (previously only existed at 300u/800i), so all three
geometries are compared under equal conditions at every scale, per the
user's explicit request (2026-09-16: "чтобы график выглядел что все в
равных условиях"). See docs/recsys_paper_diary.md, 2026-09-15/16/17.

Seed counts per scale (n=1 caveat still applies at the larger scales for
Poincare/Euclidean, which aren't re-seeded - only GINCF has repeats):
  300u/800i:   GINCF x3, Poincare x1, Euclidean x1
  1000u/1200i: GINCF x2 (seed1 still training), Poincare x1, Euclidean x1
  3000u/2500i: GINCF x3, Poincare x1, Euclidean x1
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"C:\Users\Julia\Documents\NSS_lab\документы\2027 WWW Recsys\process_docs"

base = pd.read_csv(os.path.join(HERE, "full_hyperbolicity_table_frozenproj.csv"))
scaling = pd.read_csv(os.path.join(HERE, "full_hyperbolicity_table_scaling.csv"))

SCALE_COLORS = {"300u/800i": "tab:green", "1000u/1200i": "tab:blue", "3000u/2500i": "tab:orange"}
SCALE_ORDER = ["300u/800i", "1000u/1200i", "3000u/2500i"]
TYPE_MARKERS = {"GINCF": "o", "Poincare": "^", "Euclidean": "s"}

# --- Assemble one consistent long-form table: scale, type, delta_rel, hr10 ---
records = []
for _, row in base.iterrows():
    g = row["geometry"]
    if g.startswith("GINCF seed") and "restored-best" in g:
        records.append({"scale": "300u/800i", "type": "GINCF", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
    elif g.startswith("Poincare"):
        records.append({"scale": "300u/800i", "type": "Poincare", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
    elif g.startswith("Euclidean"):
        records.append({"scale": "300u/800i", "type": "Euclidean", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})

for _, row in scaling.iterrows():
    g = row["geometry"]
    for scale_label in ["1000u/1200i", "3000u/2500i"]:
        if scale_label not in g:
            continue
        if g.startswith("GINCF"):
            records.append({"scale": scale_label, "type": "GINCF", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
        elif g.startswith("Poincare"):
            records.append({"scale": scale_label, "type": "Poincare", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
        elif g.startswith("Euclidean"):
            records.append({"scale": scale_label, "type": "Euclidean", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})

df = pd.DataFrame(records)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Left: hyperbolicity (delta_rel) vs downstream quality (HR@10) ---
ax = axes[0]
for scale_label in SCALE_ORDER:
    for geom_type in ["GINCF", "Poincare", "Euclidean"]:
        sub = df[(df["scale"] == scale_label) & (df["type"] == geom_type)]
        if sub.empty:
            continue
        ax.scatter(sub["delta_rel"], sub["hr10"], s=170, marker=TYPE_MARKERS[geom_type],
                   color=SCALE_COLORS[scale_label], edgecolor="black", linewidth=1.2, zorder=3)

# Legend: color = scale, marker shape = geometry type
scale_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                             markersize=11, label=s) for s, c in SCALE_COLORS.items()]
type_handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="gray", markeredgecolor="black",
                            markersize=11, label=t) for t, m in TYPE_MARKERS.items()]
leg1 = ax.legend(handles=scale_handles, title="scale (color)", loc="upper left")
ax.add_artist(leg1)
ax.legend(handles=type_handles, title="geometry (marker)", loc="lower right")

ax.set_xlabel("delta_rel (Gromov hyperbolicity, normalized — LOWER = more tree-like/hyperbolic)")
ax.set_ylabel("test HR@10")
ax.set_title("Hyperbolicity vs downstream quality, all 3 geometries x all 3 scales\nstill no clean hyperbolicity relationship, at any scale")
ax.grid(alpha=0.3)

# --- Right: HR@10 vs scale, all three geometries under equal conditions ---
ax = axes[1]
x_pos = {s: i for i, s in enumerate(SCALE_ORDER)}

for geom_type, marker in TYPE_MARKERS.items():
    means, xs_line = [], []
    for scale_label in SCALE_ORDER:
        sub = df[(df["scale"] == scale_label) & (df["type"] == geom_type)]
        if sub.empty:
            continue
        xs = np.full(len(sub), x_pos[scale_label]) + (np.random.RandomState(0).uniform(-0.05, 0.05, len(sub))
                                                        if len(sub) > 1 else 0)
        ax.scatter(xs, sub["hr10"], s=140, marker=marker, color=SCALE_COLORS[scale_label],
                   edgecolor="black", linewidth=1.2, zorder=3)
        means.append(sub["hr10"].mean())
        xs_line.append(x_pos[scale_label])
    ax.plot(xs_line, means, linestyle="--", color="gray", linewidth=1.5, zorder=2,
            marker=marker, markersize=9, markerfacecolor="none", markeredgecolor="black",
            label=geom_type)

ax.set_xticks(list(x_pos.values()))
ax.set_xticklabels(list(x_pos.keys()))
ax.set_xlabel("scale")
ax.set_ylabel("test HR@10")
ax.set_title("Quality vs scale, all 3 geometries under equal conditions\n(dashed line = mean per scale; GINCF's edge over Euclidean narrows as scale grows)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("Amazon Beauty: GINCF vs Poincare vs Euclidean baseline, across scale\n"
             "(eta=0.001, frozen projection, dropout=0.2 for all three) — same comparison, equal conditions, at every scale",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT_DIR, "hyperbolicity_scaling_comparison.png")
fig.savefig(out_path, dpi=130)
print(f"[Save] {out_path}")
