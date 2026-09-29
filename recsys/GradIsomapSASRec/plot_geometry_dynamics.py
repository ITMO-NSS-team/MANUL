# recsys/my_verification/plot_geometry_dynamics.py
# -*- coding: utf-8 -*-
"""
Plots per-epoch geometry diagnostics saved by analyze_geometry.py into a CSV.

Input: geometry_results.csv with at least column 'epoch' and multiple metric columns
(e.g. orc_mean, spectral_lambda2, kruskal_stress, geo_h1_count, lat_h1_count, ...)

Output: one PNG per metric + a few combined overview figures.

Usage example (Windows):
python recsys/my_verification/plot_geometry_dynamics.py ^
  --csv "c:/.../sasrec_amazon_beauty_24/geometry_results.csv" ^
  --out_dir "c:/.../sasrec_amazon_beauty_24/plots"
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt


def _safe_fname(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def plot_one_metric(df: pd.DataFrame, metric: str, out_path: str, title: str = None):
    x = df["epoch"].to_numpy()
    y = df[metric].to_numpy()

    # If everything is NaN -> skip
    if np.all(~np.isfinite(y)):
        return False

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, y, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def plot_group(df: pd.DataFrame, metrics: list[str], out_path: str, title: str):
    metrics = [m for m in metrics if m in df.columns and np.any(np.isfinite(df[m].to_numpy()))]
    if not metrics:
        return False

    plt.figure(figsize=(10, 5.5))
    for m in metrics:
        plt.plot(df["epoch"], df[m], marker="o", linewidth=1.3, markersize=2.5, label=m)

    plt.xlabel("epoch")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to geometry_results.csv")
    p.add_argument("--out_dir", required=True, help="Directory to save plots")
    p.add_argument("--only", default=None,
                   help="Optional: comma-separated list of metrics to plot (exact column names)")
    p.add_argument("--skip_individual", action="store_true",
                   help="If set, do not save 1-plot-per-metric images (only grouped plots).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise ValueError(f"'epoch' column not found in CSV. Columns: {list(df.columns)}")

    df = df.sort_values("epoch").reset_index(drop=True)

    # Convert everything except epoch to numeric where possible
    for c in df.columns:
        if c == "epoch":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Choose metrics to plot
    if args.only:
        metrics = [m.strip() for m in args.only.split(",") if m.strip()]
    else:
        metrics = [c for c in df.columns if c != "epoch"]

    # Drop columns that are all NaN
    metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]

    print(f"[Info] epochs: {df['epoch'].min()}..{df['epoch'].max()}  (n={len(df)})")
    print(f"[Info] metrics to plot: {len(metrics)}")
    print(f"[Info] out_dir: {args.out_dir}")

    # --- Individual plots ---
    if not args.skip_individual:
        saved = 0
        for m in metrics:
            out_path = os.path.join(args.out_dir, f"{_safe_fname(m)}.png")
            ok = plot_one_metric(df, m, out_path)
            saved += int(ok)
        print(f"[Save] individual metric plots: {saved} files")

    # --- Grouped overview plots (handy defaults for your CSV schema) ---
    groups = {
        "orc": [c for c in df.columns if c.startswith("orc_")],                     # orc_mean, orc_median, orc_f_neg...
        "spectral": [c for c in df.columns if c.startswith("spectral_")],          # spectral_lambda2, spectral_spectral_gap...
        "isometry": [c for c in ["kruskal_stress", "spearman_rho", "residual_variance"] if c in df.columns],
        "ph_geo": [c for c in df.columns if c.startswith("geo_")],                 # geo_h1_count, geo_h1_max_persistence...
        "ph_lat": [c for c in df.columns if c.startswith("lat_")],                 # lat_h1_count, lat_h1_max_persistence...
    }

    for gname, gmetrics in groups.items():
        if not gmetrics:
            continue
        out_path = os.path.join(args.out_dir, f"group_{gname}.png")
        ok = plot_group(df, gmetrics, out_path, title=f"{gname}: metrics vs epoch")
        if ok:
            print(f"[Save] {out_path}")

    # Extra: common “topline” plot (if columns exist)
    topline = []
    for c in ["orc_mean", "orc_f_neg", "spectral_spectral_gap", "kruskal_stress", "spearman_rho",
              "geo_h1_count", "lat_h1_count"]:
        if c in df.columns:
            topline.append(c)
    if topline:
        out_path = os.path.join(args.out_dir, "group_topline.png")
        plot_group(df, topline, out_path, title="Topline geometry diagnostics vs epoch")
        print(f"[Save] {out_path}")

    print("[Done]")


if __name__ == "__main__":
    main()
    