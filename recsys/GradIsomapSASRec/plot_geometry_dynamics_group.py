# recsys/my_verification/plot_geometry_dynamics_multi_csv.py
# -*- coding: utf-8 -*-
"""
Plot per-metric dynamics from MULTIPLE geometry_results.csv files on the same figure.

Changes vs previous version:
- NO grouped plots.
- Accept multiple --csv arguments.
- For each metric, draw multiple lines (one per CSV/run).
- Save one PNG per metric.

CSV contract:
- Must contain column: epoch
- Other columns: metrics (orc_*, spectral_*, kruskal_stress, spearman_rho, geo_*, lat_* ...)

Usage:
python plot_geometry_dynamics_multi_csv.py ^
  --csv "run_best=C:/.../run_best/geometry_results.csv" ^
  --csv "run_last=C:/.../run_last/geometry_results.csv" ^
  --out_dir "C:/.../plots_compare" ^
  --only "orc_mean,orc_f_neg,spectral_spectral_gap,lat_h1_count"

If label is not provided (no "label=" prefix), label is derived from filename.
"""
import argparse
import os
import re
from typing import Dict, Tuple, Optional

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


def _parse_csv_arg(s: str) -> Tuple[str, str]:
    """
    Accept:
      - "label=/path/to/file.csv"
      - "/path/to/file.csv"  (label inferred)
    """
    if "=" in s:
        label, path = s.split("=", 1)
        label = label.strip()
        path = path.strip().strip('"').strip("'")
        if not label:
            raise ValueError(f"Empty label in --csv {s!r}")
        return label, path
    else:
        path = s.strip().strip('"').strip("'")
        base = os.path.splitext(os.path.basename(path))[0]
        label = base
        return label, path


def _load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"'epoch' column not found in CSV: {path}\nColumns={list(df.columns)}")

    df = df.sort_values("epoch").reset_index(drop=True)

    # numeric conversion
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    for c in df.columns:
        if c == "epoch":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with NaN epoch
    df = df[np.isfinite(df["epoch"].to_numpy())].copy()
    df["epoch"] = df["epoch"].astype(int)

    return df


def _collect_metrics(dfs: Dict[str, pd.DataFrame], only: Optional[str]) -> list[str]:
    if only:
        metrics = [m.strip() for m in only.split(",") if m.strip()]
        return metrics

    # union of all metric columns (excluding epoch)
    all_cols = set()
    for df in dfs.values():
        all_cols.update([c for c in df.columns if c != "epoch"])
    return sorted(all_cols)


def plot_metric_multi(
    dfs: Dict[str, pd.DataFrame],
    metric: str,
    out_path: str,
    title: Optional[str] = None,
    marker: str = "o",
):
    """
    One plot per metric; multiple lines (one per label/df).
    Skips a run if that metric column is absent or all-NaN.
    """
    plt.figure(figsize=(9.5, 5.2))
    any_line = False

    for label, df in dfs.items():
        if metric not in df.columns:
            continue
        x = df["epoch"].to_numpy()
        y = df[metric].to_numpy()

        # If all NaN -> skip this line
        if np.all(~np.isfinite(y)):
            continue

        # Plot raw (with NaNs allowed: matplotlib will break the line)
        plt.plot(x, y, marker=marker, linewidth=1.6, markersize=3, label=label)
        any_line = True

    if not any_line:
        plt.close()
        return False

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        action="append",
        required=True,
        help='Repeatable. Either "label=PATH" or "PATH". Example: --csv "eta001=C:/.../geometry_results.csv"',
    )
    p.add_argument("--out_dir", required=True, help="Directory to save plots")
    p.add_argument(
        "--only",
        default=None,
        help="Optional: comma-separated list of metrics to plot (exact column names). "
             "If omitted, plots the union of all columns from all CSVs (excluding epoch).",
    )
    p.add_argument("--marker", default="o", help="Matplotlib marker for points (default: o). Use '' for no markers.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all CSVs
    dfs: Dict[str, pd.DataFrame] = {}
    for s in args.csv:
        label, path = _parse_csv_arg(s)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if label in dfs:
            raise ValueError(f"Duplicate label {label!r}. Please provide unique labels in --csv arguments.")
        dfs[label] = _load_one_csv(path)

    # Determine metrics to plot
    metrics = _collect_metrics(dfs, args.only)

    # Filter out metrics that are absent OR all-NaN in all dfs
    filtered = []
    for m in metrics:
        ok_any = False
        for df in dfs.values():
            if m in df.columns and not df[m].isna().all():
                ok_any = True
                break
        if ok_any:
            filtered.append(m)
    metrics = filtered

    # Print info
    print(f"[Info] loaded CSVs: {len(dfs)}")
    for label, df in dfs.items():
        print(f"  - {label}: epochs {df['epoch'].min()}..{df['epoch'].max()} (n={len(df)})")
    print(f"[Info] metrics to plot: {len(metrics)}")
    print(f"[Info] out_dir: {args.out_dir}")

    saved = 0
    for m in metrics:
        out_path = os.path.join(args.out_dir, f"{_safe_fname(m)}.png")
        ok = plot_metric_multi(dfs, m, out_path, marker=args.marker)
        saved += int(ok)

    print(f"[Save] plots: {saved} files")
    print("[Done]")


if __name__ == "__main__":
    main()
    