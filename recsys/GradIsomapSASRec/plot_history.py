"""
Построение графиков по history.json и cf_history.json.

Использование:
    python plot_history.py --history path/to/history.json \
                           --cf_history path/to/cf_history.json \
                           --out_dir plots

Можно передать только один из --history/--cf_history, или оба.
Если не передан ни один — скрипт просто ничего не делает.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")          # без GUI, чтобы работало и на сервере
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────
#  Утилиты
# ──────────────────────────────────────────────────────────────────

def _load_json(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        print(f"[WARN] файл не найден, пропускаю: {path}", flush=True)
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] загружен {path}", flush=True)
    return data


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _safe_xy(series, x=None):
    """Возвращает (x, y) для графика, отбрасывая None/NaN."""
    y = np.asarray(series, dtype=float)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)
    mask = np.isfinite(y)
    return x[mask], y[mask]


# ──────────────────────────────────────────────────────────────────
#  history.json — один рисунок с двумя панелями
# ──────────────────────────────────────────────────────────────────

def plot_history(history, out_dir, prefix="history"):
    """
    Рисует один PNG с двумя subplot-ами:
      - слева:  train_loss vs val_loss
      - справа: метрики (val_hr, val_ndcg, и train_hr если есть)

    Опционально возвращает путь к сохранённому файлу.
    """
    if history is None:
        return None

    _ensure_dir(out_dir)
    epochs = history.get("epoch")
    if epochs is None:
        # если epoch нет — используем индексы
        any_key = next(
            (k for k in ("train_loss", "val_loss", "val_hr") if k in history),
            None,
        )
        n = len(history[any_key]) if any_key else 0
        epochs = list(range(n))

    has_train_loss = "train_loss" in history and len(history["train_loss"]) > 0
    has_val_loss = "val_loss" in history and len(history["val_loss"]) > 0

    metric_keys = [k for k in ("val_hr", "val_ndcg", "train_hr", "train_ndcg")
                   if k in history and len(history[k]) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── панель 1: лоссы ──
    ax = axes[0]
    if has_train_loss:
        x, y = _safe_xy(history["train_loss"], epochs)
        ax.plot(x, y, label="train_loss", color="tab:blue")
    if has_val_loss:
        x, y = _safe_xy(history["val_loss"], epochs)
        ax.plot(x, y, label="val_loss", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    if has_train_loss or has_val_loss:
        ax.legend()

    # ── панель 2: метрики ──
    ax = axes[1]
    for k in metric_keys:
        x, y = _safe_xy(history[k], epochs)
        style = "--" if k.startswith("train") else "-"
        ax.plot(x, y, style, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Metrics")
    ax.grid(alpha=0.3)
    if metric_keys:
        ax.legend()

    fig.suptitle(f"{prefix}", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Save] {out_path}", flush=True)
    return out_path


# ──────────────────────────────────────────────────────────────────
#  cf_history.json — по одному рисунку на каждую внешнюю эпоху
# ──────────────────────────────────────────────────────────────────

def _plot_one_outer_epoch(
    outer_epoch: int,
    train_loss_inner, val_loss_inner, hr_inner,
    out_dir: str,
    prefix: str,
):
    """
    Строит два PNG для одной внешней эпохи:
      <prefix>_epoch{NN}_loss.png    — train_loss и val_loss по inner-эпохам
      <prefix>_epoch{NN}_metrics.png — val_hr (и val_ndcg если есть)
    """
    _ensure_dir(out_dir)

    # ── loss ──
    fig, ax = plt.subplots(figsize=(7, 5))
    if train_loss_inner is not None and len(train_loss_inner) > 0:
        x, y = _safe_xy(train_loss_inner)
        ax.plot(x, y, label="train_loss", color="tab:blue")
    if val_loss_inner is not None and len(val_loss_inner) > 0:
        x, y = _safe_xy(val_loss_inner)
        ax.plot(x, y, label="val_loss", color="tab:orange")
    ax.set_xlabel("inner epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Outer epoch {outer_epoch} — loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p_loss = os.path.join(out_dir, f"{prefix}_epoch{outer_epoch:02d}_loss.png")
    fig.savefig(p_loss, dpi=150)
    plt.close(fig)

    # ── metrics ──
    fig, ax = plt.subplots(figsize=(7, 5))
    has_any = False
    if hr_inner is not None and len(hr_inner) > 0:
        x, y = _safe_xy(hr_inner)
        ax.plot(x, y, label="val_hr", color="tab:green")
        has_any = True
    ax.set_xlabel("inner epoch")
    ax.set_ylabel("metric")
    ax.set_title(f"Outer epoch {outer_epoch} — metrics")
    ax.grid(alpha=0.3)
    if has_any:
        ax.legend()
    fig.tight_layout()
    p_metrics = os.path.join(out_dir, f"{prefix}_epoch{outer_epoch:02d}_metrics.png")
    fig.savefig(p_metrics, dpi=150)
    plt.close(fig)

    return p_loss, p_metrics


def plot_cf_history(cf_history, out_dir, prefix="cf"):
    """
    cf_history.json: dict со списками списков:
      train_loss: [ [loss_inner_0, loss_inner_1, ...],   # outer epoch 0
                    [loss_inner_0, ...],                  # outer epoch 1
                    ... ]
      val_loss:   аналогично
      val_hr:     аналогично
      (опц.) val_ndcg

    Для каждой outer-эпохи создаёт ДВА PNG (loss и metrics) в подпапке
    out_dir/<prefix>_per_epoch/.
    """
    if cf_history is None:
        return None

    _ensure_dir(out_dir)
    per_epoch_dir = _ensure_dir(os.path.join(out_dir, f"{prefix}_per_epoch"))

    train_all = cf_history.get("train_loss", [])
    val_all = cf_history.get("val_loss", [])
    hr_all = cf_history.get("val_hr", [])
    ndcg_all = cf_history.get("val_ndcg", [])

    n_outer = max(len(train_all), len(val_all), len(hr_all), len(ndcg_all))
    if n_outer == 0:
        print("[WARN] cf_history пуст", flush=True)
        return per_epoch_dir

    saved = []
    for i in range(n_outer):
        t = train_all[i] if i < len(train_all) else None
        v = val_all[i] if i < len(val_all) else None
        h = hr_all[i] if i < len(hr_all) else None
        nd = ndcg_all[i] if i < len(ndcg_all) else None

        p_loss, p_metrics = _plot_one_outer_epoch(
            outer_epoch=i,
            train_loss_inner=t,
            val_loss_inner=v,
            hr_inner=h,
            out_dir=per_epoch_dir,
            prefix=prefix,
        )
        saved.append((p_loss, p_metrics))
        print(f"[Save] outer {i:02d}: {os.path.basename(p_loss)}, "
              f"{os.path.basename(p_metrics)}", flush=True)

        # Отдельно достроим NDCG, если он есть — второй осью к metrics
        if nd is not None and len(nd) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            x, y = _safe_xy(nd)
            ax.plot(x, y, label="val_ndcg", color="tab:red")
            ax.set_xlabel("inner epoch")
            ax.set_ylabel("ndcg")
            ax.set_title(f"Outer epoch {i} — val_ndcg")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            p_ndcg = os.path.join(
                per_epoch_dir, f"{prefix}_epoch{i:02d}_ndcg.png"
            )
            fig.savefig(p_ndcg, dpi=150)
            plt.close(fig)
            saved[-1] = saved[-1] + (p_ndcg,)

    # сводный график: val_hr по всем outer-эпохам (последняя точка каждой)
    if len(hr_all) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        final_hr = [np.nan if len(h) == 0 else float(np.nanmax(h))
                    for h in hr_all]
        x, y = _safe_xy(final_hr)
        ax.plot(x, y, marker="o", label="best val_hr per outer epoch")
        ax.set_xlabel("outer epoch")
        ax.set_ylabel("val_hr")
        ax.set_title("Best val_hr across outer epochs")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p_summary = os.path.join(out_dir, f"{prefix}_summary_val_hr.png")
        fig.savefig(p_summary, dpi=150)
        plt.close(fig)
        print(f"[Save] {p_summary}", flush=True)

    return per_epoch_dir


# ──────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default=None,
                        help="путь к history.json (может отсутствовать)")
    parser.add_argument("--cf_history", default=None,
                        help="путь к cf_history.json (может отсутствовать)")
    parser.add_argument("--out_dir", default="plots",
                        help="куда складывать PNG")
    parser.add_argument("--prefix_history", default="history")
    parser.add_argument("--prefix_cf", default="cf")
    args = parser.parse_args()

    if not args.history and not args.cf_history:
        print("[WARN] ни --history, ни --cf_history не переданы — нечего рисовать")
        return

    _ensure_dir(args.out_dir)

    h = _load_json(args.history) if args.history else None
    if h is not None:
        plot_history(h, out_dir=args.out_dir, prefix=args.prefix_history)
    else:
        print("[skip] history.json не передан/не найден", flush=True)

    cf = _load_json(args.cf_history) if args.cf_history else None
    if cf is not None:
        plot_cf_history(cf, out_dir=args.out_dir, prefix=args.prefix_cf)
    else:
        print("[skip] cf_history.json не передан/не найден", flush=True)

    print(f"\n[done] всё сохранено в {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    #main()
    logs_folder = "recsys/GradIsomapSASRec/logs_sasrec_isomap/sasrec_amazon_beauty_17"

    h = _load_json(os.path.join(logs_folder, "history.json"))
    cf = _load_json(os.path.join(logs_folder, "cf_history.json"))

    plot_history(h, out_dir=os.path.join(logs_folder, "plots"))
    #plot_cf_history(cf, out_dir=os.path.join(logs_folder, "plots"))

