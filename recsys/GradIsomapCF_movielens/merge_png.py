import os
import math
from PIL import Image
import re
import matplotlib.pyplot as plt
import numpy as np


def create_collage(images_dir, output_path, run_name="ginmf", n_cols=3):

    pattern = rf"^{re.escape(run_name)}_inner_neumf_losses_outer(\d+)\.png$"
    image_files = []
    for f in os.listdir(images_dir):
        m = re.match(pattern, f)
        if m:
            outer_idx = int(m.group(1))
            image_files.append((outer_idx, f))

    if not image_files:
        print("В папке нет PNG-файлов формата "
              f'"{run_name}_inner_neumf_losses_outer<idx>.png"')
        return

    image_files.sort(key=lambda x: x[0])
    image_files = [fname for (_, fname) in image_files]

    n_images = len(image_files)
    n_rows = math.ceil(n_images / n_cols)

    imgs = [Image.open(os.path.join(images_dir, f)) for f in image_files]

    base_w, base_h = imgs[0].size
    resized_imgs = [img.resize((base_w, base_h), Image.LANCZOS)
                    if img.size != (base_w, base_h) else img
                    for img in imgs]

    collage_w = n_cols * base_w
    collage_h = n_rows * base_h
    collage = Image.new("RGB", (collage_w, collage_h), color="white")

    for idx, img in enumerate(resized_imgs):
        row = idx // n_cols
        col = idx % n_cols
        x = col * base_w
        y = row * base_h
        collage.paste(img, (x, y))

    collage.save(output_path)
    print(f"Коллаж сохранён в: {output_path}")


def plot_cf_losses(cf_history, images_dir, run_name="run", poly_degree=2):
    """
    Рисует графики train/val BCE loss по cf_history и добавляет 3-ю линию:
    полиномиальную аппроксимацию val loss.

    Ожидаемый формат cf_history:
      {
        "train_loss": [[...], [...], ...]  # inner-эпохи для каждого outer
        "val_loss":   [[...], [...], ...]
      }
    Также поддерживает "плоский" формат:
      {"train_loss":[...], "val_loss":[...]} -> будет считаться одним outer.
    """
    os.makedirs(images_dir, exist_ok=True)

    cf_train = cf_history.get("train_loss", [])
    cf_val   = cf_history.get("val_loss", [])

    # поддержка случая, когда не список списков, а один список
    is_nested = len(cf_train) > 0 and isinstance(cf_train[0], (list, tuple, np.ndarray))
    if not is_nested:
        cf_train = [cf_train]
        cf_val   = [cf_val]

    num_outer = len(cf_train)

    for outer_idx in range(num_outer):
        inner_train = cf_train[outer_idx] if outer_idx < len(cf_train) else None
        inner_val   = cf_val[outer_idx]   if outer_idx < len(cf_val)   else None

        if inner_train is None or len(inner_train) == 0:
            continue

        x = np.arange(len(inner_train))

        plt.figure(figsize=(8, 5))
        plt.plot(x, inner_train, marker='o', label='Train BCE loss')

        # val
        if inner_val is not None and len(inner_val) > 0:
            y_val = np.array([v if v is not None else np.nan for v in inner_val], dtype=float)
            plt.plot(x[:len(y_val)], y_val, marker='s', label='Val BCE loss')

            # аппроксимация val (полином)
            mask = ~np.isnan(y_val)
            if mask.sum() >= max(2, poly_degree + 1):
                xi = x[:len(y_val)][mask]
                yi = y_val[mask]
                deg = min(poly_degree, len(xi) - 1)  # чтобы polyfit не упал
                coeffs = np.polyfit(xi, yi, deg=deg)
                p = np.poly1d(coeffs)
                y_hat = p(x[:len(y_val)])
                plt.plot(x[:len(y_val)], y_hat, linewidth=2.0,
                         label=f'Val approx (poly deg {deg})')

        plt.xlabel('Inner epoch (cf_ep)')
        plt.ylabel('BCE loss')
        plt.title(f'CF train/val loss (outer epoch {outer_idx + 1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(images_dir, f"new_images/{run_name}_cf_losses_outer{outer_idx}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    target_folder = "logs_amazon_books_isomap_cf/717/images"
    create_collage(target_folder, "logs_amazon_books_isomap_cf/717/images/final_report_collage.png", n_cols=5)
