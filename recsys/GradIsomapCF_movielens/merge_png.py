import os
import math
from PIL import Image
import re


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


if __name__ == "__main__":
    target_folder = "logs_movielens_isomap_cf/run35/images"
    create_collage(target_folder, "logs_movielens_isomap_cf/run35/images/final_report_collage.png", n_cols=4)
