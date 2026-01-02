import random
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import argparse
from merge_all_data import merge_soi_datasets

# ======================
# 参数范围（SOI 语义不变）
# ======================
MIN_SET_SIZE = 8
MAX_SET_SIZE = 15

MIN_CELL_MAX_SIDE = 400
MAX_CELL_MAX_SIDE = 400

MIN_ODD = 0
MAX_ODD = 2

MIN_CELL_PADDING = 0
MAX_CELL_PADDING = 0

BG_COLOR = (255, 255, 255)
MAX_CANVAS_SIZE = 2048   # 对齐 soi：sample-level longest side cap


# ======================
# Noise
# ======================
# def add_gaussian_noise_pil(pil_img, sigma=0.02):
#     img = np.asarray(pil_img).astype(np.float32) / 255.0
#     noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
#     out = np.clip(img + noise, 0.0, 1.0)
#     out = (out * 255.0).astype(np.uint8)
#     return Image.fromarray(out)


# ======================
# Resize utils（来自 soi）
# ======================
def resize_image_max_side(pil_img, max_side):
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.BILINEAR)


def resize_longest_side(pil_img, max_size):
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img, 1.0
    scale = max_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.BILINEAR), scale


# ======================
# Load pool
# ======================
def load_img_pool(dir_path: Path):
    imgs = sorted(list(dir_path.glob("*.JPG")))
    if not imgs:
        raise RuntimeError(f"No png images in {dir_path}")
    return imgs


# ======================
# Generate single SOI sample
# （图像处理逻辑已对齐 soi）
# ======================
def generate_single_soi(normal_pool, anomaly_pool):
    if len(normal_pool) < 1:
        raise RuntimeError("normal pool is empty")
    if len(anomaly_pool) < 1:
        raise RuntimeError("anomaly pool is empty")

    num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)
    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_images))

    all_indices = list(range(num_images))
    odd_indices_0 = set(random.sample(all_indices, odd_k))

    normal_paths = random.choices(normal_pool, k=num_images - odd_k)
    odd_paths = (
        random.sample(anomaly_pool, odd_k)
        if len(anomaly_pool) >= odd_k
        else random.choices(anomaly_pool, k=odd_k)
    )

    img_max_side = random.randint(MIN_CELL_MAX_SIDE, MAX_CELL_MAX_SIDE)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)
    noise_sigma = random.uniform(0.03, 0.05)

    images = []
    normal_ptr = 0
    odd_ptr = 0

    for idx in range(num_images):
        if idx in odd_indices_0:
            p = odd_paths[odd_ptr]
            odd_ptr += 1
        else:
            p = normal_paths[normal_ptr]
            normal_ptr += 1

        img = Image.open(p).convert("RGB")
        img = resize_image_max_side(img, img_max_side)
        # img = add_gaussian_noise_pil(img, sigma=noise_sigma)

        # ===== padding（与 soi 一致）=====
        w, h = img.size
        padded = Image.new(
            "RGB",
            (w + 2 * cell_padding, h + 2 * cell_padding),
            BG_COLOR,
        )
        padded.paste(img, (cell_padding, cell_padding))
        images.append(padded)

    odd_indices = sorted([i + 1 for i in odd_indices_0])  # 1-based

    meta = {
        "id": str(uuid.uuid4()),
        "task_type": "soi",
        "class": "normal-vs-anomaly",
        "total_icons": num_images,
        "num_odds": odd_k,
        "odd_indices": odd_indices,
        "normal_images": [p.name for p in normal_paths],
        "odd_images": [p.name for p in odd_paths],
        "cell_padding": cell_padding,
        "img_max_side": img_max_side,
    }

    return images, meta


# ======================
# Generate SOI dataset
# ======================
def generate_soi_dataset(
    data_root: str,
    out_dir: str,
    samples: int = 1000,
    seed: int = 0,
):
    random.seed(seed)
    data_name = data_root.split("/")[-1]

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    # 清空输出目录
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_pool = load_img_pool(data_root / "Normal")
    anomaly_pool = load_img_pool(data_root / "Anomaly")

    all_annotations = []

    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    for i in range(1, samples + 1):
        imgs, meta = generate_single_soi(normal_pool, anomaly_pool)

        # ===== sample-level resize（对齐 soi）=====
        resized_imgs = []
        scales = []
        for img in imgs:
            img, scale = resize_longest_side(img, MAX_CANVAS_SIZE)
            resized_imgs.append(img)
            scales.append(scale)

        meta["resize_scale"] = min(scales)

        sample_folder = images_root / f"image_{data_name}_{i}"
        sample_folder.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(resized_imgs):
            img.save(sample_folder / f"{idx + 1}.png")

        meta["image"] = f"image_{data_name}_{i}"
        all_annotations.append(meta)

        if i % 100 == 0:
            print(f"[{i}/{samples}] generated")

    with (out_dir / "soi_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)

import os
# ======================
# CLI
# ======================
if __name__ == "__main__":
    # DATA_ROOT = "A_cropped_images/"
    #     # 清空输出目录
    # save_root = Path("A_soi_type_data/")
    # if save_root.exists():
    #     shutil.rmtree(save_root)
    # save_root.mkdir(parents=True, exist_ok=True)

    # SAMPLES = 10
    # SEED = random.randint(0, 10000)
    # THREADS = 8

    # # ===== 枚举 DATA_ROOT 下的所有子目录 =====
    # subdirs = [
    #     os.path.join(DATA_ROOT, d)
    #     for d in os.listdir(DATA_ROOT)
    #     if os.path.isdir(os.path.join(DATA_ROOT, d))
    # ]

    # print(f"[INFO] Found {len(subdirs)} sub-datasets under {DATA_ROOT}")

    # for data_root in sorted(subdirs):
    #     print(f"[INFO] Generating dataset from: {data_root}")
    #     OUT_DIR = save_root / f"soi_data_{os.path.basename(data_root)}"
    #     generate_soi_dataset(
    #         data_root=data_root,
    #         out_dir=OUT_DIR,
    #         samples=SAMPLES,
    #         seed=SEED,
    #     )
        
    SRC_ROOT = Path("A_soi_type_data")
    DST_ROOT = Path("soi_test_data")

    merge_soi_datasets(
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
    )