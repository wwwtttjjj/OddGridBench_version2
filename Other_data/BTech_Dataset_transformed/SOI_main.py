import random
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from merge_all_data import merge_soi_datasets

# ======================
# 参数范围（SOI 语义不变）
# ======================
MIN_SET_SIZE = 8
MAX_SET_SIZE = 15

MIN_CELL_MAX_SIDE = 400
MAX_CELL_MAX_SIDE = 500

MIN_ODD = 0
MAX_ODD = 2

MIN_CELL_PADDING = 0
MAX_CELL_PADDING = 0

BG_COLOR = (255, 255, 255)
MAX_CANVAS_SIZE = 2048


# ======================
# Resize utils
# ======================
def resize_image_max_side(pil_img, max_side):
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / max(w, h)
    return pil_img.resize(
        (int(round(w * scale)), int(round(h * scale))),
        Image.BILINEAR,
    )


def resize_longest_side(pil_img, max_size):
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img, 1.0
    scale = max_size / max(w, h)
    return pil_img.resize(
        (int(round(w * scale)), int(round(h * scale))),
        Image.BILINEAR,
    ), scale


# ======================
# Load pool
# ======================
def load_img_pool(img_dir: Path):
    imgs = sorted(img_dir.glob("*.bmp")) + sorted(img_dir.glob("*.png"))
    if not imgs:
        raise RuntimeError(f"No JPG images in {img_dir}")
    return imgs


# ======================
# Generate single SOI sample
# ======================
def generate_single_soi(
    normal_pool,
    anomaly_pool,
    normal_ptr_ref,
    pool_lock,
    stop_flag,
):
    # ---- 全局停机检查 ----
    if stop_flag["stop"]:
        return None, None

    if len(normal_pool) < 1:
        raise RuntimeError("normal pool is empty")

    # ---------- 所有 pool 操作在锁内 ----------
    with pool_lock:
        if stop_flag["stop"]:
            return None, None

        num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)

        # odd_k：0 / 2 少，1 多
        odd_k = random.choices([0, 1, 2], weights=[2, 6, 2])[0]
        odd_k = min(odd_k, num_images)

        # ❗ anomaly 不够 → 全局停机
        if len(anomaly_pool) < odd_k:
            stop_flag["stop"] = True
            return None, None

        odd_indices_0 = set(random.sample(range(num_images), odd_k))

        # anomaly：一次性
        odd_paths = [anomaly_pool.pop() for _ in range(odd_k)]

        # normal：尽量不复用
        need_normal = num_images - odd_k
        normal_paths = []
        for _ in range(need_normal):
            if normal_ptr_ref[0] >= len(normal_pool):
                random.shuffle(normal_pool)
                normal_ptr_ref[0] = 0
            normal_paths.append(normal_pool[normal_ptr_ref[0]])
            normal_ptr_ref[0] += 1

    # ---------- 图像处理（锁外） ----------
    img_max_side = random.randint(MIN_CELL_MAX_SIDE, MAX_CELL_MAX_SIDE)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)

    images = []
    n_ptr, a_ptr = 0, 0

    for idx in range(num_images):
        if idx in odd_indices_0:
            p = odd_paths[a_ptr]
            a_ptr += 1
        else:
            p = normal_paths[n_ptr]
            n_ptr += 1

        img = Image.open(p).convert("RGB")
        img = resize_image_max_side(img, img_max_side)

        w, h = img.size
        padded = Image.new(
            "RGB",
            (w + 2 * cell_padding, h + 2 * cell_padding),
            BG_COLOR,
        )
        padded.paste(img, (cell_padding, cell_padding))
        images.append(padded)

    odd_indices = sorted([i + 1 for i in odd_indices_0])

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
# Generate SOI dataset（多线程，正确停机）
# ======================
def generate_soi_dataset(
    data_root: str,
    out_dir: str,
    samples: int = 1000,
    seed: int = 0,
    num_threads: int = 8,
):
    random.seed(seed)
    data_name = data_root.split("/")[-1]

    data_root = Path(data_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_pool = load_img_pool(data_root / "Normal")
    anomaly_pool = load_img_pool(data_root / "Anomaly")

    random.shuffle(normal_pool)
    random.shuffle(anomaly_pool)

    normal_ptr_ref = [0]
    pool_lock = Lock()
    stop_flag = {"stop": False}

    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    all_annotations = []

    def worker(i):
        # 提前停机（防止已提交任务继续写文件）
        if stop_flag["stop"]:
            return None

        imgs, meta = generate_single_soi(
            normal_pool=normal_pool,
            anomaly_pool=anomaly_pool,
            normal_ptr_ref=normal_ptr_ref,
            pool_lock=pool_lock,
            stop_flag=stop_flag,
        )

        if imgs is None:
            return None

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
        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(worker, i)
            for i in range(1, samples + 1)
        ]

        for f in as_completed(futures):
            res = f.result()
            if res is None:
                if stop_flag["stop"]:
                    print("[INFO] Anomaly images exhausted, stop generation.")
                    break
                continue

            all_annotations.append(res)

            if len(all_annotations) % 100 == 0:
                print(f"[{len(all_annotations)}/{samples}] generated")

    with (out_dir / "soi_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)


import os
# ======================
# CLI（保持不变）
# ======================
if __name__ == "__main__":
    DATA_ROOT = "manual_images/"
        # 清空输出目录
    save_root = Path("A_soi_type_data/")
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    SAMPLES = 1000
    SEED = random.randint(0, 10000)
    THREADS = 8

    subdirs = [
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]

    print(f"[INFO] Found {len(subdirs)} sub-datasets under {DATA_ROOT}")

    for data_root in sorted(subdirs):
        print(f"[INFO] Generating dataset from: {data_root}")
        OUT_DIR = save_root / f"soi_data_{os.path.basename(data_root)}"
        generate_soi_dataset(
            data_root=data_root,
            out_dir=OUT_DIR,
            samples=SAMPLES,
            seed=SEED,
            num_threads=THREADS,
        )

    SRC_ROOT = Path("A_soi_type_data")
    DST_ROOT = Path("soi_test_data")

    merge_soi_datasets(
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
    )
