import random
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import argparse


# ======================
# 参数范围
# ======================
MIN_SET_SIZE = 12
MAX_SET_SIZE = 20

MIN_CELL_SIZE = 112
MAX_CELL_SIZE = 112

MIN_ODD = 1
MAX_ODD = 4


# ======================
# Noise
# ======================
def add_gaussian_noise_pil(pil_img, sigma=0.02):
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img + noise, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return Image.fromarray(out)


# ======================
# Load pool
# ======================
def load_img_pool(dir_path: Path):
    imgs = sorted(list(dir_path.glob("*.png")))
    if not imgs:
        raise RuntimeError(f"No png in {dir_path}")
    return imgs


# ======================
# Generate single SOI sample (normal vs anomaly)
# ======================
def generate_single_soi(normal_pool, anomaly_pool):
    if len(normal_pool) < 1:
        raise RuntimeError("normal pool is empty")
    if len(anomaly_pool) < 1:
        raise RuntimeError("anomaly pool is empty")

    num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)
    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_images))

    # 0-based indices internally, convert to 1-based at the end
    all_indices = list(range(num_images))
    odd_indices_0 = set(random.sample(all_indices, odd_k))

    # 为 normal 位置随机抽图（允许重复）
    normal_k = num_images - odd_k
    normal_paths = random.choices(normal_pool, k=normal_k)

    # 为 anomaly 位置抽图（尽量不重复；不够就允许重复）
    if len(anomaly_pool) >= odd_k:
        odd_paths = random.sample(anomaly_pool, odd_k)
    else:
        odd_paths = random.choices(anomaly_pool, k=odd_k)

    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    noise_sigma = random.uniform(0.03, 0.05)

    images = []
    odd_ptr = 0
    normal_ptr = 0

    for idx in range(num_images):
        if idx in odd_indices_0:
            p = odd_paths[odd_ptr]
            odd_ptr += 1
        else:
            p = normal_paths[normal_ptr]
            normal_ptr += 1

        img = Image.open(p).convert("RGB").resize((cell_size, cell_size), Image.BILINEAR)
        img = add_gaussian_noise_pil(img, sigma=noise_sigma)
        images.append(img)

    odd_indices = sorted([i + 1 for i in odd_indices_0])  # 1-based

    meta = {
        "id": str(uuid.uuid4()),
        "class": "normal-vs-anomaly",
        "total_icons": num_images,
        "odd_icons": [],
        "num_odds": odd_k,
        "odd_indices": odd_indices,            # 1-based
        "base_image": None,                    # 现在不再是一张 base
        "normal_images": [p.name for p in normal_paths],
        "odd_images": [p.name for p in odd_paths],
        "block_size": cell_size,
    }

    return images, meta


# ======================
# Generate SOI dataset
# ======================
def generate_soi_dataset(data_root: str, out_dir: str, samples: int = 1000, seed: int = 0):
    random.seed(seed)

    data_root = Path(data_root)
    out_dir = Path(out_dir)

    # 清空输出目录
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    normal_pool = load_img_pool(data_root / "normal")
    anomaly_pool = load_img_pool(data_root / "anomaly")

    all_annotations = []

    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    for i in range(1, samples + 1):
        imgs, meta = generate_single_soi(normal_pool, anomaly_pool)

        sample_folder = images_root / f"image_{i}"
        sample_folder.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(imgs):
            img.save(sample_folder / f"{idx + 1}.png")

        meta["image"] = f"image_{i}"

        # # 每个样本单独保存 annotation.json（不会覆盖）
        # with (sample_folder / "annotation.json").open("w", encoding="utf-8") as f:
        #     json.dump(meta, f, ensure_ascii=False, indent=2)

        all_annotations.append(meta)

        if i % 100 == 0:
            print(f"[{i}/{samples}] generated")

    with (out_dir / "soi_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)


# ======================
# CLI
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./ELPV_split/train",
        help="输入根目录，必须包含 normal/ 和 anomaly/ 两个子目录",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="样本数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（不传则随机）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./soi_test_data",
        help="输出目录（不传则在 data_root 同级创建 soi_test_data）",
    )

    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 10000)

    out_dir = args.out_dir

    generate_soi_dataset(
        data_root=args.data_root,
        out_dir=out_dir,
        samples=args.samples,
        seed=seed,
    )
