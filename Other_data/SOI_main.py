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

MIN_SET_SIZE = 10
MAX_SET_SIZE = 20

MIN_CELL_SIZE = 60
MAX_CELL_SIZE = 80

MIN_ODD = 1
MAX_ODD = 5

BG_COLOR = (255, 255, 255)


# ======================
# Noise
# ======================

def add_gaussian_noise_pil(pil_img, sigma=0.02):
    """
    pil_img: PIL.Image (RGB), uint8 [0,255]
    sigma: noise std in [0,1]
    """
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img + noise
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return Image.fromarray(out)


# ======================
# Load digit pool
# ======================

def load_digit_pool(png_root: Path):
    """
    png_root/
      0/*.png
      1/*.png
      ...
      9/*.png
    """
    pool = {}
    for d in range(10):
        ddir = png_root / str(d)
        imgs = sorted(ddir.glob("*.png"))
        if not imgs:
            raise RuntimeError(f"No png files in {ddir}")
        pool[d] = imgs
    return pool


# ======================
# Generate single SOI sample
# ======================


def generate_single_soi(digit_pool: dict):
    digit = random.choice(list(digit_pool.keys()))
    paths = digit_pool[digit]

    if len(paths) < 2:
        raise RuntimeError(f"Digit {digit} must have >= 2 images")

    num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)
    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_images))

    base_path = random.choice(paths)

    # ⚠️ 注意：内部仍然用 0-based，最后统一 +1
    all_indices = list(range(num_images))
    odd_indices_0 = set(random.sample(all_indices, odd_k))

    candidates = [p for p in paths if p != base_path]
    if len(candidates) >= odd_k:
        odd_paths = random.sample(candidates, odd_k)
    else:
        odd_paths = [random.choice(candidates) for _ in range(odd_k)]

    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    noise_sigma = random.uniform(0.03, 0.05)

    images = []
    odd_ptr = 0

    for idx in range(num_images):
        if idx in odd_indices_0:
            p = odd_paths[odd_ptr]
            odd_ptr += 1
        else:
            p = base_path

        img = Image.open(p).convert("RGB") \
            .resize((cell_size, cell_size), Image.BILINEAR)

        img = add_gaussian_noise_pil(img, sigma=noise_sigma)
        images.append(img)

    # ✅ odd index 改成 1-based
    odd_indices = sorted([i + 1 for i in odd_indices_0])

    meta = {
        "id": str(uuid.uuid4()),
        "class": digit,
        "total_icons": num_images,
        "odd_icons":[],
        "num_odds": odd_k,
        "odd_indices": odd_indices,   # ← 1-based
        "base_image": base_path.name,
        "odd_images": [p.name for p in odd_paths],
        "block_size":cell_size,
    }

    return images, meta


# ======================
# Generate SOI dataset
# ======================

def generate_soi_dataset(
    png_root: str,
    out_dir: str,
    samples: int = 1000,
    seed: int = 0
):
    random.seed(seed)

    png_root = Path(png_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    digit_pool = load_digit_pool(png_root)

    all_annotations = []

    for i in range(1, samples + 1):
        imgs, meta = generate_single_soi(digit_pool)

        sample_dir = out_dir / f"images"
        img_dir = sample_dir / f"image_{i}"
        img_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(imgs):
            img.save(img_dir / f"{idx + 1}.png")

        meta["image"] = f"images{i}"

        with (sample_dir / "annotation.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        all_annotations.append(meta)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{samples}] generated")

    with (out_dir / "soi_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)


# ======================
# CLI
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png_root",
        type=str,
        default="./mnist/mnist_png",
        help="输入数据根目录（0-9 子目录）",
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

    args = parser.parse_args()

    out_dir = "/".join(args.png_root.split("/")[:-1]) + "/soi_test_data"
    seed = args.seed if args.seed is not None else random.randint(0, 10000)

    generate_soi_dataset(
        png_root=args.png_root,
        out_dir=out_dir,
        samples=args.samples,
        seed=seed,
    )
