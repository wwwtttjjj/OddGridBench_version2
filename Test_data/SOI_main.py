import random
import json
import shutil
import uuid
from pathlib import Path
import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm  # 建议安装：pip install tqdm

# 尝试从 configs 导入，如果不存在则使用默认值
try:
    from configs import odd_nums, odd_pro
except ImportError:
    odd_nums = [0, 1, 2, 3, 4]
    odd_pro = [0.1, 0.5, 0.2, 0.1,0.1]

# ======================
# 参数范围
# ======================
MIN_SET_SIZE = 12
MAX_SET_SIZE = 16

MIN_CELL_SIZE = 100
MAX_CELL_SIZE = 150
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
    pool = {}
    if not png_root.exists():
        raise RuntimeError(f"Path not found: {png_root}")
        
    subdirs = sorted(p for p in png_root.iterdir() if p.is_dir())
    if not subdirs:
        raise RuntimeError(f"No subdirectories (0-9) found in {png_root}")

    for ddir in subdirs:
        imgs = sorted(list(ddir.glob("*.png")))
        if not imgs:
            print(f"[Warning] No png files in {ddir}")
            continue
        pool[ddir.name] = imgs
    return pool

# ======================
# Generate single SOI sample
# ======================
def generate_single_soi(digit_pool: dict):
    # 随机选择一个数字类别
    digit = random.choice(list(digit_pool.keys()))
    paths = digit_pool[digit]

    if len(paths) < 2:
        raise RuntimeError(f"Digit {digit} must have >= 2 images for odd-one-out")

    num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)
    odd_k = random.choices(odd_nums, weights=odd_pro)[0]
    odd_k = min(odd_k, num_images - 1) # 确保至少留一个 base

    # 选取基准图（Normal）
    base_path = random.choice(paths)

    # 选取异类图（Anomaly）
    candidates = [p for p in paths if p != base_path]
    if not candidates:
        odd_paths = [base_path] * odd_k
    elif len(candidates) >= odd_k:
        odd_paths = random.sample(candidates, odd_k)
    else:
        odd_paths = [random.choice(candidates) for _ in range(odd_k)]

    # 确定异类的位置 (1-based index)
    odd_indices_0 = set(random.sample(range(num_images), odd_k))
    odd_indices = sorted([i + 1 for i in odd_indices_0])

    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    noise_sigma = random.uniform(0.03, 0.05)

    images = []
    odd_ptr = 0
    for idx in range(num_images):
        p = odd_paths[odd_ptr] if idx in odd_indices_0 else base_path
        if idx in odd_indices_0: odd_ptr += 1

        img = Image.open(p).convert("RGB").resize((cell_size, cell_size), Image.BILINEAR)
        img = add_gaussian_noise_pil(img, sigma=noise_sigma)
        images.append(img)

    meta = {
        "id": str(uuid.uuid4()),
        "class": digit,
        "total_icons": num_images,
        "num_odds": odd_k,
        "odd_indices": odd_indices,
        "base_image": base_path.name,
        "odd_images": [p.name for p in odd_paths],
        "block_size": cell_size,
    }
    return images, meta

# ======================
# Generate SOI dataset
# ======================
def generate_soi_dataset(png_root: str, out_dir: str, samples: int = 1000, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    png_root = Path(png_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    images_base_dir = out_dir / "images"
    images_base_dir.mkdir(parents=True, exist_ok=True)

    digit_pool = load_digit_pool(png_root)
    all_annotations = []

    # 使用 tqdm 进度条，清晰看到运行状态
    for i in tqdm(range(1, samples + 1), desc=f"Generating {png_root.name}"):
        imgs, meta = generate_single_soi(digit_pool)

        # 每个样本建立独立文件夹
        img_dir = images_base_dir / f"image_{i}"
        img_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(imgs):
            img.save(img_dir / f"{idx + 1}.png")

        meta["image"] = f"image_{i}"
        meta["source"] = str(png_root.parent)

        # 保存单个样本的标注
        with (img_dir / "annotation.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        all_annotations.append(meta)

    # 最后保存总的测试数据 JSON
    with (out_dir / "soi_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"\n[Success] Dataset generated at: {out_dir}")

# ======================
# CLI
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png_root", type=str, default="./mnist/mnist_png", help="MNIST 0-9 根目录")
    parser.add_argument("--samples", type=int, default=100, help="样本数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    args = parser.parse_args()

    # 自动计算输出目录
    input_path = Path(args.png_root)
    # 如果路径是 .../mnist/mnist_png，则输出到 .../mnist/soi_test_data
    out_dir = input_path.parent / "soi_test_data"
    
    seed = args.seed if args.seed is not None else random.randint(0, 10000)

    generate_soi_dataset(
        png_root=str(input_path),
        out_dir=str(out_dir),
        samples=args.samples,
        seed=seed,
    )