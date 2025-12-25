import random
import json
from pathlib import Path
from PIL import Image, ImageDraw
import uuid
import numpy as np
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================
# 参数范围
# ======================
MIN_GRID = 6
MAX_GRID = 10

MIN_CELL_SIZE = 40
MAX_CELL_SIZE = 50

MIN_GAP = 10
MAX_GAP = 30

MIN_ODD = 1
MAX_ODD = 5

MIN_MARGIN = 20
MAX_MARGIN = 35

BG_COLOR = (255, 255, 255)


# ======================
# 工具函数
# ======================
def add_gaussian_noise_pil(pil_img, sigma=0.02):
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8))


def load_digit_pool(png_root: Path):
    pool = {}
    for ddir in sorted(p for p in png_root.iterdir() if p.is_dir()):
        imgs = sorted(ddir.glob("*.png"))
        if not imgs:
            raise RuntimeError(f"No png in {ddir}")
        pool[ddir.name] = imgs
    return pool


# ======================
# 单样本生成
# ======================
def generate_single_iol(digit_pool: dict):
    digit = random.choice(list(digit_pool.keys()))
    paths = digit_pool[digit]

    noise_sigma = random.uniform(0.03, 0.05)

    rows = random.randint(MIN_GRID, MAX_GRID)
    cols = random.randint(MIN_GRID, MAX_GRID)
    num_cells = rows * cols

    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_cells))

    base_path = random.choice(paths)

    all_indices = list(range(num_cells))
    odd_indices = set(random.sample(all_indices, odd_k))

    candidates = [p for p in paths if p != base_path]
    odd_paths = (
        random.sample(candidates, odd_k)
        if len(candidates) >= odd_k
        else [random.choice(candidates) for _ in range(odd_k)]
    )

    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)

    grid_w = cols * cell_size + (cols - 1) * gap
    grid_h = rows * cell_size + (rows - 1) * gap
    W = grid_w + 2 * margin
    H = grid_h + 2 * margin

    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    odd_ptr = 0

    for idx in range(num_cells):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_size + gap)
        y = margin + r * (cell_size + gap)

        if idx in odd_indices:
            img_path = odd_paths[odd_ptr]
            odd_ptr += 1
        else:
            img_path = base_path

        img = Image.open(img_path).convert("RGB").resize(
            (cell_size, cell_size), Image.BILINEAR
        )
        img = add_gaussian_noise_pil(img, sigma=noise_sigma)
        canvas.paste(img, (x, y))

        draw.rectangle(
            [x, y, x + cell_size - 1, y + cell_size - 1],
            outline=(0, 0, 0),
            width=2,
        )

    odd_positions = sorted([[i // cols + 1, i % cols + 1] for i in odd_indices])

    meta = {
        "id": str(uuid.uuid4()),
        "class": digit,
        "odd_count": odd_k,
        "odd_rows_cols": odd_positions,
        "grid_size": [rows, cols],
        "gap": gap,
        "margin": margin,
        "base_image": base_path.name,
        "odd_images": [p.name for p in odd_paths],
    }
    # print(digit)

    return canvas, meta


# ======================
# 多线程 Dataset 生成
# ======================
def generate_dataset(
    png_root: str,
    out_dir: str,
    samples: int,
    seed: int,
    num_threads: int,
):
    random.seed(seed)

    png_root = Path(png_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    digit_pool = load_digit_pool(png_root)
    annotations = []

    def worker(idx):
        img, meta = generate_single_iol(digit_pool)
        img = add_gaussian_noise_pil(img, sigma=0.01)
        name = f"image_{idx}.png"
        img.save(img_dir / name)

        W, H = img.size
        meta["image"] = name
        meta["image_size"] = [W, H]
        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(worker, i): i for i in range(samples)
        }

        for j, future in enumerate(as_completed(futures)):
            annotations.append(future.result())
            if (j + 1) % 100 == 0:
                print(f"[{j+1}/{samples}] generated")

    ann_path = out_dir / "iol_test_data.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ======================
# CLI
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png_root", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)

    args = parser.parse_args()

    args.out_dir = str(Path(args.png_root).parent / "iol_test_data")
    seed = args.seed if args.seed is not None else random.randint(0, 10000)

    generate_dataset(
        png_root=args.png_root,
        out_dir=args.out_dir,
        samples=args.samples,
        seed=seed,
        num_threads=args.threads,
    )
