import random
import json
from pathlib import Path
from PIL import Image, ImageDraw
import uuid
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================
# 参数范围（基本沿用你原来的）
# ======================
MIN_GRID = 5
MAX_GRID = 6

MIN_CELL_SIZE = 200
MAX_CELL_SIZE = 200

MIN_GAP = 10
MAX_GAP = 30

MIN_ODD = 1
MAX_ODD = 3

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


def load_image_list(img_dir: Path):
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        raise RuntimeError(f"No png images in {img_dir}")
    return imgs


# ======================
# 单样本生成（normal vs anomaly）
# ======================
def generate_single_iol(normal_imgs, anomaly_imgs):
    rows = random.randint(MIN_GRID, MAX_GRID)
    cols = random.randint(MIN_GRID, MAX_GRID)
    num_cells = rows * cols

    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_cells))

    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)

    noise_sigma = random.uniform(0.03, 0.05)

    # 随机选 anomaly 位置
    all_indices = list(range(num_cells))
    odd_indices = set(random.sample(all_indices, odd_k))

    # normal / anomaly 图像池
    normal_paths = random.choices(normal_imgs, k=num_cells - odd_k)
    anomaly_paths = random.sample(
        anomaly_imgs, odd_k
    ) if len(anomaly_imgs) >= odd_k else random.choices(anomaly_imgs, k=odd_k)

    normal_ptr = 0
    anomaly_ptr = 0

    grid_w = cols * cell_size + (cols - 1) * gap
    grid_h = rows * cell_size + (rows - 1) * gap
    W = grid_w + 2 * margin
    H = grid_h + 2 * margin

    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for idx in range(num_cells):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_size + gap)
        y = margin + r * (cell_size + gap)

        if idx in odd_indices:
            img_path = anomaly_paths[anomaly_ptr]
            anomaly_ptr += 1
        else:
            img_path = normal_paths[normal_ptr]
            normal_ptr += 1

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
        "odd_count": odd_k,
        "odd_rows_cols": odd_positions,
        "grid_size": [rows, cols],
        "gap": gap,
        "margin": margin,
        "normal_images": [p.name for p in normal_paths],
        "anomaly_images": [p.name for p in anomaly_paths],
    }

    return canvas, meta


# ======================
# Dataset 生成
# ======================
def generate_dataset(
    data_root: str,
    out_dir: str,
    samples: int,
    seed: int,
    num_threads: int,
):
    random.seed(seed)

    data_root = Path(data_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    normal_imgs = load_image_list(data_root / "normal")
    anomaly_imgs = load_image_list(data_root / "anomaly")

    annotations = []

    def worker(idx):
        img, meta = generate_single_iol(normal_imgs, anomaly_imgs)
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

    with (out_dir / "iol_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ======================
# 入口
# ======================
if __name__ == "__main__":
    DATA_ROOT = "ELPV_split/train"   # 包含 normal/ anomaly
    OUT_DIR = "iol_test_data"
    SAMPLES = 100
    SEED = random.randint(0, 10000)
    THREADS = 8

    generate_dataset(
        data_root=DATA_ROOT,
        out_dir=OUT_DIR,
        samples=SAMPLES,
        seed=SEED,
        num_threads=THREADS,
    )
