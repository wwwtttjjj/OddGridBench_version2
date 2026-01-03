import random
import json
from pathlib import Path
from PIL import Image
import uuid
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from merge_all_data import merge_iol_datasets

# ======================
# 参数设置
# ======================
MIN_GRID = 3
MAX_GRID = 5

MIN_IMG_MAX_SIDE = 400
MAX_IMG_MAX_SIDE = 400

MIN_GAP = 10
MAX_GAP = 30

MIN_ODD = 0
MAX_ODD = 2

MIN_MARGIN = 20
MAX_MARGIN = 35

MIN_CELL_PADDING = 20
MAX_CELL_PADDING = 35

BG_COLOR = (255, 255, 255)
MAX_CANVAS_SIZE = 2048  # 最终整图最长边限制


# ======================
# 图像工具函数
# ======================
# def add_gaussian_noise_pil(pil_img, sigma=0.02):
#     img = np.asarray(pil_img).astype(np.float32) / 255.0
#     noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
#     out = np.clip(img + noise, 0.0, 1.0)
#     return Image.fromarray((out * 255.0).astype(np.uint8))


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


def load_image_list(img_dir: Path):
    imgs = sorted(img_dir.glob("*.JPG"))
    if not imgs:
        raise RuntimeError(f"No JPG images in {img_dir}")
    return imgs


# ======================
# 核心生成逻辑
# ======================
def generate_single_iol(normal_imgs, anomaly_imgs):
    rows = random.randint(MIN_GRID, MAX_GRID)
    cols = random.randint(MIN_GRID, MAX_GRID)
    num_cells = rows * cols

    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_cells))
    odd_indices = set(random.sample(range(num_cells), odd_k))

    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)
    img_max_side = random.randint(MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE)
    noise_sigma = random.uniform(0.03, 0.05)

    normal_paths = random.choices(normal_imgs, k=num_cells - odd_k)
    anomaly_paths = (
        random.sample(anomaly_imgs, odd_k)
        if len(anomaly_imgs) >= odd_k
        else random.choices(anomaly_imgs, k=odd_k)
    )

    normal_ptr = 0
    anomaly_ptr = 0

    # -------- 先准备所有 cell 的 image 和 cell 尺寸 --------
    cells = []
    cell_sizes = []

    for idx in range(num_cells):
        if idx in odd_indices:
            img_path = anomaly_paths[anomaly_ptr]
            anomaly_ptr += 1
        else:
            img_path = normal_paths[normal_ptr]
            normal_ptr += 1

        img = Image.open(img_path).convert("RGB")
        img = resize_image_max_side(img, img_max_side)
        # img = add_gaussian_noise_pil(img, sigma=noise_sigma)

        w, h = img.size
        cell_w = w + 2 * cell_padding
        cell_h = h + 2 * cell_padding

        cells.append(img)
        cell_sizes.append((cell_w, cell_h))

    # -------- 计算 grid 尺寸（行高对齐） --------
    row_heights = []
    for r in range(rows):
        heights = [
            cell_sizes[r * cols + c][1]
            for c in range(cols)
        ]
        row_heights.append(max(heights))

    col_widths = []
    for c in range(cols):
        widths = [
            cell_sizes[r * cols + c][0]
            for r in range(rows)
        ]
        col_widths.append(max(widths))

    grid_w = sum(col_widths) + (cols - 1) * gap
    grid_h = sum(row_heights) + (rows - 1) * gap

    W = grid_w + 2 * margin
    H = grid_h + 2 * margin

    canvas = Image.new("RGB", (W, H), BG_COLOR)

    # -------- paste 图像 --------
    idx = 0
    y_cursor = margin
    for r in range(rows):
        x_cursor = margin
        for c in range(cols):
            img = cells[idx]
            iw, ih = img.size
            cell_w, cell_h = cell_sizes[idx]

            x_img = x_cursor + (cell_w - iw) // 2
            y_img = y_cursor + (row_heights[r] - ih) // 2

            canvas.paste(img, (x_img, y_img))

            x_cursor += col_widths[c] + gap
            idx += 1
        y_cursor += row_heights[r] + gap

    odd_positions = sorted([[i // cols + 1, i % cols + 1] for i in odd_indices])

    meta = {
        "id": str(uuid.uuid4()),
        "grid_size": [rows, cols],
        "odd_count": odd_k,
        "odd_rows_cols": odd_positions,
        "gap": gap,
        "margin": margin,
    }

    return canvas, meta


# ======================
# 数据集生成
# ======================
def generate_dataset(data_root: str, out_dir: str, samples: int, seed: int, num_threads: int):
    random.seed(seed)

    data_root = Path(data_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    normal_imgs = load_image_list(data_root / "Normal")
    anomaly_imgs = load_image_list(data_root / "Anomaly")

    annotations = []

    def worker(idx):
        img, meta = generate_single_iol(normal_imgs, anomaly_imgs)
        # img = add_gaussian_noise_pil(img, sigma=0.01)

        img, scale = resize_longest_side(img, MAX_CANVAS_SIZE)

        name = f"image_{os.path.basename(data_root)}_{idx}.png"
        img.save(img_dir / name)

        meta.update({
            "image": name,
            "image_size": list(img.size),
            "resize_scale": scale,
        })
        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(samples)]
        for j, f in enumerate(as_completed(futures)):
            annotations.append(f.result())
            if (j + 1) % 100 == 0:
                print(f"[{j+1}/{samples}] generated")

    with (out_dir / "iol_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ======================
# main
# ======================
import shutil
if __name__ == "__main__":
    DATA_ROOT = "A_cropped_images/"
        # 清空输出目录
    save_root = Path("A_iol_type_data/")
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    SAMPLES = 10
    SEED = random.randint(0, 10000)
    THREADS = 8

    # ===== 枚举 DATA_ROOT 下的所有子目录 =====
    subdirs = [
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]

    print(f"[INFO] Found {len(subdirs)} sub-datasets under {DATA_ROOT}")

    for data_root in sorted(subdirs):
        print(f"[INFO] Generating dataset from: {data_root}")
        OUT_DIR = save_root / f"iol_data_{os.path.basename(data_root)}"
        generate_dataset(
            data_root=data_root,
            out_dir=OUT_DIR,
            samples=SAMPLES,
            seed=SEED,
            num_threads=THREADS,
        )
        
    SRC_ROOT = Path("A_iol_type_data")
    DST_ROOT = Path("iol_test_data")

    merge_iol_datasets(
        src_root=SRC_ROOT,
        dst_root=DST_ROOT,
    )