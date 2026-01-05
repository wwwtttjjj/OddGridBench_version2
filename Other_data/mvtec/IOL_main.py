import random
import json
from pathlib import Path
from PIL import Image
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock
from merge_all_data import merge_iol_datasets

# ======================
# 参数设置
# ======================
MIN_GRID = 3
MAX_GRID = 5

MIN_IMG_MAX_SIDE = 400
MAX_IMG_MAX_SIDE = 500

MIN_GAP = 10
MAX_GAP = 20

MIN_MARGIN = 20
MAX_MARGIN = 35

MIN_CELL_PADDING = 20
MAX_CELL_PADDING = 35

BG_COLOR = (255, 255, 255)
MAX_CANVAS_SIZE = 2048


# ======================
# 图像工具
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


def load_image_list(img_dir: Path):
    imgs = sorted(img_dir.glob("*.JPG")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.bmp"))
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")
    return imgs


# ======================
# 只负责拼图（不 random grid）
# ======================
def generate_single_iol_from_paths(
    rows,
    cols,
    normal_paths,
    anomaly_paths,
):
    num_cells = rows * cols
    odd_k = len(anomaly_paths)

    odd_indices = set(random.sample(range(num_cells), odd_k))

    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)
    img_max_side = random.randint(MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE)

    cells = []
    cell_sizes = []

    n_ptr, a_ptr = 0, 0

    for idx in range(num_cells):
        if idx in odd_indices:
            img_path = anomaly_paths[a_ptr]
            a_ptr += 1
        else:
            img_path = normal_paths[n_ptr]
            n_ptr += 1

        img = Image.open(img_path).convert("RGB")
        img = resize_image_max_side(img, img_max_side)

        w, h = img.size
        cells.append(img)
        cell_sizes.append((w + 2 * cell_padding, h + 2 * cell_padding))

    row_heights = [
        max(cell_sizes[r * cols + c][1] for c in range(cols))
        for r in range(rows)
    ]
    col_widths = [
        max(cell_sizes[r * cols + c][0] for r in range(rows))
        for c in range(cols)
    ]

    grid_w = sum(col_widths) + (cols - 1) * gap
    grid_h = sum(row_heights) + (rows - 1) * gap

    W = grid_w + 2 * margin
    H = grid_h + 2 * margin

    canvas = Image.new("RGB", (W, H), BG_COLOR)

    idx = 0
    y_cursor = margin
    for r in range(rows):
        x_cursor = margin
        for c in range(cols):
            img = cells[idx]
            iw, ih = img.size
            cell_w, _ = cell_sizes[idx]

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
def generate_dataset(data_root, out_dir, samples, seed, num_threads):
    random.seed(seed)

    data_root = Path(data_root)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    normal_all = load_image_list(data_root / "Normal")
    anomaly_all = load_image_list(data_root / "Abnormal")

    random.shuffle(normal_all)
    random.shuffle(anomaly_all)

    normal_pool = normal_all.copy()
    anomaly_pool = anomaly_all.copy()
    normal_ptr = 0

    lock = Lock()
    stop_flag = {"stop": False}
    annotations = []

    def worker(idx):
        nonlocal normal_ptr

        # ✅ 已停机：直接退出，绝不写文件
        if stop_flag["stop"]:
            return None

        # odd 数量偏向 1
        odd_k = random.choices([0, 1, 2], weights=[2, 6, 2])[0]

        rows = random.randint(MIN_GRID, MAX_GRID)
        cols = random.randint(MIN_GRID, MAX_GRID)
        num_cells = rows * cols
        need_normal = num_cells - odd_k

        with lock:
            if stop_flag["stop"]:
                return None

            if len(anomaly_pool) < odd_k:
                # ✅ anomaly 不够：全局停机
                stop_flag["stop"] = True
                return None

            anomaly_paths = [anomaly_pool.pop() for _ in range(odd_k)]

            normal_paths = []
            for _ in range(need_normal):
                if normal_ptr >= len(normal_pool):
                    random.shuffle(normal_pool)
                    normal_ptr = 0
                normal_paths.append(normal_pool[normal_ptr])
                normal_ptr += 1

        # 锁外做重活
        img, meta = generate_single_iol_from_paths(
            rows=rows,
            cols=cols,
            normal_paths=normal_paths,
            anomaly_paths=anomaly_paths,
        )

        # ✅ 再次检查：避免锁外耗时期间已经停机（保险）
        if stop_flag["stop"]:
            return None

        img, scale = resize_longest_side(img, MAX_CANVAS_SIZE)

        name = f"image_{data_root.name}_{idx}.png"
        img.save(img_dir / name)

        meta.update({
            "image": name,
            "image_size": list(img.size),
            "resize_scale": scale,
            # "source_data": f"VisA_{data_root.name}",
        })
        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(samples)]

        for f in as_completed(futures):
            res = f.result()
            if res is None:
                if stop_flag["stop"]:
                    print("[INFO] Anomaly images exhausted, stop generation.")
                    break
                continue
            annotations.append(res)

    with (out_dir / "iol_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ======================
# main
# ======================
if __name__ == "__main__":
    DATA_ROOT = "A_cropped_images"
    SAVE_ROOT = Path("A_iol_type_data")

    if SAVE_ROOT.exists():
        shutil.rmtree(SAVE_ROOT)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    SAMPLES = 1000
    SEED = random.randint(0, 10000)
    THREADS = 8

    subdirs = [
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]

    print(f"[INFO] Found {len(subdirs)} sub-datasets")

    for data_root in sorted(subdirs):
        print(f"[INFO] Generating from {data_root}")
        OUT_DIR = SAVE_ROOT / f"iol_data_{os.path.basename(data_root)}"
        generate_dataset(
            data_root=data_root,
            out_dir=OUT_DIR,
            samples=SAMPLES,
            seed=SEED,
            num_threads=THREADS,
        )

    merge_iol_datasets(
        src_root=SAVE_ROOT,
        dst_root=Path("iol_test_data"),
    )
