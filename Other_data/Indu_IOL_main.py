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

from configs import (
    MIN_GRID, MAX_GRID,
    MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE,
    MIN_GAP, MAX_GAP,
    MIN_MARGIN, MAX_MARGIN,
    MIN_CELL_PADDING, MAX_CELL_PADDING,
    BG_COLOR, MAX_CANVAS_SIZE,
    resize_image_max_side, load_image_list
)

# ======================
# 拼图函数（不变）
# ======================
def generate_single_iol_from_items(rows, cols, cell_items):
    num_cells = rows * cols

    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)
    img_max_side = random.randint(MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE)

    cells = []
    cell_sizes = []
    cells_info = []
    odd_indices = set()

    for idx in range(num_cells):
        item = cell_items[idx]

        if item is None:
            img = None
            original_name = None
            label = None
        else:
            img_path = item["path"]
            label = item["label"]
            original_name = Path(img_path).name

            if label == "anomaly":
                odd_indices.add(idx)

            img = Image.open(img_path).convert("RGB")
            img, _ = resize_image_max_side(img, img_max_side)

        cells_info.append({
            "cell_index": idx,
            "grid_pos": [idx // cols + 1, idx % cols + 1],
            "original_name": original_name,
            "label": label
        })

        cells.append(img)

        if img is None:
            cell_sizes.append((img_max_side + 2 * cell_padding, img_max_side + 2 * cell_padding))
        else:
            cell_sizes.append((img.width + 2 * cell_padding, img.height + 2 * cell_padding))

    row_heights = [max(cell_sizes[r * cols + c][1] for c in range(cols)) for r in range(rows)]
    col_widths = [max(cell_sizes[r * cols + c][0] for r in range(rows)) for c in range(cols)]

    canvas = Image.new(
        "RGB",
        (
            sum(col_widths) + (cols - 1) * gap + 2 * margin,
            sum(row_heights) + (rows - 1) * gap + 2 * margin
        ),
        BG_COLOR
    )

    idx = 0
    y_cursor = margin

    for r in range(rows):
        x_cursor = margin
        for c in range(cols):
            img = cells[idx]

            if img is not None:
                x_img = x_cursor + (col_widths[c] - img.width) // 2
                y_img = y_cursor + (row_heights[r] - img.height) // 2
                canvas.paste(img, (x_img, y_img))

            x_cursor += col_widths[c] + gap
            idx += 1

        y_cursor += row_heights[r] + gap

    meta = {
        "id": str(uuid.uuid4()),
        "grid_size": [rows, cols],
        "odd_count": len(odd_indices),
        "odd_rows_cols": sorted([[i // cols + 1, i % cols + 1] for i in odd_indices]),
        "source_cells": cells_info
    }

    return canvas, meta

def choose_iol_grid_size(item_count):
    valid_rows = [
        rows for rows in range(MIN_GRID, MAX_GRID + 1)
        if rows * rows >= item_count
    ]

    if valid_rows:
        rows = random.choice(valid_rows)
    else:
        rows = MAX_GRID

    return rows, rows


def split_pool_into_iol_groups(image_pool, min_last_items=5):
    groups = []
    ptr = 0

    while ptr < len(image_pool):
        rows = random.randint(MIN_GRID, MAX_GRID)
        num_cells = rows * rows

        items = image_pool[ptr:ptr + num_cells]
        ptr += num_cells

        groups.append({
            "rows": rows,
            "cols": rows,
            "items": items
        })

    # 最后一组太少，就和倒数第二组重新平分
    if len(groups) >= 2 and len(groups[-1]["items"]) < min_last_items:
        merged_items = groups[-2]["items"] + groups[-1]["items"]
        random.shuffle(merged_items)

        left_size = len(merged_items) // 2
        left_items = merged_items[:left_size]
        right_items = merged_items[left_size:]

        left_rows, left_cols = choose_iol_grid_size(len(left_items))
        right_rows, right_cols = choose_iol_grid_size(len(right_items))

        groups[-2] = {
            "rows": left_rows,
            "cols": left_cols,
            "items": left_items
        }
        groups[-1] = {
            "rows": right_rows,
            "cols": right_cols,
            "items": right_items
        }

    return groups

# ======================
# 数据集生成（修复版）
# ======================
def generate_dataset(data_root, out_dir, samples, seed, num_threads):
    random.seed(seed)

    data_root, out_dir = Path(data_root), Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    normal_pool = [
        {"path": p, "label": "normal"}
        for p in load_image_list(data_root / "Normal")
    ]

    anomaly_pool = [
        {"path": p, "label": "anomaly"}
        for p in load_image_list(data_root / "Anomaly")
    ]

    image_pool = normal_pool + anomaly_pool
    random.shuffle(image_pool)

    sample_groups = split_pool_into_iol_groups(
        image_pool=image_pool,
        min_last_items=5
    )

    sample_groups = sample_groups[:samples]

    annotations = []

    def worker(idx, group):
        rows = group["rows"]
        cols = group["cols"]
        num_cells = rows * cols

        cell_items = list(group["items"])

        # 不够一个完整网格的位置用 None 补齐
        if len(cell_items) < num_cells:
            cell_items.extend([None] * (num_cells - len(cell_items)))

        img, meta = generate_single_iol_from_items(
            rows, cols, cell_items
        )

        img, scale = resize_image_max_side(img, MAX_CANVAS_SIZE)

        name = f"image_{data_root.name}_{idx}.png"
        img.save(img_dir / name)

        meta.update({
            "image": name,
            "image_size": list(img.size),
            "resize_scale": scale
        })

        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(worker, i, group)
            for i, group in enumerate(sample_groups)
        ]

        for f in as_completed(futures):
            res = f.result()

            if res:
                annotations.append(res)

    with (out_dir / "iol_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

# ======================
# merge（不变）
# ======================

def merge_all_details(src_root, image_dir):
    src_root = Path(src_root)
    combined_data = []

    json_files = list(src_root.glob("*/iol_test_data.json"))
    print(f"[INFO] Merging {len(json_files)} metadata files...")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

            category = json_file.parent.name.replace("iol_data_", "")

            for entry in data:
                entry["category"] = category
                entry["dataset_name"] = str(src_root.parent)
                entry["image_dir"] = image_dir

            combined_data.extend(data)

    save_path = src_root / "all_iol_combined_metadata.json"

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Total merged entries: {len(combined_data)}")
    print(f"[SUCCESS] Final metadata saved to: {save_path}")


# ======================
# main（保持你原样，不动）
# ======================
def main(DATA_NAME, IMAGE_DIR):
    DATA_ROOT = Path(DATA_NAME) / IMAGE_DIR
    SAVE_ROOT = Path(DATA_NAME) / "A_iol_type_data"

    if SAVE_ROOT.exists():
        shutil.rmtree(SAVE_ROOT)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    SAMPLES = 100000
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

    merge_all_details(SAVE_ROOT, IMAGE_DIR)

    merge_iol_datasets(
        src_root=SAVE_ROOT,
        dst_root=Path(DATA_NAME) / "iol_test_data",
    )

# ======================
# main
# ======================
if __name__ == "__main__":
    
    DATA_NAME = "BTech_Dataset_transformed"
    IMAGE_DIR = "manual_images/"
    main(DATA_NAME, IMAGE_DIR)
    
    DATA_NAME = "mvtec"
    IMAGE_DIR = "A_cropped_images/"
    main(DATA_NAME, IMAGE_DIR)
        
    DATA_NAME = "ELPV"
    IMAGE_DIR = "ELPV_split/"
    
    main(DATA_NAME, IMAGE_DIR)
    
    DATA_NAME = "VisA"
    IMAGE_DIR = "A_cropped_images/"
    main(DATA_NAME, IMAGE_DIR)
