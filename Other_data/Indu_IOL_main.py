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

from configs import MIN_GRID, MAX_GRID, odd_nums, odd_pro, MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE, MIN_GAP, MAX_GAP, MIN_MARGIN, MAX_MARGIN, MIN_CELL_PADDING, MAX_CELL_PADDING, BG_COLOR, MAX_CANVAS_SIZE
from configs import resize_image_max_side, load_image_list


# ======================
# 只负责拼图（不 random grid）
# ======================

def generate_single_iol_from_paths(rows, cols, normal_paths, anomaly_paths):
    num_cells = rows * cols
    odd_k = len(anomaly_paths)
    odd_indices = set(random.sample(range(num_cells), odd_k))

    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)
    img_max_side = random.randint(MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE)

    cells = []
    cell_sizes = []
    cells_info = [] # 记录图片原始来源和标签

    n_ptr, a_ptr = 0, 0
    for idx in range(num_cells):
        is_anomaly = idx in odd_indices
        img_path = anomaly_paths[a_ptr] if is_anomaly else normal_paths[n_ptr]
        
        cells_info.append({
            "cell_index": idx,
            "grid_pos": [idx // cols + 1, idx % cols + 1],
            "original_name": Path(img_path).name,
            "label": "anomaly" if is_anomaly else "normal"
        })

        if is_anomaly: a_ptr += 1
        else: n_ptr += 1

        img = Image.open(img_path).convert("RGB")
        img, _ = resize_image_max_side(img, img_max_side)
        cells.append(img)
        cell_sizes.append((img.width + 2 * cell_padding, img.height + 2 * cell_padding))

    # 计算网格尺寸并绘图
    row_heights = [max(cell_sizes[r * cols + c][1] for c in range(cols)) for r in range(rows)]
    col_widths = [max(cell_sizes[r * cols + c][0] for r in range(rows)) for c in range(cols)]
    
    canvas = Image.new("RGB", (sum(col_widths) + (cols - 1) * gap + 2 * margin, 
                               sum(row_heights) + (rows - 1) * gap + 2 * margin), BG_COLOR)

    idx = 0
    y_cursor = margin
    for r in range(rows):
        x_cursor = margin
        for c in range(cols):
            img = cells[idx]
            x_img = x_cursor + (col_widths[c] - img.width) // 2
            y_img = y_cursor + (row_heights[r] - img.height) // 2
            canvas.paste(img, (x_img, y_img))
            x_cursor += col_widths[c] + gap
            idx += 1
        y_cursor += row_heights[r] + gap

    meta = {
        "id": str(uuid.uuid4()),
        "grid_size": [rows, cols],
        "odd_count": odd_k,
        "odd_rows_cols": sorted([[i // cols + 1, i % cols + 1] for i in odd_indices]),
        "source_cells": cells_info  # 包含所有子图详细溯源
    }
    return canvas, meta

# ======================
# 数据集生成
# ======================
def generate_dataset(data_root, out_dir, samples, seed, num_threads):
    random.seed(seed)
    data_root, out_dir = Path(data_root), Path(out_dir)
    if out_dir.exists(): shutil.rmtree(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    normal_pool = load_image_list(data_root / "Normal")
    anomaly_pool = load_image_list(data_root / "Anomaly")
    random.shuffle(normal_pool)
    random.shuffle(anomaly_pool)

    normal_ptr = 0
    lock = Lock()
    stop_flag = {"stop": False}
    annotations = []

    def worker(idx):
        nonlocal normal_ptr
        if stop_flag["stop"]: return None
        
        odd_k = random.choices(odd_nums, weights=odd_pro)[0]
        rows = random.randint(MIN_GRID, MAX_GRID)
        num_cells = rows * rows

        with lock:
            if len(anomaly_pool) < odd_k:
                stop_flag["stop"] = True
                return None
            anomaly_paths = [anomaly_pool.pop() for _ in range(odd_k)]
            normal_paths = []
            for _ in range(num_cells - odd_k):
                if normal_ptr >= len(normal_pool):
                    random.shuffle(normal_pool); normal_ptr = 0
                normal_paths.append(normal_pool[normal_ptr]); normal_ptr += 1

        img, meta = generate_single_iol_from_paths(rows, rows, normal_paths, anomaly_paths)
        img, scale = resize_image_max_side(img, MAX_CANVAS_SIZE)
        name = f"image_{data_root.name}_{idx}.png"
        img.save(img_dir / name)
        
        meta.update({"image": name, "image_size": list(img.size), "resize_scale": scale})
        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(samples)]
        for f in as_completed(futures):
            res = f.result()
            if res: annotations.append(res)

    # 仅保存一个包含全部信息的 JSON
    with (out_dir / "iol_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

def merge_all_details(src_root, image_dir):
    """
    将所有子文件夹下的 iol_test_data.json 合并为一个总的源数据追踪文件
    """
    src_root = Path(src_root)
    combined_data = []
    
    # 搜索子目录下的 json 文件
    json_files = list(src_root.glob("*/iol_test_data.json"))
    print(f"[INFO] Merging {len(json_files)} metadata files...")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # 可以在这里为每个 entry 增加所属类别标签
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
    
def save_detailed_info(annotations, out_dir, filename="iol_source_details.json"):
    """
    保存每张生成图的详细组成信息，包括每个 Cell 的来源路径和标签。
    """
    out_dir = Path(out_dir)
    save_path = out_dir / filename
    
    # 提取关键信息构建更简洁的详情表
    detailed_data = []
    for ann in annotations:
        detailed_data.append({
            "image_id": ann["id"],
            "output_image_name": ann["image"],
            "grid_size": ann["grid_size"],
            "composition": ann["cells_info"] # 这里包含每个格子的原始信息
        })

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Detailed source information saved to: {save_path}")
    
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
    
    # DATA_NAME = "BTech_Dataset_transformed"
    # IMAGE_DIR = "manual_images/"
    # main(DATA_NAME, IMAGE_DIR)
    
    # DATA_NAME = "mvtec"
    # IMAGE_DIR = "A_cropped_images/"
    # main(DATA_NAME, IMAGE_DIR)
    
    DATA_NAME = "VisA"
    IMAGE_DIR = "A_cropped_images/"
    main(DATA_NAME, IMAGE_DIR)
    
    # DATA_NAME = "ELPV"
    # IMAGE_DIR = "ELPV_split/"
    
    main(DATA_NAME, IMAGE_DIR)
