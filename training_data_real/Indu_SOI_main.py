import random
import json
import shutil
import uuid
from pathlib import Path
import os

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from merge_all_data import merge_soi_datasets
from configs import (
    MIN_SET_SIZE, MAX_SET_SIZE,
    odd_nums, odd_pro,
    MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE,
    MIN_CELL_PADDING, MAX_CELL_PADDING,
    BG_COLOR, MAX_CANVAS_SIZE,
    load_image_list, resize_image_max_side
)

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
    with pool_lock:
        # ❗ 只有完全没有 anomaly 才退出（这里不设置 stop_flag）
        if len(anomaly_pool) == 0:
            return None, None

        num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)

        # 原始采样
        odd_k = random.choices(odd_nums, weights=odd_pro)[0]

        # ✅ 自动降级（关键）
        odd_k = min(odd_k, num_images, len(anomaly_pool))

        # 正常数量
        need_normal = num_images - odd_k

        # 随机位置
        odd_indices_set = set(random.sample(range(num_images), odd_k))

        # ✅ 先消费 anomaly（关键）
        odd_paths = [anomaly_pool.pop() for _ in range(odd_k)]

        # ✅ 在消费之后再决定 stop（核心修复！！！）
        if len(anomaly_pool) == 0:
            stop_flag["stop"] = True

        # normal 无限复用
        normal_paths = []
        for _ in range(need_normal):
            if normal_ptr_ref[0] >= len(normal_pool):
                random.shuffle(normal_pool)
                normal_ptr_ref[0] = 0
            normal_paths.append(normal_pool[normal_ptr_ref[0]])
            normal_ptr_ref[0] += 1

    # ---------- 图像处理 ----------
    img_max_side = random.randint(MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)

    images = []
    source_images_info = []

    n_ptr, a_ptr = 0, 0

    for idx in range(num_images):
        is_anomaly = idx in odd_indices_set

        if is_anomaly:
            p = odd_paths[a_ptr]
            a_ptr += 1
        else:
            p = normal_paths[n_ptr]
            n_ptr += 1

        source_images_info.append({
            "index_in_sequence": idx + 1,
            "original_filename": p.name,
            "label": "anomaly" if is_anomaly else "normal"
        })

        img = Image.open(p).convert("RGB")
        img, scale = resize_image_max_side(img, img_max_side)

        w, h = img.size
        padded = Image.new(
            "RGB",
            (w + 2 * cell_padding, h + 2 * cell_padding),
            BG_COLOR
        )
        padded.paste(img, (cell_padding, cell_padding))
        images.append(padded)

    meta = {
        "id": str(uuid.uuid4()),
        "task_type": "soi",
        "total_icons": num_images,
        "num_odds": odd_k,
        "odd_indices": sorted([i + 1 for i in odd_indices_set]),
        "source_images_details": source_images_info,
        "cell_padding": cell_padding,
        "img_max_side": img_max_side
    }

    return images, meta


# ======================
# Generate SOI dataset
# ======================
def generate_soi_dataset(data_root, out_dir, samples=1000, seed=0, num_threads=8):
    random.seed(seed)

    data_name = Path(data_root).name
    data_root, out_dir = Path(data_root), Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    normal_pool = load_image_list(data_root / "Normal")
    anomaly_pool = load_image_list(data_root / "Anomaly")

    random.shuffle(normal_pool)
    random.shuffle(anomaly_pool)

    normal_ptr_ref = [0]
    stop_flag = {"stop": False}

    pool_lock = Lock()
    all_annotations = []

    def worker(i):
        if stop_flag["stop"]:
            return None

        imgs, meta = generate_single_soi(
            normal_pool,
            anomaly_pool,
            normal_ptr_ref,
            pool_lock,
            stop_flag
        )

        if imgs is None:
            return None

        sample_folder_name = f"image_{data_name}_{i}"
        sample_folder = images_root / sample_folder_name
        sample_folder.mkdir(parents=True, exist_ok=True)

        scales = []
        for idx, img in enumerate(imgs):
            img, scale = resize_image_max_side(img, MAX_CANVAS_SIZE)
            img.save(sample_folder / f"{idx + 1}.png")
            scales.append(scale)

        meta.update({
            "image": sample_folder_name,
            "resize_scale": min(scales)
        })

        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(1, samples + 1)]

        for f in as_completed(futures):
            res = f.result()

            if res:
                all_annotations.append(res)

            # ✅ stop 但继续把已提交的线程跑完
            if stop_flag["stop"]:
                continue

    with (out_dir / "soi_test_data.json").open("w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)


# ======================
# Merge metadata
# ======================
def merge_all_soi_details(src_root, image_dir):
    src_root = Path(src_root)
    combined_data = []

    json_files = list(src_root.glob("*/soi_test_data.json"))
    print(f"[INFO] Merging {len(json_files)} SOI metadata files...")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

            category = json_file.parent.name.replace("soi_data_", "")

            for entry in data:
                entry["category"] = category
                entry["dataset_name"] = str(src_root.parent)
                entry["image_dir"] = image_dir

            combined_data.extend(data)

    save_path = src_root / "all_soi_combined_metadata.json"

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Merged {len(combined_data)} entries into {save_path}")


# ======================
# Main
# ======================
def main(DATA_NAME, IMAGE_DIR):
    DATA_ROOT = Path(DATA_NAME) / IMAGE_DIR
    SAVE_ROOT = Path(DATA_NAME) / "A_soi_type_data"

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

    print(f"[INFO] Found {len(subdirs)} sub-datasets under {DATA_ROOT}")

    for data_root in sorted(subdirs):
        print(f"[INFO] Generating dataset from: {data_root}")

        OUT_DIR = SAVE_ROOT / f"soi_data_{os.path.basename(data_root)}"

        generate_soi_dataset(
            data_root=data_root,
            out_dir=OUT_DIR,
            samples=SAMPLES,
            seed=SEED,
            num_threads=THREADS,
        )

    merge_all_soi_details(SAVE_ROOT, IMAGE_DIR)

    merge_soi_datasets(
        src_root=SAVE_ROOT,
        dst_root=Path(DATA_NAME) / "soi_test_data",
    )

# ======================
# CLI（保持不变）
# ======================
if __name__ == "__main__":
    
    DATA_NAME = "GOODADS"
    IMAGE_DIR = "manual_images/"
    main(DATA_NAME, IMAGE_DIR)
    
    DATA_NAME = "mvtec_ad2/"
    IMAGE_DIR = "manual_images/"
    main(DATA_NAME, IMAGE_DIR)
    
    DATA_NAME = "MVTEC_LOCO"
    IMAGE_DIR = "manual_images/"
    main(DATA_NAME, IMAGE_DIR)
    
    
    main(DATA_NAME, IMAGE_DIR)