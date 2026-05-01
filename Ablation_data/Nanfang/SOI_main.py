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
from configs_1 import (
    MIN_SET_SIZE, MAX_SET_SIZE,
    odd_nums, odd_pro,
    MIN_CELL_PADDING, MAX_CELL_PADDING,
    BG_COLOR,
)

# ---------- 图像处理 ----------
MIN_IMG_SIDE = 100
MAX_IMG_SIDE = 600
    
def load_image_list(img_dir: Path):
    imgs = sorted(img_dir.glob("*.JPG")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.bmp")) + sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")
    return imgs

# ======================
# 新增：尺寸约束 resize
# ======================
def resize_image_in_range(img, min_side, max_side):
    w, h = img.size
    min_curr = min(w, h)
    max_curr = max(w, h)

    # 在范围内 → 不动
    if min_curr >= min_side and max_curr <= max_side:
        return img, 1.0

    # 需要放大
    if min_curr < min_side:
        scale = min_side / min_curr
    # 需要缩小
    else:
        scale = max_side / max_curr

    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize((new_w, new_h), Image.BICUBIC)
    return img, scale


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
        if len(anomaly_pool) == 0:
            return None, None

        num_images = random.randint(MIN_SET_SIZE, MAX_SET_SIZE)
        odd_k = random.choices(odd_nums, weights=odd_pro)[0]
        odd_k = min(odd_k, num_images, len(anomaly_pool))

        need_normal = num_images - odd_k
        odd_indices_set = set(random.sample(range(num_images), odd_k))

        # 先消费 anomaly
        odd_paths = [anomaly_pool.pop() for _ in range(odd_k)]

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

        # ✅ 新规则 resize
        img, scale = resize_image_in_range(
            img,
            MIN_IMG_SIDE,
            MAX_IMG_SIDE
        )

        # ✅ padding（如果你不想统一外框，可以删掉这一段）
        w, h = img.size
        padded = Image.new(
            "RGB",
            (w + 2 * cell_padding, h + 2 * cell_padding),
            BG_COLOR
        )
        padded.paste(img, (cell_padding, cell_padding))

        images.append(padded)

        # 👉 如果你要完全自由尺寸，用这一行替代：
        # images.append(img)

    meta = {
        "id": str(uuid.uuid4()),
        "task_type": "soi",
        "total_icons": num_images,
        "num_odds": odd_k,
        "odd_indices": sorted([i + 1 for i in odd_indices_set]),
        "source_images_details": source_images_info,
        "cell_padding": cell_padding,
        "img_resize_rule": {
            "min_side": MIN_IMG_SIDE,
            "max_side": MAX_IMG_SIDE
        }
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

        # ✅ 不再 resize，直接保存
        for idx, img in enumerate(imgs):
            img.save(sample_folder / f"{idx + 1}.png")

        meta.update({
            "image": sample_folder_name,
            "resize_scale": 1.0
        })

        return meta

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(1, samples + 1)]

        for f in as_completed(futures):
            res = f.result()

            if res:
                all_annotations.append(res)

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
                entry["dataset_name"] = "Nanfang"
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
# CLI
# ======================
if __name__ == "__main__":

    DATA_NAME = "./"
    IMAGE_DIR = "filter_images/"
    main(DATA_NAME, IMAGE_DIR)