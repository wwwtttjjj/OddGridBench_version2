import random
import json
import shutil
import uuid
from pathlib import Path
import os

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from merge_all_data import merge_soi_datasets

from configs import (
    MIN_SET_SIZE, MAX_SET_SIZE,
    MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE,
    MIN_CELL_PADDING, MAX_CELL_PADDING,
    BG_COLOR, MAX_CANVAS_SIZE,
    load_image_list, resize_image_max_side
)

# ======================
# Generate single SOI sample
# ======================
def split_pool_into_groups(image_pool, min_size, max_size):
    groups = []
    ptr = 0

    while ptr < len(image_pool):
        group_size = random.randint(min_size, max_size)
        groups.append(image_pool[ptr:ptr + group_size])
        ptr += group_size

    # 如果最后一组数量小于 min_size，就和倒数第二组重新均分
    if len(groups) >= 2 and len(groups[-1]) < min_size:
        merged = groups[-2] + groups[-1]
        random.shuffle(merged)

        left_size = len(merged) // 2
        groups[-2] = merged[:left_size]
        groups[-1] = merged[left_size:]

    return groups


def generate_single_soi(sampled_items):
    img_max_side = random.randint(MIN_IMG_MAX_SIDE, MAX_IMG_MAX_SIDE)
    cell_padding = random.randint(MIN_CELL_PADDING, MAX_CELL_PADDING)

    images = []
    source_images_info = []
    odd_indices = []

    for idx, item in enumerate(sampled_items):
        p = item["path"]
        label = item["label"]

        if label == "anomaly":
            odd_indices.append(idx + 1)

        source_images_info.append({
            "index_in_sequence": idx + 1,
            "original_filename": p.name,
            "label": label
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
        "total_icons": len(sampled_items),
        "num_odds": len(odd_indices),
        "odd_indices": sorted(odd_indices),
        "source_images_details": source_images_info,
        "cell_padding": cell_padding,
        "img_max_side": img_max_side
    }

    return images, meta


def generate_soi_dataset(data_root, out_dir, samples=1000, seed=0, num_threads=8):
    random.seed(seed)

    data_name = Path(data_root).name
    data_root, out_dir = Path(data_root), Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

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

    sample_groups = split_pool_into_groups(
        image_pool=image_pool,
        min_size=MIN_SET_SIZE,
        max_size=MAX_SET_SIZE
    )

    # 如果外部传入 samples，就最多生成 samples 组
    sample_groups = sample_groups[:samples]

    all_annotations = []

    def worker(i, sampled_items):
        imgs, meta = generate_single_soi(sampled_items)

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
        futures = [
            executor.submit(worker, i, sampled_items)
            for i, sampled_items in enumerate(sample_groups, start=1)
        ]

        for f in as_completed(futures):
            res = f.result()

            if res:
                all_annotations.append(res)

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
    THREADS = 16

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
    DATASETS = [
        ("BTech_Dataset_transformed", "A_cropped_images/"),
        # ("ELPV", "A_cropped_images/"),
        # ("mvtec", "A_cropped_images/"),
        # ("VisA", "A_cropped_images/"),
        # ("GOODADS", "A_cropped_images/"),
        # ("MPDD", "A_cropped_images/"),
        # ("RAD", "A_cropped_images/"),
    ]

    for DATA_NAME, IMAGE_DIR in DATASETS:
        print(f"\n[INFO] Processing dataset: {DATA_NAME}, image dir: {IMAGE_DIR}")
        main(DATA_NAME, IMAGE_DIR)

    
