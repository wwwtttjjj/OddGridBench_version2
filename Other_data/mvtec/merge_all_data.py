import json
import shutil
from pathlib import Path


def merge_iol_datasets(
    src_root: Path,
    dst_root: Path,
    images_subdir: str = "images",
    json_name: str = "iol_test_data.json",
):
    """
    Merge multiple iol_data_xxx datasets into one directory.

    Result structure:
        dst_root/
        ├── images/
        │   ├── xxx_*.png
        │   └── ...
        └── iol_test_data.json

    Args:
        src_root (Path): directory containing iol_data_xxx subfolders
        dst_root (Path): output merged directory
        images_subdir (str): images directory name (default: images)
        json_name (str): json filename in each sub-dataset
    """

    dst_images = dst_root / images_subdir
    dst_json = dst_root / json_name

    # =========================
    # 初始化目标目录
    # =========================
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_images.mkdir(parents=True, exist_ok=True)

    merged_records = []

    # =========================
    # 遍历所有子数据集
    # =========================
    for subdir in sorted(src_root.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith("iol_data_"):
            continue

        dataset_name = subdir.name.replace("iol_data_", "")
        print(f"[INFO] Merging dataset: {dataset_name}")

        images_dir = subdir / images_subdir
        json_path = subdir / json_name

        if not images_dir.exists() or not json_path.exists():
            print(f"[WARN] Skip {subdir}, missing images or json")
            continue

        # ---------- 1. 读取 json ----------
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            records = list(data.values())
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError(f"Unsupported json format in {json_path}")

        # ---------- 2. 拷贝图片 & 修改 image 路径 ----------
        for item in records:
            img_rel = item.get("image")
            if img_rel is None:
                continue

            src_img = images_dir / img_rel
            if not src_img.exists():
                print(f"[WARN] Missing image: {src_img}")
                continue

            new_img_name = f"{src_img.name}"
            dst_img = dst_images / new_img_name

            shutil.copy2(src_img, dst_img)

            # 更新 json 中的 image 字段（保持你现在的行为）
            item["image"] = new_img_name

            # 记录来源数据集
            item["source_dataset"] = f"MVTEC_{dataset_name}"

        merged_records.extend(records)

    # =========================
    # 保存合并后的 json
    # =========================
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Merge finished")
    print(f"     Total samples : {len(merged_records)}")
    print(f"     Images dir    : {dst_images}")
    print(f"     Json file     : {dst_json}")
    
    
    
def merge_soi_datasets(
    src_root: Path,
    dst_root: Path,
    images_dir_name: str = "images",
    json_name: str = "soi_test_data.json",
):
    """
    Merge SOI datasets while KEEPING image directory structure.

    From:
        soi_data_xxx/
            images/
                images_xxx_1/
                    1.png, 2.png, ...
            soi_test_data.json

    To:
        dst_root/
            images/
                images_xxx_1/
                images_xxx_2/
                ...
            soi_test_data.json
    """

    dst_images_root = dst_root / images_dir_name
    dst_json = dst_root / json_name

    # ===== reset output =====
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_images_root.mkdir(parents=True, exist_ok=True)

    merged_records = []

    # ===== iterate all soi_data_xxx =====
    for dataset_dir in sorted(src_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if not dataset_dir.name.startswith("soi_data_"):
            continue

        dataset_name = dataset_dir.name.replace("soi_data_", "")
        print(f"[INFO] Merging SOI dataset: {dataset_name}")

        src_images_root = dataset_dir / images_dir_name
        src_json = dataset_dir / json_name

        if not src_images_root.exists() or not src_json.exists():
            print(f"[WARN] Skip {dataset_dir}, missing images or json")
            continue

        # ---- copy image directories ----
        for sample_dir in sorted(src_images_root.iterdir()):
            if not sample_dir.is_dir():
                continue

            dst_sample_dir = dst_images_root / sample_dir.name
            if dst_sample_dir.exists():
                raise RuntimeError(
                    f"Duplicate sample directory name detected: {sample_dir.name}"
                )

            shutil.copytree(sample_dir, dst_sample_dir)

        # ---- load & merge json ----
        with open(src_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise TypeError(f"{src_json} must be a list")

        for item in records:
            # image 字段保持不变（仍然是 images_xxx_k）
            item["source_dataset"] = f"MVTEC_{dataset_name}"
            merged_records.append(item)

    # ===== save merged json =====
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=2)

    print("\n[OK] SOI merge finished")
    print(f"     Total samples : {len(merged_records)}")
    print(f"     Images dir    : {dst_images_root}")
    print(f"     Json file     : {dst_json}")

