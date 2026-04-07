import json
import shutil
from pathlib import Path

def extract_original_images_from_json(source_data, json_path, output_root, data_type="iol"):
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[SKIP] 文件不存在: {json_path}")
        return 0, 0 # 返回两个计数器

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_root = Path(output_root)
    count = 0          # 总拷贝数
    anomaly_count = 0  # 异常图片拷贝数
    missing_files = []

    print(f"[INFO] 正在处理 {data_type.upper()} 类型，共 {len(data)} 条条目...")

    for entry in data:
        dataset_name = entry.get("dataset_name", "")
        image_dir = entry.get("image_dir", "")
        category = entry.get("category", "")
        base_origin_path = Path(source_data) / dataset_name / image_dir / category

        cells = entry.get("source_cells") or entry.get("source_images_details") or []
        
        for cell in cells:
            label = str(cell["label"]).lower() 
            orig_name = cell.get("original_filename") or cell.get("original_name")
            if not orig_name: continue

            # 判断是否为异常
            is_anomaly = (label != "normal")
            label_dir = "Anomaly" if is_anomaly else "Normal"
            
            src_file = base_origin_path / label_dir / orig_name
            dst_dir = output_root / data_type / dataset_name / category / label_dir
            
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / orig_name

            if src_file.exists():
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    count += 1
                    if is_anomaly:
                        anomaly_count += 1
            else:
                missing_files.append(str(src_file))

    print(f"--- {data_type.upper()} 处理完成 ---")
    print(f"    >> 新拷贝总数: {count} 张")
    print(f"    >> 其中异常(Anomaly)图片: {anomaly_count} 张")
    
    if missing_files:
        print(f"[WARNING] 缺失文件数: {len(missing_files)} (唯一文件数: {len(set(missing_files))})")
    
    return count, anomaly_count

# ======================
# 封装后的 Main 函数
# ======================
def main(DATA_NAME, SOURCE_ROOT="../Other_data", TARGET_ROOT="single_data"):
    SOURCE_ROOT = Path(SOURCE_ROOT)
    TARGET_DIR = Path(TARGET_ROOT)
    
    tasks = [
        ("iol", SOURCE_ROOT / DATA_NAME / "A_iol_type_data" / "all_iol_combined_metadata.json"),
        ("soi", SOURCE_ROOT / DATA_NAME / "A_soi_type_data" / "all_soi_combined_metadata.json")
    ]

    total_new_copied = 0
    total_anomaly_copied = 0

    for d_type, json_path in tasks:
        print(f"\n{'='*60}")
        print(f"[START] 任务类型: {d_type.upper()}")
        
        c, ac = extract_original_images_from_json(
            source_data=SOURCE_ROOT, 
            json_path=json_path, 
            output_root=TARGET_DIR, 
            data_type=d_type
        )
        total_new_copied += c
        total_anomaly_copied += ac

    print(f"\n{'#'*60}")
    print(f"【所有任务最终统计】")
    print(f"累计新拷贝总数: {total_new_copied}")
    print(f"累计异常(Anomaly)图片总数: {total_anomaly_copied}")
    print(f"{'#'*60}")

if __name__ == "__main__":
    # main("VisA")
    main("BTech_Dataset_transformed")
    # main("mvtec")
    # main("ELPV")
    