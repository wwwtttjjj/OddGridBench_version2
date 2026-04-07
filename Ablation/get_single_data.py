import json
import shutil
from pathlib import Path

def extract_original_images_from_json(source_data, json_path, output_root, data_type="iol"):
    """
    通用提取逻辑，保持不变
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[SKIP] 文件不存在: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_root = Path(output_root)
    count = 0
    missing_files = []

    print(f"[INFO] 正在处理 {data_type.upper()} 类型，共 {len(data)} 条条目...")

    for entry in data:
        dataset_name = entry.get("dataset_name", "")
        image_dir = entry.get("image_dir", "")
        category = entry.get("category", "")
        base_origin_path = Path(source_data) / dataset_name / image_dir / category

        cells = entry.get("source_cells") or entry.get("source_images_details") or []
        
        for cell in cells:
            label = cell["label"] 
            orig_name = cell.get("original_filename") or cell.get("original_name")
            if not orig_name: continue

            label_dir = "Normal" if label.lower() == "normal" else "Anomaly"
            
            src_file = base_origin_path / label_dir / orig_name
            dst_dir = output_root / data_type / dataset_name / category / label_dir
            
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / orig_name

            if src_file.exists():
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    count += 1
            else:
                missing_files.append(str(src_file))

    print(f"--- {data_type.upper()} 处理完成 (新拷贝: {count} 张) ---")
    if missing_files:
        print(f"[WARNING] 缺失文件数: {len(missing_files)} (唯一文件数: {len(set(missing_files))})")

# ======================
# 封装后的 Main 函数
# ======================
def main(DATA_NAME, SOURCE_ROOT="../Other_data", TARGET_ROOT="single_data"):
    """
    只需要输入 DATA_NAME (如 'VisA')。
    SOURCE_ROOT 和 TARGET_ROOT 提供了合理的默认值。
    """
    SOURCE_ROOT = Path(SOURCE_ROOT)
    TARGET_DIR = Path(TARGET_ROOT)
    
    # # 1. 运行前全局清理
    # if TARGET_DIR.exists():
    #     print(f"[CLEAN] 正在清理旧目录: {TARGET_DIR}")
    #     shutil.rmtree(TARGET_DIR)
    # TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 自动构建 IOL 和 SOI 的 JSON 路径
    # 逻辑：SOURCE_ROOT / DATA_NAME / A_xxx_type_data / all_xxx_combined_metadata.json
    tasks = [
        ("iol", SOURCE_ROOT / DATA_NAME / "A_iol_type_data" / "all_iol_combined_metadata.json"),
        ("soi", SOURCE_ROOT / DATA_NAME / "A_soi_type_data" / "all_soi_combined_metadata.json")
    ]

    # 3. 循环执行任务
    for d_type, json_path in tasks:
        print(f"\n{'='*60}")
        print(f"[START] 任务类型: {d_type.upper()}")
        print(f"[PATH]  JSON位置: {json_path}")
        
        extract_original_images_from_json(
            source_data=SOURCE_ROOT, 
            json_path=json_path, 
            output_root=TARGET_DIR, 
            data_type=d_type
        )

# ======================
# 运行
# ======================
if __name__ == "__main__":
    # 现在 main 函数非常清爽，只传 DATA_NAME
    main("VisA")
    # 如果要换数据集，直接改这一行即可
    main("mvtec")
    main("BTech_Dataset_transformed")