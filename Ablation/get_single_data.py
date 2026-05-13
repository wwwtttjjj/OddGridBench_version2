import json
import shutil
import os
from pathlib import Path

# ======================
# 配置参数
# ======================
ONLY_ANOMALY = False    
CLEAR_TARGET = False   

def extract_original_images_and_record(source_data, json_path, output_root, data_type):
    """
    处理单个 JSON 任务，物理拷贝图片并返回该任务对应的 metadata 字典
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[SKIP] 文件不存在: {json_path}")
        return None, 0, 0 

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_root = Path(output_root)
    count = 0          
    anomaly_count = 0  
    
    # 局部字典：只记录当前这个 JSON 文件的信息
    local_metadata = {}

    print(f"[INFO] 正在处理: {data_type.upper()} | 源文件: {json_path.name}")

    for entry in data:
        dataset_name = entry.get("dataset_name", "")
        image_dir = entry.get("image_dir", "")
        entry_category = entry.get("category", "")  # train/test
        
        # 业务分类 (如 "capsule")
        biz_category = entry.get("category", "default") 
        base_origin_path = Path(source_data) / dataset_name / image_dir / biz_category

        cells = entry.get("source_cells") or entry.get("source_images_details") or []
        
        for cell in cells:
            label = str(cell.get("label", "")).lower() 
            orig_name = cell.get("original_filename") or cell.get("original_name")
            if not orig_name: continue

            is_anomaly = (label != "normal")
            if ONLY_ANOMALY and not is_anomaly:
                continue
            
            label_dir = "Anomaly" if is_anomaly else "Normal"
            src_file = base_origin_path / label_dir / orig_name
            
            # --- 物理存储路径 (保持简洁) ---
            dst_dir = output_root / dataset_name / biz_category / label_dir
            dst_file = dst_dir / orig_name
            
            # 使用相对于 output_root 的路径作为 Key
            meta_key = str(dst_file.relative_to(output_root))
            
            if meta_key not in local_metadata:
                local_metadata[meta_key] = {
                    "filename": orig_name,
                    "count": 0,
                    "resize_scale": [],
                    "category": entry_category,
                    "dataset_name": dataset_name,
                    "image_dir": image_dir,
                    "label": label,
                    "physical_path": str(dst_file)
                }
            
            # 记录信息
            local_metadata[meta_key]["count"] += 1
            scale = cell.get("resize_scale", 1.0)
            local_metadata[meta_key]["resize_scale"].append(scale)

            # --- 物理拷贝 ---
            print(f"   >> 处理图片: {src_file} | 目标: {dst_file} | 异常: {is_anomaly}")
            if src_file.exists():
                if not dst_file.exists():
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    count += 1
                    if is_anomaly:
                        anomaly_count += 1
            
    return local_metadata, count, anomaly_count

def main(DATA_NAME, SOURCE_ROOT="../Test_data", TARGET_ROOT="single_data"):
    SOURCE_ROOT = Path(SOURCE_ROOT)
    TARGET_DIR = Path(TARGET_ROOT)
    
    # 定义任务：(类型, 原始JSON路径)
    tasks = [
        ("iol", SOURCE_ROOT / DATA_NAME / "A_iol_type_data" / "all_iol_combined_metadata.json"),
    ]
    print(tasks)
    for d_type, json_path in tasks:
        # 1. 提取并拷贝
        meta, c, ac = extract_original_images_and_record(
            source_data=SOURCE_ROOT, 
            json_path=json_path, 
            output_root=TARGET_DIR, 
            data_type=d_type
        )
        # 2. 如果有数据，则保存对应的 JSON 文件
        if meta:
            # 命名规则：dataset_name_type.json (例如 VisA_iol.json)
            out_json_name = f"{DATA_NAME}_{d_type}.json"
            out_json_path = TARGET_DIR / out_json_name
            print(f"   >> 处理完成: {out_json_name} | 拷贝了 {c} 张图片，其中异常 {ac} 张")
            with open(out_json_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
            print(f"   >> 已生成 JSON: {out_json_name} (拷贝 {c} 张)")

if __name__ == "__main__":
    datasets = ["VisA", "BTech_Dataset_transformed", "mvtec", "ELPV","MPDD","RAD", "GOODADS"]
    TARGET_PATH = Path("single_data")
    
    # 初始化清空目录
    if CLEAR_TARGET and TARGET_PATH.exists():
        print(f"清空目标目录: {TARGET_PATH}")
        shutil.rmtree(TARGET_PATH)
    
    TARGET_PATH.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        main(ds)
        