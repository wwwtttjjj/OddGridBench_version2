import json
from pathlib import Path

def count_combined_statistics(root_dir="."):
    root_path = Path(root_dir)
    total_iol = 0
    total_soi = 0
    
    # 定义任务类型及其对应的子路径
    task_map = {
        "IOL": "iol_test_data/iol_test_data.json",
        "SOI": "soi_test_data/soi_test_data.json"
    }

    # 打印表头
    header = f"{'Dataset Name':<25} | {'IOL Count':<12} | {'SOI Count':<12}"
    print(header)
    print("-" * len(header))

    # 获取所有子目录并排序
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    for subdir in subdirs:
        # 存储当前数据集的统计结果
        counts = {"IOL": 0, "SOI": 0}
        has_data = False

        for task_label, rel_path in task_map.items():
            target_json = subdir / rel_path
            
            if target_json.exists():
                has_data = True
                try:
                    with open(target_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 统计样本数量（假设是 List）
                        current_count = len(data) if isinstance(data, list) else 0
                        counts[task_label] = current_count
                        
                        # 累加到全局总计
                        if task_label == "IOL":
                            total_iol += current_count
                        else:
                            total_soi += current_count
                except Exception:
                    counts[task_label] = "Error"
            else:
                counts[task_label] = "-" # 表示未找到该类型的文件夹

        # 仅当该目录包含 IOL 或 SOI 数据时才打印
        if has_data:
            print(f"{subdir.name:<25} | {str(counts['IOL']):<12} | {str(counts['SOI']):<12}")

    # 打印总计栏
    print("-" * len(header))
    print(f"{'TOTAL SAMPLES':<25} | {total_iol:<12} | {total_soi:<12}")

if __name__ == "__main__":
    # 执行统计
    count_combined_statistics()