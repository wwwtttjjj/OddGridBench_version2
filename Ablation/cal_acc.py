import json
import os
import csv
import re

def get_model_size(model_name):
    match = re.search(r'(\d+)B', model_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def calculate_metrics_to_csv(folder_path, output_csv):
    stats = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list): data = [data]
                    
                    for item in data:
                        model = item.get('model_name', 'Unknown_Model')
                        dataset = item.get('dataset_name', 'Unknown_Dataset')
                        key = (model, dataset)
                        
                        # 获取权重次数 (默认为 1，以防字段缺失)
                        # 兼容你数据中的 'original_count' 或 'count' 字段
                        weight = item.get('original_count') or item.get('count') or 1
                        
                        if key not in stats:
                            stats[key] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}
                        
                        predict = str(item.get('extract_answer', '')).strip().lower()
                        gt = str(item.get('gt', '')).strip().lower()
                        pos_labels = ['1', 'true', 'yes', 'correct']
                        
                        # --- 核心改动：统计量乘以权重 ---
                        stats[key]["total"] += weight
                        
                        is_pred_pos = predict in pos_labels
                        is_gt_pos = gt in pos_labels

                        if is_pred_pos and is_gt_pos:
                            stats[key]["tp"] += weight
                        elif is_pred_pos and not is_gt_pos:
                            stats[key]["fp"] += weight
                        elif not is_pred_pos and is_gt_pos:
                            stats[key]["fn"] += weight
                        else:
                            stats[key]["tn"] += weight
                            
            except Exception as e:
                print(f"解析文件 {filename} 出错: {e}")

    if not stats: return

    sorted_keys = sorted(
        stats.keys(), 
        key=lambda x: (x[1], get_model_size(x[0]), x[0])
    )

    # 这里的 headers 保持不变
    headers = ['dataset_name', 'model_name', 'Acc', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'TN', 'FN']

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for key in sorted_keys:
                model_name, dataset_name = key
                s = stats[key]
                tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
                total = s["total"]

                acc = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                writer.writerow([
                    dataset_name, 
                    model_name, 
                    f"{acc:.2%}",
                    f"{precision:.2%}", 
                    f"{recall:.2%}", 
                    f"{f1:.2%}",
                    tp, fp, tn, fn
                ])
        print(f"统计完成！已按权重累加并写入: {output_csv}")
    except Exception as e:
        print(f"写入出错: {e}")

# calculate_metrics_to_csv('./results_old', 'model_performance_report_old.csv')
calculate_metrics_to_csv('./results', 'model_performance_report.csv')
