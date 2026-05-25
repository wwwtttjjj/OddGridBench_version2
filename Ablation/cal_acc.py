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

    if not os.path.exists(folder_path):
        print(f"目录 {folder_path} 不存在！")
        return

    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            for item in data:
                model = item.get('model_name', 'Unknown_Model')
                dataset = item.get('dataset_name', 'Unknown_Dataset')
                key = (model, dataset)

                # 获取权重次数，兼容 original_count / count
                weight = item.get('original_count') or item.get('count') or 1

                predict = str(item.get('extract_answer', '')).strip().lower()
                gt = str(item.get('gt', '')).strip().lower()

                # 正负标签
                pos_labels = ['1', 'true', 'yes', 'correct']
                neg_labels = ['0', 'false', 'no', 'incorrect']

                is_pred_pos = predict in pos_labels
                is_pred_neg = predict in neg_labels

                is_gt_pos = gt in pos_labels
                is_gt_neg = gt in neg_labels

                # 如果 GT 本身非法，跳过，不计入 total
                if not is_gt_pos and not is_gt_neg:
                    print(f"非法 GT，已跳过: file={filename}, gt={gt}")
                    continue

                if key not in stats:
                    stats[key] = {
                        "tp": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": 0,
                        "total": 0,
                        "invalid_pred": 0
                    }

                # 只有合法 GT 才计入 total
                stats[key]["total"] += weight

                # -----------------------------
                # 核心逻辑：
                # 如果预测既不是 yes 也不是 no，则一律算错
                # -----------------------------
                if not is_pred_pos and not is_pred_neg:
                    stats[key]["invalid_pred"] += weight

                    if is_gt_pos:
                        # GT 是 yes，预测非法，相当于漏检
                        stats[key]["fn"] += weight
                    else:
                        # GT 是 no，预测非法，相当于误报
                        stats[key]["fp"] += weight

                elif is_pred_pos and is_gt_pos:
                    stats[key]["tp"] += weight

                elif is_pred_pos and is_gt_neg:
                    stats[key]["fp"] += weight

                elif is_pred_neg and is_gt_pos:
                    stats[key]["fn"] += weight

                elif is_pred_neg and is_gt_neg:
                    stats[key]["tn"] += weight

        except Exception as e:
            print(f"解析文件 {filename} 出错: {e}")

    if not stats:
        print(f"目录 {folder_path} 中没有有效统计结果。")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sorted_keys = sorted(
        stats.keys(),
        key=lambda x: (x[1], get_model_size(x[0]), x[0])
    )

    headers = [
        'dataset_name',
        'model_name',
        'Acc',
        'Precision',
        'Recall',
        'F1',
        'TP',
        'FP',
        'TN',
        'FN',
        'Invalid_Pred'
    ]

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for key in sorted_keys:
                model_name, dataset_name = key
                s = stats[key]

                tp = s["tp"]
                fp = s["fp"]
                tn = s["tn"]
                fn = s["fn"]
                total = s["total"]
                invalid_pred = s["invalid_pred"]

                acc = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                writer.writerow([
                    dataset_name,
                    model_name,
                    f"{acc:.2%}",
                    f"{precision:.2%}",
                    f"{recall:.2%}",
                    f"{f1:.2%}",
                    tp,
                    fp,
                    tn,
                    fn,
                    invalid_pred
                ])

        print(f"统计完成！已写入: {output_csv}")

    except Exception as e:
        print(f"写入出错: {e}")


# 示例调用
calculate_metrics_to_csv(
    './results_one-example',
    'final_results/model_performance_report_one-example.csv'
)

calculate_metrics_to_csv(
    './results_two-examples',
    'final_results/model_performance_report_two-examples.csv'
)

calculate_metrics_to_csv(
    './results_zero-shot',
    'final_results/model_performance_report_zero-shot.csv'
)
