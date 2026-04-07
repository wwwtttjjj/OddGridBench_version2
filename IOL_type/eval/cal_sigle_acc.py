import json
import csv
import re
from pathlib import Path

# =========================
# 工具函数：标准化坐标
# =========================
def normalize_gt(answer):
    if not answer: return set()
    out = set()
    for item in answer:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.add((int(item[0]), int(item[1])))
    return out

def normalize_pred(extract_answer):
    if extract_answer == "" or extract_answer is None: return None
    if isinstance(extract_answer, list):
        if len(extract_answer) == 0: return set()
        coords = set()
        coord_re = re.compile(r'^\((\d+),(\d+)\)$')
        for s in extract_answer:
            if not isinstance(s, str): return None
            m = coord_re.match(s.strip())
            if not m: return None
            coords.add((int(m.group(1)), int(m.group(2))))
        return coords
    return None

# =========================
# 核心计算逻辑：基于单元格的二分类指标
# =========================
def compute_cell_confusion_matrix(pred_set, gt_set, grid_size):
    """
    计算单张大图中所有单元格的 TP, FP, TN, FN
    """
    rows, cols = grid_size
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # 如果格式错误，该图所有 Odd 单元格算 FN，所有 Normal 单元格算 TN (假设模型什么都没说)
    # 或者根据严格标准，格式错误直接让该图贡献 0 TP
    is_invalid = (pred_set is None)
    current_pred = pred_set if not is_invalid else set()

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            cell = (r, c)
            is_pred_odd = cell in current_pred
            is_gt_odd = cell in gt_set
            
            if is_gt_odd and is_pred_odd:
                tp += 1
            elif is_gt_odd and not is_pred_odd:
                fn += 1
            elif not is_gt_odd and is_pred_odd:
                fp += 1
            else:
                tn += 1
                
    return tp, fp, tn, fn

# =========================
# 文件与目录处理
# =========================
def eval_json_file(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    img_count = len(data)

    for sample in data:
        gt_set = normalize_gt(sample.get("answer", []))
        pred_set = normalize_pred(sample.get("extract_answer", ""))
        
        # --- 修改部分：支持字符串格式的 grid_size ---
        grid_size = sample.get("grid_size", [3, 3])
        if isinstance(grid_size, str):
            try:
                # 将 "[3, 3]" 转换为 [3, 3]
                grid_size = json.loads(grid_size)
            except:
                # 如果解析失败，回退到默认值
                grid_size = [3, 3]
        # ------------------------------------------

        tp, fp, tn, fn = compute_cell_confusion_matrix(pred_set, gt_set, grid_size)
        
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    # ... 后面的计算逻辑保持不变 ...
    total_cells = total_tp + total_fp + total_tn + total_fn
    acc = (total_tp + total_tn) / total_cells if total_cells > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "images": img_count,
        "total_cells": total_cells,
        "tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn,
        "acc": acc, "precision": precision, "recall": recall, "f1": f1
    }

def eval_all(json_dir_list, out_csv):
    results = []

    for dir_path in json_dir_list:
        path_obj = Path(dir_path)
        json_files = sorted(path_obj.glob("*.json"))
        
        for jp in json_files:
            m = eval_json_file(jp)
            
            print(f"Model: {jp.stem:30}")
            print(f" ├─ Cells: {m['total_cells']} (Odd GT: {m['tp']+m['fn']} | Normal GT: {m['tn']+m['fp']})")
            print(f" └─ Metrics: Acc: {m['acc']*100:.2f}% | P: {m['precision']*100:.2f}% | R: {m['recall']*100:.2f}% | F1: {m['f1']*100:.2f}%")
            print(f" └─ Detail: TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}\n")

            results.append({
                "model": jp.stem,
                "Images": m['images'],
                "Acc": f"{m['acc']*100:.2f}%",
                "Precision": f"{m['precision']*100:.2f}%",
                "Recall": f"{m['recall']*100:.2f}%",
                "F1": f"{m['f1']*100:.2f}%",
                "TP": m['tp'],
                "FP": m['fp'],
                "TN": m['tn'],
                "FN": m['fn']
            })

    # 写入 CSV
    fieldnames = ["model", "Images", "Acc", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# =========================
# 执行部分
# =========================
if __name__ == "__main__":
    # 1. 定义你要处理的目录列表
    target_dirs = ["VisA_output", "MVTEC_output", "BTech_output"] 
    
    # 2. 定义存放结果的总文件夹
    out_root = Path("results_single")
    out_root.mkdir(parents=True, exist_ok=True) # 如果不存在则创建

    print(f"=== 开始全量指标评估 (Cell-level) ===\n")

    for json_dir in target_dirs:
        # 去掉路径末尾的斜杠，防止生成的文件名带路径
        folder_name = json_dir.rstrip("/").replace("/", "_")
        
        # 3. 根据目录名动态生成输出文件名
        # 例如：results/VisA_output_cell_metrics.csv
        out_csv_path = out_root / f"{folder_name}_cell_metrics.csv"
        
        print(f"--- 正在处理目录: {json_dir} ---")
        eval_all([json_dir], str(out_csv_path))
        print(f"✅ 该目录评估完成，结果保存至: {out_csv_path}\n")

    print(f"🚀 所有任务处理完毕，请在 '{out_root}' 文件夹下查看结果。")