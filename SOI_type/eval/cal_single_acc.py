import json
import csv
import re
import os
from pathlib import Path
from tqdm import tqdm

# =========================
# 1. 标准化工具
# =========================
def normalize_gt(answer):
    """将 [9] 或 ["9"] 统一转换为整数集合 {9}"""
    if not answer:
        return set()
    return {int(x) for x in answer}

def normalize_pred(extract_answer):
    """
    支持格式: "image1,image2" 或 ["image1"] 或 []
    提取失败返回 None (视为格式错误)
    """
    if extract_answer is None:
        return None
    
    # 统一转为列表处理
    raw_input = extract_answer if isinstance(extract_answer, list) else str(extract_answer).split(",")
    
    indices = set()
    # 正则：匹配 image 后面跟着的数字，忽略大小写和空格
    soi_re = re.compile(r"image\s*(\d+)", re.IGNORECASE)

    for item in raw_input:
        item = str(item).strip()
        if not item: continue
        
        m = soi_re.search(item)
        if m:
            indices.add(int(m.group(1)))
        else:
            # 如果列表里有东西但匹配不到 imageN 格式，视为格式错误
            return None
            
    return indices

# =========================
# 2. 核心计算逻辑 (Cell/Unit Level)
# =========================
def compute_soi_metrics(pred_set, gt_set, total_count):
    """
    计算混淆矩阵：TP, FP, TN, FN
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # 如果格式错误 (None)，视为模型在该样本中没提供任何有效预测
    # 此时所有 GT 中的 Odd 变为 FN，所有 Normal 变为 TN
    current_pred = pred_set if pred_set is not None else set()

    for i in range(1, total_count + 1):
        is_pred_odd = i in current_pred
        is_gt_odd = i in gt_set
        
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
# 3. 评估逻辑
# =========================
def eval_json_file(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0
    
    for sample in data:
        gt = normalize_gt(sample.get("answer", []))
        pred = normalize_pred(sample.get("extract_answer", ""))
        
        # 自动获取该组图片的数量，默认 9
        total_imgs = sample.get("total_images") or sample.get("image_num") or 9
        if total_imgs != 8:
            continue
        tp, fp, tn, fn = compute_soi_metrics(pred, gt, int(total_imgs))
        t_tp += tp; t_fp += fp; t_tn += tn; t_fn += fn

    # 计算最终比率
    total_units = t_tp + t_fp + t_tn + t_fn
    acc = (t_tp + t_tn) / total_units if total_units > 0 else 0
    precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
    recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "count": len(data),
        "tp": t_tp, "fp": t_fp, "tn": t_tn, "fn": t_fn,
        "acc": acc, "p": precision, "r": recall, "f1": f1
    }

def run_soi_evaluation(dir_list):
    # 创建结果目录
    out_dir = Path("results_single")
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in dir_list:
        path_obj = Path(d)
        if not path_obj.exists():
            print(f"跳过不存在的目录: {d}")
            continue
            
        json_files = sorted(path_obj.glob("*.json"))
        csv_file = out_dir / f"{path_obj.name}_soi_metrics.csv"
        
        all_rows = []
        print(f"\n>>> 正在处理 SOI 目录: {d}")

        for jp in tqdm(json_files):
            m = eval_json_file(jp)
            all_rows.append({
                "model": jp.stem,
                "Total_Images": m['count'],
                "Acc": f"{m['acc']*100:.2f}%",
                "Precision": f"{m['p']*100:.2f}%",
                "Recall": f"{m['r']*100:.2f}%",
                "F1": f"{m['f1']*100:.2f}%",
                "TP": m['tp'], "FP": m['fp'], "TN": m['tn'], "FN": m['fn']
            })

        # 写入结果
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fields = ["model", "Total_Images", "Acc", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"✅ 完成！结果已保存至: {csv_file}")

# =========================
# 4. 入口
# =========================
if __name__ == "__main__":
    # 在此填入你的 SOI 结果目录名
    my_dirs = [
        "VisA_output", 
        "BTech_output",
        "MVTEC_output",
        "BTech_output",
        "ELPV_output",
        "hanzi_output",
        "icon_output",
        "mnist_output",
        "nanfang_output",
    ]
    run_soi_evaluation(my_dirs)