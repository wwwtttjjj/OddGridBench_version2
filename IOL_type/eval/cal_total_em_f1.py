import json
import csv
import re
from pathlib import Path


# =========================
# Normalize GT: answer -> [(r,c), ...]
# =========================
def normalize_gt(answer):
    if not answer:
        return []
    out = []
    for item in answer:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
    return out


# =========================
# Normalize Pred: extract_answer -> [(r,c), ...] or None (invalid)
# extract_answer modes:
#   [] -> []
#   "" -> invalid
#   ["(2,3)"] -> [(2,3)]
# =========================
_COORD_STR_RE = re.compile(r'^\((\d+),(\d+)\)$')

def normalize_pred(extract_answer):
    if extract_answer == "" or extract_answer is None:
        return None  # invalid

    if isinstance(extract_answer, list):
        if len(extract_answer) == 0:
            return []

        coords = []
        for s in extract_answer:
            if not isinstance(s, str):
                return None
            s = s.strip()
            m = _COORD_STR_RE.match(s)
            if not m:
                return None
            coords.append((int(m.group(1)), int(m.group(2))))
        return coords

    return None


# =========================
# EM / F1 (set-level)
# =========================
def compute_em_f1(pred, gt):
    # 格式错误：直接错
    if pred is None:
        return 0, 0.0

    pred_set = set(pred)
    gt_set = set(gt)

    # 无异常：预测也无异常 -> 满分
    if len(gt_set) == 0 and len(pred_set) == 0:
        return 1, 1.0

    em = int(pred_set == gt_set)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return em, f1


# =========================
# Evaluate one json file
# =========================
def eval_json_file(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    em_sum = 0.0
    f1_sum = 0.0
    n = 0

    for sample in data:
        gt = normalize_gt(sample.get("answer", []))
        pred = normalize_pred(sample.get("extract_answer", ""))

        em, f1 = compute_em_f1(pred, gt)
        # print(f"ID: {sample.get('id')}, EM: {em}, F1: {f1:.3f}, GT: {gt}, Pred: {pred}")
        em_sum += em
        f1_sum += f1
        n += 1

    em_mean = em_sum / n if n else 0.0
    f1_mean = f1_sum / n if n else 0.0
    return em_mean, f1_mean

def model_size_key(model_name: str):
    """
    从模型名中提取规模数字，用于排序
    e.g. Qwen3-VL-32B-Instruct -> 32
    """
    m = re.search(r"-(\d+)B-", model_name)
    return int(m.group(1)) if m else float("inf")
# ========================

# =========================
# Evaluate a directory
# =========================
def eval_json_dir(json_dir: str, out_csv: str):
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        return

    results = {}

    for jp in json_files:
        em, f1 = eval_json_file(jp)
        results[jp.stem] = {"EM": em, "F1": f1}

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "EM", "F1"])

        # ✅ 按模型规模排序
        for model_name in sorted(results.keys(), key=model_size_key):
            m = results[model_name]
            writer.writerow([
                model_name,
                f"{m['EM'] * 100:.2f}",
                f"{m['F1'] * 100:.2f}",
            ])

    print(f"✅ Saved CSV to: {out_csv}")

def generate_combined_report(all_results, datasets, out_path):
    """
    生成总表：行是模型，列是 数据集_EM 和 数据集_F1
    all_results: { dataset_name: { model_name: {EM: x, F1: y} } }
    """
    # 1. 提取所有出现的模型名并按规模排序
    all_models = set()
    for ds in all_results:
        all_models.update(all_results[ds].keys())
    sorted_models = sorted(list(all_models), key=model_size_key)

    # 2. 准备表头
    header = ["model"]
    for ds in datasets:
        header.append(f"{ds}_EM")
        header.append(f"{ds}_F1")

    # 3. 写入 CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for model in sorted_models:
            row = [model]
            for ds in datasets:
                # 如果某个模型在某个数据集没结果，填空或0
                res = all_results.get(ds, {}).get(model, {"EM": 0, "F1": 0})
                row.append(f"{res['EM'] * 100:.2f}")
                row.append(f"{res['F1'] * 100:.2f}")
            writer.writerow(row)

    print(f"\n📊 总表已生成: {out_path}")

def save_markdown_report(all_results, datasets, out_path):
    """
    生成带双行表头视觉效果的 Markdown 总表并保存到文件
    """
    all_models = set()
    for ds in all_results:
        all_models.update(all_results[ds].keys())
    sorted_models = sorted(list(all_models), key=model_size_key)

    lines = []
    
    # 第一行表头：数据集名称（每个数据集占两列空间）
    # 使用 空白单元格 来模拟合并效果
    header_row1 = "| model | " + " | ".join([f" **{ds}** | " for ds in datasets]) + " |"
    
    # 第二行表头：具体的指标名称 (EM, F1)
    header_row2 = "| 指标 | " + " | ".join([" EM | F1 " for _ in datasets]) + " |"
    
    # 第三行：Markdown 必须的分割线
    separator = "| :--- | " + " | ".join([" :---: | :---: " for _ in datasets]) + " |"
    
    lines.append(header_row1)
    lines.append(header_row2)
    lines.append(separator)

    # 数据行
    for model in sorted_models:
        row = [f"**{model}**"]
        for ds in datasets:
            res = all_results.get(ds, {}).get(model, {"EM": 0, "F1": 0})
            row.append(f"{res['EM']*100:.2f}")
            row.append(f"{res['F1']*100:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    # 写入文件
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"✅ Markdown 总表已保存至: {out_path}")
# =========================
# 修改后的 main 逻辑
# =========================
if __name__ == "__main__":
    json_dirs = [
        "mnist_output", "hanzi_output", "icon_output", "BTech_output",
        "MVTEC_loco_output", "ELPV_output", "MVTEC_output", "VisA_output",
        "GOODADS_output","MPDD_output","RAD_output"
    ]

    out_root = Path("results_total")
    out_root.mkdir(parents=True, exist_ok=True)

    # 用于存放所有数据的汇总字典
    # 格式: { "mnist": { "Qwen-7B": {"EM": 0.9, "F1": 0.95}, ... }, ... }
    total_summary = {}

    for dir_name in json_dirs:
        input_path = Path(dir_name)
        if not input_path.exists():
            print(f"⚠️ 跳过目录: {dir_name} (不存在)")
            continue
            
        print(f"\n=== Evaluating {dir_name}/ ===")
        
        # 1. 计算当前目录下的所有模型
        json_files = sorted(input_path.glob("*.json"))
        current_ds_results = {}
        
        for jp in json_files:
            em, f1 = eval_json_file(jp)
            current_ds_results[jp.stem] = {"EM": em, "F1": f1}
        
        # 存入汇总大字典
        total_summary[dir_name] = current_ds_results

        # 2. 同时也保存你原来的单数据集 CSV
        out_csv = out_root / f"{dir_name}_results.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "EM", "F1"])
            for model_name in sorted(current_ds_results.keys(), key=model_size_key):
                m = current_ds_results[model_name]
                writer.writerow([model_name, f"{m['EM']*100:.2f}", f"{m['F1']*100:.2f}"])

    # =========================
    # 生成最终的总表
    # =========================
    combined_csv_path = out_root / "SUMMARY_TOTAL_REPORT.csv"
    generate_combined_report(total_summary, json_dirs, combined_csv_path)
    md_out_path = out_root / "SUMMARY_TOTAL_REPORT.md"
    save_markdown_report(total_summary, json_dirs, md_out_path)
