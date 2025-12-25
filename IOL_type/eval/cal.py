import json
import csv
import re
from pathlib import Path
import argparse


# =========================
# Parse extract_answer
# =========================
def parse_extract_answer(s: str):
    """
    "(5,1),(7,4)" -> [(5,1),(7,4)]
    ""            -> []
    None          -> []
    """
    if not s:
        return []
    matches = re.findall(r"\((\d+),(\d+)\)", s)
    return [(int(r), int(c)) for r, c in matches]


# =========================
# EM / F1 for one sample
# =========================
def compute_em_f1(pred, gt):
    pred_set = set(map(tuple, pred))
    gt_set = set(map(tuple, gt))
    # print(f"[DEBUG] pred_set: {pred_set}, gt_set: {gt_set}")
    # Exact Match
    em = int(pred_set == gt_set)

    # Set-level overlap
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return em, f1


# =========================
# Evaluate one json file
# =========================
def eval_json_file(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    em_list = []
    f1_list = []

    for sample in data:
        # ✅ 正确字段对齐
        pred = parse_extract_answer(sample.get("extract_answer", ""))
        gt = sample.get("answer", [])

        em, f1 = compute_em_f1(pred, gt)
        if em == 0:
            print(sample.get("class", []))
        em_list.append(em)
        f1_list.append(f1)
        # print(f"[DEBUG] id={sample.get('id')} EM={em} F1={f1:.4f}")

    em_mean = sum(em_list) / len(em_list) if em_list else 0.0
    f1_mean = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return em_mean, f1_mean


# =========================
# Evaluate a directory
# =========================
def eval_json_dir(json_dir: str, out_csv: str):
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        return
        # raise RuntimeError(f"No json files found in {json_dir}")

    results = {}

    for jp in json_files:
        em, f1 = eval_json_file(jp)
        results[jp.stem] = {   # 👈 用文件名（不含 .json）作为模型名
            "EM": em,
            "F1": f1,
        }
        # print(f"[OK] {jp.name}: EM={em:.4f}, F1={f1:.4f}")

    # =========================
    # Write CSV (models as rows, metrics as columns)
    # =========================
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # header: model, EM, F1
        header = ["model", "EM", "F1"]
        writer.writerow(header)

        # each model is one row
        for model_name, metric_dict in results.items():
            row = [
                model_name,
                f"{metric_dict['EM']:.6f}",
                f"{metric_dict['F1']:.6f}",
            ]
            writer.writerow(row)

    print(f"\n✅ Saved CSV to: {out_csv}")


# =========================
# CLI
# =========================
if __name__ == "__main__":

    json_dirs = [
        "mnist_output/",
        "hanzi_output/",
        "icon_output/",
    ]

    out_root = Path("results_emf1")
    out_root.mkdir(parents=True, exist_ok=True)

    for json_dir in json_dirs:
        json_dir = json_dir.rstrip("/")  # 防止多 /
        out_csv = out_root / f"{json_dir}_results.csv"

        print(f"\n=== Evaluating {json_dir}/ ===")
        eval_json_dir(f"{json_dir}/", str(out_csv))

