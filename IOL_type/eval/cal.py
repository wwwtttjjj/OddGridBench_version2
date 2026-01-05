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
        print(f"ID: {sample.get('id')}, EM: {em}, F1: {f1:.3f}, GT: {gt}, Pred: {pred}")
        em_sum += em
        f1_sum += f1
        n += 1

    em_mean = em_sum / n if n else 0.0
    f1_mean = f1_sum / n if n else 0.0
    return em_mean, f1_mean


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
        for model_name, metric in results.items():
            writer.writerow([
                model_name,
                f"{metric['EM'] * 100:.1f}",
                f"{metric['F1'] * 100:.1f}",
            ])

    print(f"✅ Saved CSV to: {out_csv}")


# =========================
# main
# =========================
if __name__ == "__main__":
    json_dirs = [
        # "mnist_output/",
        # "hanzi_output/",
        # "icon_output/",
        "BTech_output/",
        "MVTEC_loco_output/",
        "MVTEC_output/",
        "VisA_output/",
        
    ]

    out_root = Path("results_emf1")
    out_root.mkdir(parents=True, exist_ok=True)

    for json_dir in json_dirs:
        json_dir = json_dir.rstrip("/")
        out_csv = out_root / f"{json_dir}_results.csv"
        print(f"\n=== Evaluating {json_dir}/ ===")
        eval_json_dir(f"{json_dir}/", str(out_csv))
