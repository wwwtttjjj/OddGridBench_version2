import json
import csv
import re
from pathlib import Path


# =========================
# Parse extract_answer
# =========================
def parse_extract_answer(s: str):
    """
    "image3" -> [3]
    ""       -> []
    None     -> []
    """
    if not s:
        return []
    m = re.search(r"image(\d+)", s)
    if not m:
        return []
    return [int(m.group(1))]


# =========================
# EM / F1 for one sample
# =========================
def compute_em_f1(pred, gt):
    """
    pred: List[int]
    gt:   List[int]
    """
    pred_set = set(pred)
    gt_set = set(gt)

    # Exact Match (zero-tolerance set match)
    em = int(pred_set == gt_set)

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
        pred = parse_extract_answer(sample.get("extract_answer", ""))
        gt = sample.get("answer", [])

        em, f1 = compute_em_f1(pred, gt)

        if em == 0:
            print(sample.get("class", []))

        em_list.append(em)
        f1_list.append(f1)

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

    results = {}

    for jp in json_files:
        em, f1 = eval_json_file(jp)
        results[jp.stem] = {
            "EM": em,
            "F1": f1,
        }

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "EM", "F1"])

        for model_name, metric_dict in results.items():
            writer.writerow([
                model_name,
                f"{metric_dict['EM']:.6f}",
                f"{metric_dict['F1']:.6f}",
            ])

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
        json_dir = json_dir.rstrip("/")
        out_csv = out_root / f"{json_dir}_results.csv"

        print(f"\n=== Evaluating {json_dir}/ ===")
        eval_json_dir(f"{json_dir}/", str(out_csv))
