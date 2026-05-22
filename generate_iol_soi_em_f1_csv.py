
#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_ROWS = [
    ("InternVL3.5-2B", "InternVL3_5-2B"),
    ("InternVL3.5-4B", "InternVL3_5-4B"),
    ("InternVL3.5-8B", "InternVL3_5-8B"),
    ("InternVL3.5-38B", "InternVL3_5-38B"),
    ("Qwen3-VL-2B", "Qwen3-VL-2B-Instruct"),
    ("Qwen3-VL-4B", "Qwen3-VL-4B-Instruct"),
    ("Qwen3-VL-8B", "Qwen3-VL-8B-Instruct"),
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    ("Qwen3.5-2B", "Qwen3.5-2B"),
    ("Qwen3.5-4B", "Qwen3.5-4B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
    ("Gemma4-E2B-it", "gemma-4-E2B-it"),
    ("Gemma4-E4B-it", "gemma-4-E4B-it"),
    ("Gemma4-26B-A4B-it", "gemma-4-26B-A4B-it"),
    ("Gemma4-31B-it", "gemma-4-31B-it"),
    ("Ours-DAPO-EM-2B", "Qwen3_vl_2B_TOTAL_EM_dapo_step_200"),
    ("Ours-DAPO-EM-4B", "Qwen3_vl_4B_TOTAL_EM_dapo_step_200"),
]

SYNTHETIC = [("ICON", ["icon_output"]), ("MNIST", ["mnist_output"]), ("Hanzi", ["hanzi_output"])]
FIXED = [("MVTEC", ["MVTEC_output"]), ("VISA", ["VisA_output"]), ("Btec", ["BTech_output"])]
VIEW_VARIANT = [("MPDD/RAD", ["MPDD_output", "RAD_output"]), ("GOODADS", ["GOODADS_output"])]
DATASET_COLUMNS = SYNTHETIC + [("Total", [name for _, names in SYNTHETIC for name in names])] + FIXED + [("Total", [name for _, names in FIXED for name in names])] + VIEW_VARIANT + [("Total", [name for _, names in VIEW_VARIANT for name in names])]

IOL_COORD_RE = re.compile(r"^\((\d+),(\d+)\)$")
SOI_IMAGE_RE = re.compile(r"^image(\d+)$")


def normalize_iol_gt(answer):
    out = []
    for item in answer or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
    return out


def normalize_iol_pred(extract_answer):
    if extract_answer == "" or extract_answer is None or not isinstance(extract_answer, list):
        return None
    out = []
    for item in extract_answer:
        if not isinstance(item, str):
            return None
        match = IOL_COORD_RE.match(item.strip())
        if not match:
            return None
        out.append((int(match.group(1)), int(match.group(2))))
    return out


def normalize_soi_gt(answer):
    return [int(x) for x in (answer or [])]


def normalize_soi_pred(extract_answer):
    if extract_answer is None:
        return None
    if extract_answer == []:
        return []
    if isinstance(extract_answer, list):
        parts = extract_answer
    elif isinstance(extract_answer, str) and extract_answer.strip():
        parts = [p.strip() for p in extract_answer.split(",")]
    else:
        return None
    out = []
    for item in parts:
        if not isinstance(item, str):
            return None
        match = SOI_IMAGE_RE.match(item.strip())
        if not match:
            return None
        out.append(int(match.group(1)))
    return out


def compute_em_f1(pred, gt):
    if pred is None:
        return 0.0, 0.0
    pred_set = set(pred)
    gt_set = set(gt)
    if not pred_set and not gt_set:
        return 1.0, 1.0
    em = 1.0 if pred_set == gt_set else 0.0
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return em, f1


def eval_json_file(path, task):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] skip {path}: {exc}")
        return None
    if not isinstance(data, list):
        return None
    em_sum = 0.0
    f1_sum = 0.0
    n = 0
    for sample in data:
        if task == "iol":
            gt = normalize_iol_gt(sample.get("answer", []))
            pred = normalize_iol_pred(sample.get("extract_answer", ""))
        else:
            gt = normalize_soi_gt(sample.get("answer", []))
            pred = normalize_soi_pred(sample.get("extract_answer", ""))
        em, f1 = compute_em_f1(pred, gt)
        em_sum += em
        f1_sum += f1
        n += 1
    if not n:
        return None
    return {"EM": em_sum / n * 100.0, "F1": f1_sum / n * 100.0}


def collect_results(eval_dir, task):
    results = {}
    eval_dir = Path(eval_dir)
    for dataset_dir in sorted(eval_dir.glob("*_output")):
        if not dataset_dir.is_dir():
            continue
        dataset_results = {}
        for json_path in sorted(dataset_dir.glob("*.json")):
            metrics = eval_json_file(json_path, task)
            if metrics is not None:
                dataset_results[json_path.stem] = metrics
        if dataset_results:
            results[dataset_dir.name] = dataset_results
    return results


def average_metrics(items):
    vals = [x for x in items if x is not None]
    if not vals:
        return None
    return {"EM": sum(x["EM"] for x in vals) / len(vals), "F1": sum(x["F1"] for x in vals) / len(vals)}


def metric_for_model(results, model_key, dataset_names):
    vals = [results.get(ds, {}).get(model_key) for ds in dataset_names]
    return average_metrics(vals)


def metric_cells(metric):
    if metric is None:
        return ["", ""]
    return [f"{metric['EM']:.1f}", f"{metric['F1']:.1f}"]


def build_csv_rows(results):
    header1 = ["Models"]
    header1 += ["Synthetic Homogeneous Scenario"] + [""] * (2 * (len(SYNTHETIC) + 1) - 1)
    header1 += ["Fixed-View Industrial Scenario"] + [""] * (2 * (len(FIXED) + 1) - 1)
    header1 += ["View-Variant Industrial Scenario"] + [""] * (2 * (len(VIEW_VARIANT) + 1) - 1)
    header2 = [""]
    for label, _ in DATASET_COLUMNS:
        header2.extend([label, ""])
    header3 = [""]
    for _ in DATASET_COLUMNS:
        header3.extend(["EM", "F1"])

    rows = [header1, header2, header3]
    for display_name, model_key in MODEL_ROWS:
        row = [display_name]
        for _, dataset_names in DATASET_COLUMNS:
            row.extend(metric_cells(metric_for_model(results, model_key, dataset_names)))
        rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate separate IOL and SOI merged EM/F1 CSV tables.")
    parser.add_argument("--iol-dir", default=str(ROOT / "IOL_type" / "eval"))
    parser.add_argument("--soi-dir", default=str(ROOT / "SOI_type" / "eval"))
    parser.add_argument("--out-dir", default=str(ROOT / "merged_reports"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    iol_rows = build_csv_rows(collect_results(args.iol_dir, "iol"))
    soi_rows = build_csv_rows(collect_results(args.soi_dir, "soi"))
    iol_out = out_dir / "IOL_merged_em_f1.csv"
    soi_out = out_dir / "SOI_merged_em_f1.csv"
    write_csv(iol_out, iol_rows)
    write_csv(soi_out, soi_rows)
    print(f"Saved: {iol_out}")
    print(f"Saved: {soi_out}")


if __name__ == "__main__":
    main()
