#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(__file__).resolve().parent

MODEL_ROWS = [
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    ("Qwen3.5-VL-27B", "Qwen3.5-27B"),
    ("Qwen3-VL-4B", "Qwen3-VL-4B-Instruct"),
    ("Qwen3-VL-4B-DAPO-EM", "Qwen3_vl_4B_TOTAL_EM_dapo_step_200"),
    ("Qwen3-VL-4B-DAPO-F1", "Qwen3_vl_4B_TOTAL_F1_dapo_step_200"),
]

BASE_SCENARIO_TOTALS = [
    (
        "Synthetic Scenario",
        ["icon_output", "mnist_output", "hanzi_output"],
    ),
    (
        "Fixed-View Industrial Scenario",
        ["MVTEC_output", "VisA_output", "BTech_output"],
    ),
    (
        "View-Variant Industrial Scenario",
        ["MPDD_output", "RAD_output", "GOODADS_output"],
    ),
]
ALL_DATASETS = [dataset for _, datasets in BASE_SCENARIO_TOTALS for dataset in datasets]
SCENARIO_TOTALS = BASE_SCENARIO_TOTALS + [("Total", ALL_DATASETS)]

ERROR_TYPES = [
    ("no_overlap", "Semantic Misidentification"),
    ("partial", "Incomplete or Coarse-grained Prediction"),
    ("extract_failed", "Output Format Violation"),
]

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


def classify_error(pred, gt):
    if pred is None:
        return "extract_failed"

    pred_set = set(pred)
    gt_set = set(gt)
    if pred_set == gt_set:
        return "exact"
    if not (pred_set & gt_set):
        return "no_overlap"
    return "partial"


def empty_stats():
    return {"n": 0, "counts": Counter()}


def merge_stats(items):
    merged = empty_stats()
    for item in items:
        if not item:
            continue
        merged["n"] += item["n"]
        merged["counts"].update(item["counts"])
    return merged if merged["n"] else None


def eval_json_file(path, task):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] skip {path}: {exc}")
        return None
    if not isinstance(data, list):
        return None

    stats = empty_stats()
    for sample in data:
        if task == "iol":
            gt = normalize_iol_gt(sample.get("answer", []))
            pred = normalize_iol_pred(sample.get("extract_answer", ""))
        else:
            gt = normalize_soi_gt(sample.get("answer", []))
            pred = normalize_soi_pred(sample.get("extract_answer", ""))
        stats["counts"][classify_error(pred, gt)] += 1
        stats["n"] += 1
    return stats if stats["n"] else None


def collect_results(eval_dir, task):
    results = {}
    for dataset_dir in sorted(Path(eval_dir).glob("*_output")):
        if not dataset_dir.is_dir():
            continue
        dataset_results = {}
        for json_path in sorted(dataset_dir.glob("*.json")):
            stats = eval_json_file(json_path, task)
            if stats is not None:
                dataset_results[json_path.stem] = stats
        if dataset_results:
            results[dataset_dir.name] = dataset_results
    return results


def stats_for_model(results, model_key, dataset_names):
    return merge_stats(results.get(dataset, {}).get(model_key) for dataset in dataset_names)


def error_total(stats):
    counts = stats["counts"]
    return counts["no_overlap"] + counts["partial"] + counts["extract_failed"]


def value_cells(stats, value_mode, denominator):
    if stats is None:
        return [""] * len(ERROR_TYPES)

    counts = stats["counts"]
    if value_mode == "count":
        return [str(counts[key]) for key, _ in ERROR_TYPES]

    total = stats["n"] if denominator == "all" else error_total(stats)
    if total == 0:
        return ["0.0"] * len(ERROR_TYPES)
    return [f"{counts[key] / total * 100.0:.1f}" for key, _ in ERROR_TYPES]


def build_csv_rows(results, value_mode, denominator):
    metric_labels = [label for _, label in ERROR_TYPES]
    header1 = ["Models"]
    header2 = [""]
    for scenario, _ in SCENARIO_TOTALS:
        header1.extend([scenario] + [""] * (len(metric_labels) - 1))
        header2.extend(metric_labels)

    rows = [header1, header2]
    for display_name, model_key in MODEL_ROWS:
        row = [display_name]
        for _, dataset_names in SCENARIO_TOTALS:
            stats = stats_for_model(results, model_key, dataset_names)
            row.extend(value_cells(stats, value_mode, denominator))
        rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)


def build_final_total_rows(task_results, value_mode, denominator):
    rows = [["Task", "Models"] + [label for _, label in ERROR_TYPES]]
    for task_name, results in task_results:
        for display_name, model_key in MODEL_ROWS:
            stats = stats_for_model(results, model_key, ALL_DATASETS)
            rows.append([task_name, display_name] + value_cells(stats, value_mode, denominator))
    return rows


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def make_latex_table(rows, task_name, value_mode, denominator):
    col_count = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    metric_desc = "error-type distributions" if denominator == "error" else "all-sample error rates"
    if value_mode == "count":
        metric_desc = "error counts"

    align = "l" + "c" * (col_count - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{latex_escape(task_name)} total-only {latex_escape(metric_desc)}.}}",
        rf"\label{{tab:{task_name.lower()}_total_error_analysis}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    header1 = [r"\multirow{2}{*}{Models}"]
    for scenario, _ in SCENARIO_TOTALS:
        header1.append(rf"\multicolumn{{3}}{{c}}{{{latex_escape(scenario)}}}")
    lines.append(" & ".join(header1) + r" \\")
    cmidrules = []
    for idx in range(len(SCENARIO_TOTALS)):
        start_col = 2 + idx * len(ERROR_TYPES)
        end_col = start_col + len(ERROR_TYPES) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
    lines.append(" ".join(cmidrules))
    lines.append(" & ".join([""] + [latex_escape(label) for _, label in ERROR_TYPES] * len(SCENARIO_TOTALS)) + r" \\")
    lines.append(r"\midrule")

    for row in rows[2:]:
        cells = [latex_escape(row[0])]
        cells.extend(latex_escape(cell) if cell else "--" for cell in row[1:col_count])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_latex(path, rows, task_name, value_mode, denominator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_latex_table(rows, task_name, value_mode, denominator), encoding="utf-8")


def generate_for_task(task_name, eval_dir, task, out_dir, value_mode, denominator):
    results = collect_results(eval_dir, task)
    rows = build_csv_rows(results, value_mode, denominator)
    value_suffix = f"{value_mode}_{denominator}" if value_mode == "percent" else value_mode
    csv_path = out_dir / f"{task_name}_total_error_analysis_{value_suffix}.csv"
    tex_path = out_dir / "latex_tables" / f"{task_name}_total_error_analysis_{value_suffix}.tex"
    write_csv(csv_path, rows)
    write_latex(tex_path, rows, task_name, value_mode, denominator)
    return csv_path, tex_path, results


def main():
    parser = argparse.ArgumentParser(description="Generate total-only IOL/SOI error analysis tables.")
    parser.add_argument("--iol-dir", default=str(ROOT / "IOL_type" / "eval"))
    parser.add_argument("--soi-dir", default=str(ROOT / "SOI_type" / "eval"))
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "reports_total"))
    parser.add_argument("--value-mode", choices=["percent", "count"], default="percent")
    parser.add_argument(
        "--denominator",
        choices=["error", "all"],
        default="error",
        help="Percent denominator: error samples only, or all samples. Ignored for count mode.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    iol_csv, iol_tex, iol_results = generate_for_task("IOL", args.iol_dir, "iol", out_dir, args.value_mode, args.denominator)
    soi_csv, soi_tex, soi_results = generate_for_task("SOI", args.soi_dir, "soi", out_dir, args.value_mode, args.denominator)

    outputs = [iol_csv, iol_tex, soi_csv, soi_tex]
    value_suffix = f"{args.value_mode}_{args.denominator}" if args.value_mode == "percent" else args.value_mode
    final_total_path = out_dir / f"IOL_SOI_final_total_error_analysis_{value_suffix}.csv"
    final_total_rows = build_final_total_rows([("IOL", iol_results), ("SOI", soi_results)], args.value_mode, args.denominator)
    write_csv(final_total_path, final_total_rows)
    outputs.append(final_total_path)

    tex_outputs = [path for path in outputs if path.suffix == ".tex"]
    combined_path = out_dir / "latex_tables" / f"IOL_SOI_total_error_analysis_{value_suffix}.tex"
    combined_path.write_text("\n".join(path.read_text(encoding="utf-8") for path in tex_outputs), encoding="utf-8")
    outputs.append(combined_path)

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
