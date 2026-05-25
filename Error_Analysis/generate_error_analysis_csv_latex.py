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
    ("InternVL3.5-2B", "InternVL3_5-2B"),
    ("InternVL3.5-4B", "InternVL3_5-4B"),
    ("InternVL3.5-8B", "InternVL3_5-8B"),
    ("InternVL3.5-38B", "InternVL3_5-38B"),
    ("Qwen3-VL-2B", "Qwen3-VL-2B-Instruct"),
    ("Qwen3-VL-4B", "Qwen3-VL-4B-Instruct"),
    ("Qwen3-VL-8B", "Qwen3-VL-8B-Instruct"),
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    ("Gemma4-E2B-it", "gemma-4-E2B-it"),
    ("Gemma4-E4B-it", "gemma-4-E4B-it"),
    ("Gemma4-26B-A4B-it", "gemma-4-26B-A4B-it"),
    ("Gemma4-31B-it", "gemma-4-31B-it"),
    ("Qwen3.5-2B", "Qwen3.5-2B"),
    ("Qwen3.5-4B", "Qwen3.5-4B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
]

SYNTHETIC = [("ICON", ["icon_output"]), ("MNIST", ["mnist_output"]), ("Hanzi", ["hanzi_output"])]
FIXED = [("MVTEC", ["MVTEC_output"]), ("VISA", ["VisA_output"]), ("Btec", ["BTech_output"])]
VIEW_VARIANT = [("MPDD/RAD", ["MPDD_output", "RAD_output"]), ("GOODADS", ["GOODADS_output"])]

DATASET_DISPLAY_NAMES = {
    "ICON": "IconSim",
    "icon": "IconSim",
    "MNIST": "DigitSim",
    "mnist": "DigitSim",
    "Hanzi": "HanziSim",
    "hanzi": "HanziSim",
    "MVTEC": "MVTec AD",
    "MVTec": "MVTec AD",
    "mvtec": "MVTec AD",
    "MVTEC_AD2": "MVTec AD 2",
    "MVTEC AD 2": "MVTec AD 2",
    "mvtec_ad2": "MVTec AD 2",
    "VISA": "VisA",
    "visa": "VisA",
    "Btec": "BTAD",
    "BTech": "BTAD",
    "BTech_Dataset_transformed": "BTAD",
    "GOODADS": "GoodsAD",
    "GoodAD": "GoodsAD",
    "goodsad": "GoodsAD",
}
DATASET_COLUMNS = (
    SYNTHETIC
    + [("Total", [name for _, names in SYNTHETIC for name in names])]
    + FIXED
    + [("Total", [name for _, names in FIXED for name in names])]
    + VIEW_VARIANT
    + [("Total", [name for _, names in VIEW_VARIANT for name in names])]
)

IOL_COORD_RE = re.compile(r"^\((\d+),(\d+)\)$")
SOI_IMAGE_RE = re.compile(r"^image(\d+)$")

BROAD_ERROR_TYPES = [
    ("partial", "Partial"),
    ("no_overlap", "NoOverlap"),
    ("extract_failed", "ExtractFail"),
]
DETAIL_ERROR_TYPES = [
    ("partial_under", "Under"),
    ("partial_over", "Over"),
    ("partial_mixed", "Mixed"),
    ("no_overlap", "NoOverlap"),
    ("extract_failed", "ExtractFail"),
]


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

    overlap = pred_set & gt_set
    if not overlap:
        return "no_overlap"
    if pred_set < gt_set:
        return "partial_under"
    if pred_set > gt_set:
        return "partial_over"
    return "partial_mixed"


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
    eval_dir = Path(eval_dir)
    for dataset_dir in sorted(eval_dir.glob("*_output")):
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
    return merge_stats(results.get(ds, {}).get(model_key) for ds in dataset_names)


def broad_counts(stats):
    counts = stats["counts"]
    return {
        "partial": counts["partial_under"] + counts["partial_over"] + counts["partial_mixed"],
        "no_overlap": counts["no_overlap"],
        "extract_failed": counts["extract_failed"],
    }


def error_total(stats, error_types):
    if error_types == BROAD_ERROR_TYPES:
        return sum(broad_counts(stats).values())
    counts = stats["counts"]
    return sum(counts[key] for key, _ in error_types)


def percent_cells(stats, error_types, denominator):
    if stats is None:
        return [""] * len(error_types)
    if error_types == BROAD_ERROR_TYPES:
        counts = broad_counts(stats)
    else:
        counts = stats["counts"]
    total = stats["n"] if denominator == "all" else error_total(stats, error_types)
    if total == 0:
        return ["0.0"] * len(error_types)
    return [f"{counts[key] / total * 100.0:.1f}" for key, _ in error_types]


def count_cells(stats, error_types):
    if stats is None:
        return [""] * len(error_types)
    if error_types == BROAD_ERROR_TYPES:
        counts = broad_counts(stats)
    else:
        counts = stats["counts"]
    return [str(counts[key]) for key, _ in error_types]


def build_csv_rows(results, error_types, value_mode, denominator):
    metric_labels = [label for _, label in error_types]
    header1 = [
        "Models",
        "Synthetic Homogeneous Scenario",
    ] + [""] * (len(SYNTHETIC) * len(metric_labels) + len(metric_labels) - 1)
    header1 += [
        "Fixed-View Industrial Scenario",
    ] + [""] * (len(FIXED) * len(metric_labels) + len(metric_labels) - 1)
    header1 += [
        "View-Variant Industrial Scenario",
    ] + [""] * (len(VIEW_VARIANT) * len(metric_labels) + len(metric_labels) - 1)

    header2 = [""]
    for label, _ in DATASET_COLUMNS:
        header2.extend([label] + [""] * (len(metric_labels) - 1))

    header3 = [""]
    for _ in DATASET_COLUMNS:
        header3.extend(metric_labels)

    rows = [header1, header2, header3]
    for display_name, model_key in MODEL_ROWS:
        row = [display_name]
        for _, dataset_names in DATASET_COLUMNS:
            stats = stats_for_model(results, model_key, dataset_names)
            if value_mode == "percent":
                row.extend(percent_cells(stats, error_types, denominator))
            else:
                row.extend(count_cells(stats, error_types))
        rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def latex_escape(text):
    text = str(text)
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
    return "".join(replacements.get(ch, ch) for ch in text)


def nonempty_spans(row, start_col=1, end_col=None):
    if end_col is None:
        end_col = len(row)
    starts = [(idx, cell.strip()) for idx, cell in enumerate(row[start_col:end_col], start_col) if cell.strip()]
    spans = []
    for pos, (idx, label) in enumerate(starts):
        next_idx = starts[pos + 1][0] if pos + 1 < len(starts) else end_col
        spans.append((idx, next_idx - idx, label))
    return spans


def cmidrules(spans):
    parts = []
    for start_idx, span, _ in spans:
        first = start_idx + 1
        last = start_idx + span
        parts.append(rf"\cmidrule(lr){{{first}-{last}}}")
    return " ".join(parts)


def display_model_name(model_name):
    if model_name.startswith("Gemma"):
        return model_name.replace("-it", "")
    return model_name


def display_dataset_name(dataset_name):
    return DATASET_DISPLAY_NAMES.get(dataset_name.strip(), dataset_name)


def model_group(model_name):
    if model_name.startswith("InternVL"):
        return "InternVL"
    if model_name.startswith("Qwen3-VL"):
        return "Qwen3-VL"
    if model_name.startswith("Qwen3.5"):
        return "Qwen3.5"
    if model_name.startswith("Gemma"):
        return "Gemma"
    return model_name.split("-")[0]


def make_latex_table(rows, task_name, value_mode, detail, denominator):
    if len(rows) < 4:
        raise ValueError("CSV must contain three header rows and at least one data row")

    header1, header2, header3 = rows[:3]
    data_rows = rows[3:]
    col_count = max(len(r) for r in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    group_spans = nonempty_spans(header1, 1, col_count)
    dataset_spans = nonempty_spans(header2, 1, col_count)
    align = "c" * col_count
    suffix = "detailed directional" if detail else "broad"
    if value_mode == "percent":
        metric_desc = "error-type distributions" if denominator == "error" else "all-sample error rates"
    else:
        metric_desc = "error counts"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\caption{{{latex_escape(task_name)} {latex_escape(suffix)} {latex_escape(metric_desc)} across datasets.}}",
        rf"\label{{tab:{task_name.lower()}_{'detail' if detail else 'broad'}_error_analysis}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    group_cells = [rf"\multirow{{3}}{{*}}{{{latex_escape(header1[0].strip() or 'Models')}}}"]
    cursor = 1
    for start, span, label in group_spans:
        while cursor < start:
            group_cells.append("")
            cursor += 1
        group_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(display_dataset_name(label))}}}")
        cursor = start + span
    lines.append(" & ".join(group_cells) + r" \\")
    if group_spans:
        lines.append(cmidrules(group_spans))

    dataset_cells = [""]
    cursor = 1
    for start, span, label in dataset_spans:
        while cursor < start:
            dataset_cells.append("")
            cursor += 1
        dataset_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(display_dataset_name(label))}}}")
        cursor = start + span
    lines.append(" & ".join(dataset_cells) + r" \\")
    if dataset_spans:
        lines.append(cmidrules(dataset_spans))

    metric_cells = [""] + [latex_escape(cell.strip()) for cell in header3[1:col_count]]
    lines.append(" & ".join(metric_cells) + r" \\")
    lines.append(r"\midrule")

    prev_group = None
    for row in data_rows:
        current_group = model_group(row[0].strip())
        if prev_group is not None and current_group != prev_group:
            lines.append(r"\midrule")
        prev_group = current_group

        cells = [latex_escape(display_model_name(row[0].strip()))]
        for col in range(1, col_count):
            cells.append(latex_escape(row[col].strip()) if row[col].strip() else "--")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_latex(path, rows, task_name, value_mode, detail, denominator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_latex_table(rows, task_name, value_mode, detail, denominator), encoding="utf-8")


def generate_for_task(task_name, eval_dir, task, out_dir, detail, value_mode, denominator):
    error_types = DETAIL_ERROR_TYPES if detail else BROAD_ERROR_TYPES
    results = collect_results(eval_dir, task)
    rows = build_csv_rows(results, error_types, value_mode, denominator)
    suffix = "detail" if detail else "broad"
    value_suffix = f"{value_mode}_{denominator}" if value_mode == "percent" else value_mode
    csv_path = out_dir / f"{task_name}_error_analysis_{suffix}_{value_suffix}.csv"
    tex_path = out_dir / "latex_tables" / f"{task_name}_error_analysis_{suffix}_{value_suffix}.tex"
    write_csv(csv_path, rows)
    write_latex(tex_path, rows, task_name, value_mode, detail, denominator)
    return csv_path, tex_path


def main():
    parser = argparse.ArgumentParser(description="Generate IOL/SOI error analysis CSV and LaTeX tables.")
    parser.add_argument("--iol-dir", default=str(ROOT / "IOL_type" / "eval"))
    parser.add_argument("--soi-dir", default=str(ROOT / "SOI_type" / "eval"))
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "reports"))
    parser.add_argument("--detail", action="store_true", help="Split partial matches into under/over/mixed directions")
    parser.add_argument("--value-mode", choices=["percent", "count"], default="percent")
    parser.add_argument(
        "--denominator",
        choices=["error", "all"],
        default="error",
        help="Percent denominator: error samples only, or all samples. Ignored for count mode.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    outputs = []
    outputs.extend(generate_for_task("IOL", args.iol_dir, "iol", out_dir, args.detail, args.value_mode, args.denominator))
    outputs.extend(generate_for_task("SOI", args.soi_dir, "soi", out_dir, args.detail, args.value_mode, args.denominator))

    tex_outputs = [path for path in outputs if path.suffix == ".tex"]
    value_suffix = f"{args.value_mode}_{args.denominator}" if args.value_mode == "percent" else args.value_mode
    combined_path = out_dir / "latex_tables" / f"IOL_SOI_error_analysis_{'detail' if args.detail else 'broad'}_{value_suffix}.tex"
    combined_path.write_text("\n".join(path.read_text(encoding="utf-8") for path in tex_outputs), encoding="utf-8")
    outputs.append(combined_path)

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
