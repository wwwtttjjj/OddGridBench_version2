#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

MODEL_ROWS = [
    (
        "Qwen3-VL-4B Baseline",
        "SOI",
        "Qwen3-VL-4B-Instruct",
        PROJECT_ROOT / "SOI_type" / "eval",
        "soi",
    ),
    (
        "Qwen3-VL-4B Baseline",
        "IOL",
        "Qwen3-VL-4B-Instruct",
        PROJECT_ROOT / "IOL_type" / "eval",
        "iol",
    ),
    (
        "Qwen3-VL-4B IOL-type",
        "SOI",
        "Qwen3_vl_4B_IOL_EM_dapo_step_200",
        PROJECT_ROOT / "SOI_type" / "eval",
        "soi",
    ),
    (
        "Qwen3-VL-4B SOI-type",
        "IOL",
        "Qwen3_vl_4B_SOI_EM_dapo_step_200",
        PROJECT_ROOT / "IOL_type" / "eval",
        "iol",
    ),
]

SCENARIOS = [
    (
        "Synthetic Scenario",
        [
            ("ICON", ["icon_output"]),
            ("MNIST", ["mnist_output"]),
            ("Hanzi", ["hanzi_output"]),
        ],
    ),
    (
        "Fixed-View Industrial Scenario",
        [
            ("MVTEC", ["MVTEC_output"]),
            ("VISA", ["VisA_output"]),
            ("Btec", ["BTech_output"]),
        ],
    ),
    (
        "View-Variant Industrial Scenario",
        [
            ("MPDD/RAD", ["MPDD_output", "RAD_output"]),
            ("GOODADS", ["GOODADS_output"]),
        ],
    ),
]

DATASET_DISPLAY_NAMES = {
    "ICON": "IconSim",
    "MNIST": "DigitSim",
    "Hanzi": "HanziSim",
    "MVTEC": "MVTec AD",
    "VISA": "VisA",
    "Btec": "BTAD",
    "GOODADS": "GoodsAD",
    "MPDD/RAD": "MPDD/RAD",
}

IOL_COORD_RE = re.compile(r"^\((\d+),(\d+)\)$")
SOI_IMAGE_RE = re.compile(r"^image(\d+)$")


def normalize_iol_gt(answer):
    out = []
    for item in answer or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
    return out


def normalize_iol_pred(extract_answer):
    if extract_answer is None or extract_answer == "":
        return None
    if extract_answer == []:
        return []
    if not isinstance(extract_answer, list):
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
    if extract_answer is None or extract_answer == "":
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
    return {"EM": em_sum / n * 100.0, "F1": f1_sum / n * 100.0, "N": n}


def find_model_file(eval_dir, dataset_dir_name, model_key, allow_old=True):
    eval_dir = Path(eval_dir)
    candidates = [eval_dir / dataset_dir_name / f"{model_key}.json"]
    if allow_old:
        candidates.append(eval_dir / "z_oldoutput" / dataset_dir_name / f"{model_key}.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def average_metrics(items):
    vals = [item for item in items if item is not None]
    if not vals:
        return None
    return {
        "EM": sum(item["EM"] for item in vals) / len(vals),
        "F1": sum(item["F1"] for item in vals) / len(vals),
    }


def metric_cells(metric):
    if metric is None:
        return ["", ""]
    return [f"{metric['EM']:.1f}", f"{metric['F1']:.1f}"]


def dataset_columns():
    columns = []
    for scenario_name, datasets in SCENARIOS:
        scenario_dataset_labels = []
        for label, dir_names in datasets:
            columns.append((scenario_name, label, [label]))
            scenario_dataset_labels.append(label)
        columns.append((scenario_name, "Total", scenario_dataset_labels))
    return columns


def collect_model_metrics(eval_dir, task, model_key, allow_old=True):
    metrics = {}
    missing = []
    for _, datasets in SCENARIOS:
        for label, dir_names in datasets:
            found = None
            for dataset_dir_name in dir_names:
                path = find_model_file(eval_dir, dataset_dir_name, model_key, allow_old=allow_old)
                if path is not None:
                    found = path
                    break
            if found is None:
                missing.append(label)
                continue
            result = eval_json_file(found, task)
            if result is not None:
                metrics[label] = result
    return metrics, missing


def build_csv_rows(allow_old=True):
    columns = dataset_columns()
    header1 = ["Models", "Eval Data"]
    header2 = ["", ""]
    header3 = ["", ""]
    last_scenario = None
    for scenario_name, label, _ in columns:
        header1.extend([scenario_name if scenario_name != last_scenario else "", ""])
        header2.extend([label, ""])
        header3.extend(["EM", "F1"])
        last_scenario = scenario_name

    rows = [header1, header2, header3]
    warnings = []
    for display_name, eval_data, model_key, eval_dir, task in MODEL_ROWS:
        metrics, missing = collect_model_metrics(eval_dir, task, model_key, allow_old=allow_old)
        if missing:
            warnings.append(f"{display_name} on {eval_data}: missing {', '.join(missing)}")
        row = [display_name, eval_data]
        for _, _, dir_names in columns:
            row.extend(metric_cells(average_metrics(metrics.get(name) for name in dir_names)))
        rows.append(row)
    return rows, warnings


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)


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


def display_dataset_name(dataset_name):
    return DATASET_DISPLAY_NAMES.get(dataset_name.strip(), dataset_name)


def nonempty_spans(row, start_col=2, end_col=None):
    if end_col is None:
        end_col = len(row)
    starts = [(idx, cell.strip()) for idx, cell in enumerate(row[start_col:end_col], start_col) if cell.strip()]
    spans = []
    for pos, (idx, label) in enumerate(starts):
        next_idx = starts[pos + 1][0] if pos + 1 < len(starts) else end_col
        spans.append((idx, next_idx - idx, label))
    return spans


def cmidrules(spans):
    return " ".join(rf"\cmidrule(lr){{{start + 1}-{start + span}}}" for start, span, _ in spans)


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_columns(data_rows, metric_start_col=2):
    best = {}
    if not data_rows:
        return best
    col_count = max(len(row) for row in data_rows)
    for col in range(metric_start_col, col_count):
        vals = [parse_float(row[col]) for row in data_rows if col < len(row) and row[col].strip()]
        vals = [v for v in vals if v is not None]
        if vals:
            best[col] = max(vals)
    return best


def format_metric(value, best_value=None, bold_best=True):
    value = value.strip()
    if not value:
        return "--"
    formatted = latex_escape(value)
    numeric = parse_float(value)
    if bold_best and numeric is not None and best_value is not None and abs(numeric - best_value) < 1e-9:
        return rf"\textbf{{{formatted}}}"
    return formatted


def make_latex_table(rows, bold_best=True):
    if len(rows) < 4:
        raise ValueError("CSV must contain three header rows and at least one data row")
    header1, header2, header3 = rows[:3]
    data_rows = rows[3:]
    col_count = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    group_spans = nonempty_spans(header1, 2, col_count)
    dataset_spans = nonempty_spans(header2, 2, col_count)
    best = best_columns(data_rows, 2) if bold_best else {}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Cross-type 4B DAPO-EM evaluation results on opposite data types.}",
        r"\label{tab:cross_type_4b_dapo_em_f1}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{'c' * col_count}}}",
        r"\toprule",
    ]

    first_row = [
        rf"\multirow{{3}}{{*}}{{{latex_escape(header1[0])}}}",
        rf"\multirow{{3}}{{*}}{{{latex_escape(header1[1])}}}",
    ]
    cursor = 2
    for start, span, label in group_spans:
        while cursor < start:
            first_row.append("")
            cursor += 1
        first_row.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(label)}}}")
        cursor = start + span
    lines.append(" & ".join(first_row) + r" \\")
    lines.append(cmidrules(group_spans))

    second_row = ["", ""]
    cursor = 2
    for start, span, label in dataset_spans:
        while cursor < start:
            second_row.append("")
            cursor += 1
        second_row.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(display_dataset_name(label))}}}")
        cursor = start + span
    lines.append(" & ".join(second_row) + r" \\")
    lines.append(cmidrules(dataset_spans))

    lines.append(" & ".join(["", ""] + [latex_escape(cell.strip()) for cell in header3[2:col_count]]) + r" \\")
    lines.append(r"\midrule")
    for row in data_rows:
        cells = [latex_escape(row[0].strip()), latex_escape(row[1].strip())]
        for col in range(2, col_count):
            cells.append(format_metric(row[col], best.get(col), bold_best))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def generate(out_dir, latex_dir, allow_old=True, bold_best=True):
    out_dir = Path(out_dir)
    latex_dir = Path(latex_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    rows, warnings = build_csv_rows(allow_old=allow_old)
    csv_path = out_dir / "cross_type_4b_dapo_em_f1.csv"
    tex_path = latex_dir / "cross_type_4b_dapo_em_f1.tex"
    write_csv(csv_path, rows)
    tex_path.write_text(make_latex_table(rows, bold_best=bold_best), encoding="utf-8")
    return [csv_path, tex_path], warnings


def main():
    parser = argparse.ArgumentParser(description="Generate opposite-data-type EM/F1 table for IOL-type and SOI-type 4B DAPO-EM models.")
    parser.add_argument("--out-dir", default=str(ROOT / "merged_reports"))
    parser.add_argument("--latex-dir", default=str(ROOT / "merged_reports" / "latex_tables"))
    parser.add_argument("--no-old-fallback", action="store_true", help="Do not fall back to eval/z_oldoutput when a current result file is missing")
    parser.add_argument("--no-bold-best", action="store_true", help="Do not bold the best value in each EM/F1 column")
    args = parser.parse_args()

    outputs, warnings = generate(
        args.out_dir,
        args.latex_dir,
        allow_old=not args.no_old_fallback,
        bold_best=not args.no_bold_best,
    )
    for warning in warnings:
        print(f"[WARN] {warning}")
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
