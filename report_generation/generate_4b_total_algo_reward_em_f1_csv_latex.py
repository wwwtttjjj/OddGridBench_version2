#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

ROW_SPECS = [
    ("Qwen3-VL-4B", "-", "-", "Qwen3-VL-4B-Instruct"),
    ("VCP-4B", "GRPO", "EM", "Qwen3_vl_4B_TOTAL_EM_grpo_step_200"),
    ("VCP-4B", "GRPO", "F1", "Qwen3_vl_4B_TOTAL_F1_grpo_step_200"),
    ("VCP-4B", "GSPO", "EM", "Qwen3_vl_4B_TOTAL_EM_gspo_step_200"),
    ("VCP-4B", "GSPO", "F1", "Qwen3_vl_4B_TOTAL_F1_gspo_step_200"),
    ("VCP-4B", "DAPO", "EM", "Qwen3_vl_4B_TOTAL_EM_dapo_step_200"),
    ("VCP-4B", "DAPO", "F1", "Qwen3_vl_4B_TOTAL_F1_dapo_step_200"),
]
SYNTHETIC = [("ICON", ["icon_output"]), ("MNIST", ["mnist_output"]), ("Hanzi", ["hanzi_output"])]
FIXED = [("MVTEC", ["MVTEC_output"]), ("VISA", ["VisA_output"]), ("Btec", ["BTech_output"])]
VIEW_VARIANT = [("MPDD/RAD", ["MPDD_output", "RAD_output"]), ("GOODADS", ["GOODADS_output"])]
SUMMARY_DATASET_COLUMNS = (
    [("Synthetic Scenario", "Total", [name for _, names in SYNTHETIC for name in names])]
    + [("Fixed-View Scenario", "Total", [name for _, names in FIXED for name in names])]
    + [("View-Variant Scenario", "Total", [name for _, names in VIEW_VARIANT for name in names])]
)
DETAILED_DATASET_COLUMNS = (
    [("Synthetic Scenario", label, names) for label, names in SYNTHETIC]
    + [("Fixed-View Scenario", label, names) for label, names in FIXED]
    + [("View-Variant Scenario", label, names) for label, names in VIEW_VARIANT]
)

DATASET_DISPLAY_NAMES = {
    "ICON": "IconSim",
    "MNIST": "DigitSim",
    "Hanzi": "HanziSim",
    "MVTEC": "MVTec AD",
    "VISA": "VisA",
    "Btec": "BTAD",
    "MPDD/RAD": "MPDD/RAD",
    "GOODADS": "GoodsAD",
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
    # Empty string means extraction failed/no parsed answer; only [] is an explicit empty prediction.
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
    # Empty string means extraction failed/no parsed answer; only [] is an explicit empty prediction.
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
    if any(value is None for value in vals):
        return None
    return average_metrics(vals)


def metric_cells(metric):
    if metric is None:
        return ["", ""]
    return [f"{metric['EM']:.1f}", f"{metric['F1']:.1f}"]


def build_csv_rows(results, dataset_columns, include_dataset_row=True):
    header1 = ["Models", "Strategy", "Reward"]
    for scenario, _, _ in dataset_columns:
        header1.extend([scenario, ""])

    header_rows = [header1]
    if include_dataset_row:
        header2 = ["", "", ""]
        for _, label, _ in dataset_columns:
            header2.extend([label, ""])
        header_rows.append(header2)

    metric_header = ["", "", ""]
    for _ in dataset_columns:
        metric_header.extend(["EM", "F1"])
    header_rows.append(metric_header)

    rows = header_rows
    for model_display, strategy, reward, model_key in ROW_SPECS:
        row = [model_display, strategy, reward]
        for _, _, dataset_names in dataset_columns:
            row.extend(metric_cells(metric_for_model(results, model_key, dataset_names)))
        rows.append(row)
    return rows

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


def nonempty_spans(row, start_col=3, end_col=None):
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


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_columns(data_rows, metric_start_col=3):
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


def baseline_columns(data_rows, metric_start_col=3):
    if not data_rows:
        return {}
    baseline = data_rows[0]
    return {
        col: parse_float(baseline[col])
        for col in range(metric_start_col, len(baseline))
        if baseline[col].strip() and parse_float(baseline[col]) is not None
    }


def format_delta(delta):
    if delta is None or abs(delta) < 0.05:
        return ""
    if delta > 0:
        return rf" {{\scriptsize\textcolor{{green}}{{↑+{delta:.1f}}}}}"
    return rf" {{\scriptsize\textcolor{{red}}{{↓{delta:.1f}}}}}"


def format_metric(value, best_value=None, baseline_value=None, bold_best=True, show_delta=False):
    value = value.strip()
    if not value:
        return "--"
    formatted = latex_escape(value)
    numeric = parse_float(value)
    if bold_best and numeric is not None and best_value is not None and abs(numeric - best_value) < 1e-9:
        formatted = rf"\textbf{{{formatted}}}"
    if show_delta and numeric is not None and baseline_value is not None:
        formatted += format_delta(numeric - baseline_value)
    return formatted

def latex_row_model_cell(row_idx, data_rows):
    if row_idx == 0:
        return latex_escape(data_rows[row_idx][0].strip())
    if row_idx == 1:
        sequence_rows = len(data_rows) - 1
        return rf"\multirow{{{sequence_rows}}}{{*}}{{{latex_escape(data_rows[row_idx][0].strip())}}}"
    return ""


def make_latex_core(rows, bold_best=True, width=r"\columnwidth"):
    if len(rows) < 3:
        raise ValueError("CSV must contain header rows and at least one data row")

    header1 = rows[0]
    has_dataset_row = len(rows) >= 4 and any(cell.strip() and cell.strip() not in {"EM", "F1"} for cell in rows[1][3:])
    header2 = rows[1] if has_dataset_row else None
    metric_header = rows[2] if has_dataset_row else rows[1]
    data_rows = rows[3:] if has_dataset_row else rows[2:]

    col_count = max(len(r) for r in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    group_spans = nonempty_spans(header1, 3, col_count)
    dataset_spans = nonempty_spans(header2, 3, col_count) if has_dataset_row else []
    best = best_columns(data_rows, metric_start_col=3) if bold_best else {}
    baseline = baseline_columns(data_rows, metric_start_col=3)
    row_span = 3 if has_dataset_row else 2

    align = "c" * col_count
    lines = [
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    group_cells = [
        rf"\multirow{{{row_span}}}{{*}}{{{latex_escape(header1[0])}}}",
        rf"\multirow{{{row_span}}}{{*}}{{{latex_escape(header1[1])}}}",
        rf"\multirow{{{row_span}}}{{*}}{{{latex_escape(header1[2])}}}",
    ]
    cursor = 3
    for start, span, label in group_spans:
        while cursor < start:
            group_cells.append("")
            cursor += 1
        group_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(display_dataset_name(label))}}}")
        cursor = start + span
    lines.append(" & ".join(group_cells) + r" \\")
    if group_spans:
        lines.append(cmidrules(group_spans))

    if has_dataset_row:
        dataset_cells = ["", "", ""]
        cursor = 3
        for start, span, label in dataset_spans:
            while cursor < start:
                dataset_cells.append("")
                cursor += 1
            dataset_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(display_dataset_name(label))}}}")
            cursor = start + span
        lines.append(" & ".join(dataset_cells) + r" \\")
        if dataset_spans:
            lines.append(cmidrules(dataset_spans))

    metric_row = ["", "", ""] + [latex_escape(cell.strip()) for cell in metric_header[3:col_count]]
    lines.append(" & ".join(metric_row) + r" \\")
    lines.append(r"\midrule")

    for row_idx, row in enumerate(data_rows):
        model_cell = latex_row_model_cell(row_idx, data_rows)
        cells = [
            model_cell,
            latex_escape(row[1].strip()),
            latex_escape(row[2].strip()),
        ]
        for col in range(3, col_count):
            cells.append(
                format_metric(
                    row[col],
                    best.get(col),
                    baseline.get(col),
                    bold_best,
                    show_delta=row_idx > 0,
                )
            )
        lines.append(" & ".join(cells) + r" \\")

        next_strategy = data_rows[row_idx + 1][1].strip() if row_idx + 1 < len(data_rows) else None
        current_strategy = row[1].strip()
        if current_strategy == "-":
            lines.append(r"\midrule")
        elif current_strategy in {"GRPO", "GSPO"} and next_strategy != current_strategy:
            lines.append(rf"\cmidrule(lr){{2-{col_count}}}")

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def latex_task_display_name(task_name):
    return {"IOL": "Grid-based", "SOI": "Sequence-based"}.get(task_name, task_name)


def make_latex_table(rows, task_name, table_kind, bold_best=True):
    kind_id = table_kind.lower().replace(" ", "_").replace("-", "_")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        rf"\caption{{{latex_escape(latex_task_display_name(task_name))} 4B baseline plus GRPO/GSPO/DAPO total-data EM- and F1-reward {latex_escape(table_kind)} results.}}",
        rf"\label{{tab:{task_name.lower()}_4b_total_algo_reward_em_f1_{kind_id}}}",
        make_latex_core(rows, bold_best=bold_best, width=r"\columnwidth"),
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def make_combined_latex_table(task_rows, bold_best=True):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\caption{Grid-based and Sequence-based 4B baseline plus GRPO/GSPO/DAPO total-data EM- and F1-reward scenario-total results.}",
        r"\label{tab:vdp-models}",
    ]
    for idx, (task_name, rows) in enumerate(task_rows):
        if idx:
            lines.append(r"\par\vspace{2.2em}")
        panel = chr(ord("a") + idx)
        lines.extend([
            rf"\textbf{{({panel}) {latex_escape(latex_task_display_name(task_name))}}}\\[2pt]",
            make_latex_core(rows, bold_best=bold_best, width=r"\columnwidth"),
        ])
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)

def generate(iol_dir, soi_dir, out_dir, latex_dir, bold_best=True):
    out_dir = Path(out_dir)
    latex_dir = Path(latex_dir)
    latex_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    summary_task_rows = []
    detailed_task_rows = []

    for task_name, eval_dir, task in [
        ("IOL", iol_dir, "iol"),
        ("SOI", soi_dir, "soi"),
    ]:
        results = collect_results(eval_dir, task)

        summary_rows = build_csv_rows(results, SUMMARY_DATASET_COLUMNS, include_dataset_row=False)
        summary_task_rows.append((task_name, summary_rows))
        summary_csv_path = out_dir / f"{task_name}_4b_total_algo_reward_em_f1.csv"
        write_csv(summary_csv_path, summary_rows)
        outputs.append(summary_csv_path)

        detailed_rows = build_csv_rows(results, DETAILED_DATASET_COLUMNS, include_dataset_row=True)
        detailed_task_rows.append((task_name, detailed_rows))
        detailed_csv_path = out_dir / f"{task_name}_4b_total_algo_reward_em_f1_detailed.csv"
        write_csv(detailed_csv_path, detailed_rows)
        outputs.append(detailed_csv_path)

    summary_combined_path = latex_dir / "IOL_SOI_4b_total_algo_reward_em_f1_tables.tex"
    summary_combined_path.write_text(make_combined_latex_table(summary_task_rows, bold_best=bold_best), encoding="utf-8")
    outputs.append(summary_combined_path)

    # detailed_combined_path = latex_dir / "IOL_SOI_4b_total_algo_reward_em_f1_detailed_tables.tex"
    # detailed_combined_path.write_text(make_combined_latex_table(detailed_task_rows, bold_best=bold_best), encoding="utf-8")
    # outputs.append(detailed_combined_path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate 4B baseline plus GRPO/GSPO/DAPO total-data EM/F1 reward CSV and LaTeX tables.")
    parser.add_argument("--iol-dir", default=str(PROJECT_ROOT / "IOL_type" / "eval"))
    parser.add_argument("--soi-dir", default=str(PROJECT_ROOT / "SOI_type" / "eval"))
    parser.add_argument("--out-dir", default=str(ROOT / "merged_reports"))
    parser.add_argument("--latex-dir", default=str(ROOT / "merged_reports" / "latex_tables"))
    parser.add_argument("--no-bold-best", action="store_true", help="Do not bold the best value in each EM/F1 column")
    args = parser.parse_args()

    outputs = generate(
        args.iol_dir,
        args.soi_dir,
        args.out_dir,
        args.latex_dir,
        bold_best=not args.no_bold_best,
    )
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
