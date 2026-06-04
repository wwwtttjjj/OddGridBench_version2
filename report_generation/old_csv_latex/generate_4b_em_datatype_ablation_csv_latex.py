#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TRAIN_DATATYPES = ["SOI", "IOL", "SYS", "REAL", "TOTAL"]
ALGOS = ["grpo", "gspo", "dapo"]
MODEL_SIZE = "4B"
FUNCTION_TYPE = "EM"

ROW_SPECS = [("Baseline", "", "Qwen3-VL-4B", "Qwen3-VL-4B-Instruct")]
for train_datatype in TRAIN_DATATYPES:
    for algo in ALGOS:
        display_algo = algo.upper()
        display_name = f"VCP-4B-{train_datatype}-{display_algo}"
        model_key = f"Qwen3_vl_{MODEL_SIZE}_{train_datatype}_{FUNCTION_TYPE}_{algo}_step_200"
        ROW_SPECS.append((train_datatype, display_algo, display_name, model_key))

SYNTHETIC = [("ICON", ["icon_output"]), ("MNIST", ["mnist_output"]), ("Hanzi", ["hanzi_output"])]
FIXED = [("MVTEC", ["MVTEC_output"]), ("VISA", ["VisA_output"]), ("Btec", ["BTech_output"])]
VIEW_VARIANT = [("MPDD/RAD", ["MPDD_output", "RAD_output"]), ("GOODADS", ["GOODADS_output"])]
DATASET_COLUMNS = (
    SYNTHETIC
    + [("Total", [name for _, names in SYNTHETIC for name in names])]
    + FIXED
    + [("Total", [name for _, names in FIXED for name in names])]
    + VIEW_VARIANT
    + [("Total", [name for _, names in VIEW_VARIANT for name in names])]
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
    return average_metrics(vals)


def metric_cells(metric):
    if metric is None:
        return ["", ""]
    return [f"{metric['EM']:.1f}", f"{metric['F1']:.1f}"]


def build_csv_rows(results):
    header1 = ["Train Data", "Algo", "Models"]
    header1 += ["Synthetic Scenario"] + [""] * (2 * (len(SYNTHETIC) + 1) - 1)
    header1 += ["Fixed-View Industrial Scenario"] + [""] * (2 * (len(FIXED) + 1) - 1)
    header1 += ["View-Variant Industrial Scenario"] + [""] * (2 * (len(VIEW_VARIANT) + 1) - 1)
    header2 = ["", "", ""]
    for label, _ in DATASET_COLUMNS:
        header2.extend([label, ""])
    header3 = ["", "", ""]
    for _ in DATASET_COLUMNS:
        header3.extend(["EM", "F1"])

    rows = [header1, header2, header3]
    for train_datatype, algo, model_display, model_key in ROW_SPECS:
        row = [train_datatype, algo, model_display]
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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


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


def format_metric(value, best_value=None, bold_best=True):
    value = value.strip()
    if not value:
        return "--"
    formatted = latex_escape(value)
    numeric = parse_float(value)
    if bold_best and numeric is not None and best_value is not None and abs(numeric - best_value) < 1e-9:
        formatted = rf"\textbf{{{formatted}}}"
    return formatted


def make_latex_table(rows, task_name, bold_best=True):
    if len(rows) < 4:
        raise ValueError("CSV must contain three header rows and at least one data row")

    header1, header2, header3 = rows[:3]
    data_rows = rows[3:]
    col_count = max(len(r) for r in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    group_spans = nonempty_spans(header1, 3, col_count)
    dataset_spans = nonempty_spans(header2, 3, col_count)
    best = best_columns(data_rows) if bold_best else {}

    align = "ccc" + "c" * (col_count - 3)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\caption{{{latex_escape(task_name)} 4B EM reward data-type ablation results across datasets.}}",
        rf"\label{{tab:{task_name.lower()}_4b_em_datatype_ablation}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    group_cells = [
        rf"\multirow{{3}}{{*}}{{{latex_escape(header1[0])}}}",
        rf"\multirow{{3}}{{*}}{{{latex_escape(header1[1])}}}",
        rf"\multirow{{3}}{{*}}{{{latex_escape(header1[2])}}}",
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

    metric_cells = ["", "", ""] + [latex_escape(cell.strip()) for cell in header3[3:col_count]]
    lines.append(" & ".join(metric_cells) + r" \\")
    lines.append(r"\midrule")

    prev_train = None
    for row in data_rows:
        current_train = row[0].strip()
        if prev_train is not None and current_train != prev_train:
            lines.append(r"\midrule")
        prev_train = current_train

        cells = [latex_escape(row[0].strip()), latex_escape(row[1].strip()), latex_escape(row[2].strip())]
        for col in range(3, col_count):
            cells.append(format_metric(row[col], best.get(col), bold_best))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def generate(iol_dir, soi_dir, out_dir, latex_dir, bold_best=True):
    out_dir = Path(out_dir)
    latex_dir = Path(latex_dir)
    outputs = []
    for task_name, eval_dir, task in [
        ("IOL", iol_dir, "iol"),
        ("SOI", soi_dir, "soi"),
    ]:
        rows = build_csv_rows(collect_results(eval_dir, task))
        csv_path = out_dir / f"{task_name}_4b_em_datatype_ablation.csv"
        write_csv(csv_path, rows)
        tex_path = latex_dir / f"{task_name}_4b_em_datatype_ablation.tex"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(make_latex_table(rows, task_name, bold_best=bold_best), encoding="utf-8")
        outputs.extend([csv_path, tex_path])

    combined_path = latex_dir / "IOL_SOI_4b_em_datatype_ablation_tables.tex"
    combined_path.write_text(
        "\n".join((latex_dir / f"{task}_4b_em_datatype_ablation.tex").read_text(encoding="utf-8") for task in ["IOL", "SOI"]),
        encoding="utf-8",
    )
    outputs.append(combined_path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate 4B EM reward data-type ablation CSV and LaTeX tables for IOL/SOI.")
    parser.add_argument("--iol-dir", default=str(ROOT / "IOL_type" / "eval"))
    parser.add_argument("--soi-dir", default=str(ROOT / "SOI_type" / "eval"))
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
