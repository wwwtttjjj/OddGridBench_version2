#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CSVS = [
    ("IOL", ROOT / "merged_reports" / "IOL_merged_em_f1.csv"),
    ("SOI", ROOT / "merged_reports" / "SOI_merged_em_f1.csv"),
]

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

MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)B", re.IGNORECASE)


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


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
        # LaTeX columns are 1-based. CSV col 0 is Models, so col idx maps to idx + 1.
        first = start_idx + 1
        last = start_idx + span
        parts.append(rf"\cmidrule(lr){{{first}-{last}}}")
    return " ".join(parts)


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_columns(data_rows, metric_start_col=1):
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


def model_size_b(model_name):
    match = MODEL_SIZE_RE.search(model_name)
    return float(match.group(1)) if match else None


def small_model_columns(data_rows, metric_start_col=1, max_size_b=10.0):
    small_rows = [row for row in data_rows if (model_size_b(row[0]) or float("inf")) < max_size_b]
    return best_columns(small_rows, metric_start_col)


def format_metric(value, best_value=None, small_best_value=None, bold_best=True, underline_small_best=True):
    value = value.strip()
    if not value:
        return "--"
    formatted = latex_escape(value)
    numeric = parse_float(value)
    if bold_best and numeric is not None and best_value is not None and abs(numeric - best_value) < 1e-9:
        formatted = rf"\textbf{{{formatted}}}"
    if underline_small_best and numeric is not None and small_best_value is not None and abs(numeric - small_best_value) < 1e-9:
        formatted = rf"\underline{{{formatted}}}"
    return formatted


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


def make_latex_table(rows, task_name, bold_best=True, underline_small_best=True):
    if len(rows) < 4:
        raise ValueError("CSV must contain three header rows and at least one data row")

    header1, header2, header3 = rows[:3]
    data_rows = rows[3:]
    col_count = max(len(r) for r in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    group_spans = nonempty_spans(header1, 1, col_count)
    dataset_spans = nonempty_spans(header2, 1, col_count)
    best = best_columns(data_rows) if bold_best else {}
    small_best = small_model_columns(data_rows) if underline_small_best else {}

    align = "c" * col_count
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(rf"\caption{{{latex_escape(task_name)} EM and F1 results across datasets.}}")
    lines.append(rf"\label{{tab:{task_name.lower()}_em_f1}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")

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
            cells.append(format_metric(row[col], best.get(col), small_best.get(col), bold_best, underline_small_best))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from IOL/SOI merged EM/F1 CSV files.")
    parser.add_argument("--csv-dir", default=str(ROOT / "merged_reports"), help="Directory containing IOL_merged_em_f1.csv and SOI_merged_em_f1.csv")
    parser.add_argument("--out-dir", default=str(ROOT / "merged_reports" / "latex_tables"), help="Directory for .tex outputs")
    parser.add_argument("--no-bold-best", action="store_true", help="Do not bold the best value in each EM/F1 column")
    parser.add_argument("--no-underline-small-best", action="store_true", help="Do not underline the best value among models smaller than 10B in each EM/F1 column")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for task_name, default_path in DEFAULT_CSVS:
        csv_path = csv_dir / default_path.name
        rows = read_csv(csv_path)
        tex = make_latex_table(
            rows,
            task_name,
            bold_best=not args.no_bold_best,
            underline_small_best=not args.no_underline_small_best,
        )
        out_path = out_dir / f"{task_name}_merged_em_f1.tex"
        out_path.write_text(tex, encoding="utf-8")
        outputs.append(out_path)

    combined = "\n".join(path.read_text(encoding="utf-8") for path in outputs)
    combined_path = out_dir / "IOL_SOI_merged_em_f1_tables.tex"
    combined_path.write_text(combined, encoding="utf-8")

    for path in outputs + [combined_path]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
