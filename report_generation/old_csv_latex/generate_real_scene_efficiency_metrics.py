#!/usr/bin/env python3
import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import generate_real_scene_big_models_single_metrics as base


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
EFFECTIVE_ROOT = PROJECT_ROOT / "Effective"

MODEL_SPECS = [
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
    ("Gemma4-31B-it", "gemma-4-31B-it"),
]

ROW_TYPES = [
    ("Grid-base", "iol"),
    ("Sequence-base", "soi"),
    ("Zero-shot", "zero-shot"),
    ("One-shot", "one-example"),
    ("Two-shot", "two-examples"),
]

SCENARIOS = [
    (
        "Fixed-View Industrial Scenario",
        [
            ("MVTec AD", {"grid": "MVTEC", "single": "mvtec"}),
            ("VisA", {"grid": "VisA", "single": "VisA"}),
            ("BTAD", {"grid": "BTech", "single": "BTech_Dataset_transformed"}),
        ],
    ),
    (
        "View-Variant Industrial Scenario",
        [
            ("MPDD/RAD", {"grid": ["MPDD", "RAD"], "single": ["MPDD", "RAD"]}),
            ("GoodsAD", {"grid": "GOODADS", "single": "GOODADS"}),
        ],
    ),
]

DATA_DIRS = {
    "iol": EFFECTIVE_ROOT / "iol_output",
    "soi": EFFECTIVE_ROOT / "soi_output",
    "zero-shot": EFFECTIVE_ROOT / "single_results_zero-shot",
    "one-example": EFFECTIVE_ROOT / "single_results_one-example",
    "two-examples": EFFECTIVE_ROOT / "single_results_two-examples",
}

METRICS = ["Time", "Tok"]
METRIC_CACHE = {}
WARNED_MISSING = set()


@dataclass
class EfficiencyStat:
    total_time: float = 0.0
    total_tokens: float = 0.0
    total_images: int = 0
    has_tokens: bool = False

    def add(self, other):
        self.total_time += other.total_time
        self.total_tokens += other.total_tokens
        self.total_images += other.total_images
        self.has_tokens = self.has_tokens or other.has_tokens

    @property
    def avg_time(self):
        return self.total_time / self.total_images if self.total_images else None

    @property
    def avg_tokens(self):
        if not self.total_images or not self.has_tokens:
            return None
        return self.total_tokens / self.total_images


def ensure_list(value):
    return value if isinstance(value, list) else [value]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def parse_grid_cell_count(grid_size):
    if grid_size is None:
        return 1
    if isinstance(grid_size, str):
        try:
            grid_size = ast.literal_eval(grid_size)
        except Exception:
            try:
                grid_size = json.loads(grid_size)
            except Exception:
                return 1
    if isinstance(grid_size, (list, tuple)) and len(grid_size) >= 2:
        try:
            return int(grid_size[0]) * int(grid_size[1])
        except (TypeError, ValueError):
            return 1
    return 1


def get_num_images(item, row_type):
    if row_type in {"zero-shot", "one-example", "two-examples"}:
        return 1
    if row_type == "iol":
        return parse_grid_cell_count(item.get("grid_size", "[1,1]"))
    if row_type == "soi":
        return int(item.get("image_num") or item.get("total_images") or 1)
    return 1


def process_file(path, row_type):
    stat = EfficiencyStat()
    for item in read_json(path):
        inference_time = item.get("inference_time")
        if inference_time is None:
            continue
        num_images = get_num_images(item, row_type)
        completion_tokens = item.get("completion_tokens")

        stat.total_time += float(inference_time)
        if completion_tokens is not None:
            stat.total_tokens += float(completion_tokens)
            stat.has_tokens = True
        stat.total_images += int(num_images)
    return stat if stat.total_images else None


def filename_for(row_type, dataset_name, model_key):
    if row_type in {"zero-shot", "one-example", "two-examples"}:
        return f"iol_{dataset_name}_{model_key}.json"
    return f"{dataset_name}_{model_key}.json"


def eval_efficiency(row_type, dataset_key, model_key):
    out = EfficiencyStat()
    data_dir = DATA_DIRS[row_type]
    for dataset_name in ensure_list(dataset_key):
        path = data_dir / filename_for(row_type, dataset_name, model_key)
        if not path.exists():
            if path not in WARNED_MISSING:
                WARNED_MISSING.add(path)
                print(f"[WARN] missing {path}")
            continue
        cache_key = (row_type, path)
        if cache_key not in METRIC_CACHE:
            METRIC_CACHE[cache_key] = process_file(path, row_type)
        stat = METRIC_CACHE[cache_key]
        if stat is not None:
            out.add(stat)
    return out if out.total_images else None


def format_time(value):
    return "" if value is None else f"{value:.2f}"


def format_tokens(value):
    return "" if value is None else f"{value:.2f}"


def metric_cells(stat):
    if stat is None:
        return ["", ""]
    return [format_time(stat.avg_time), format_tokens(stat.avg_tokens)]


def collect_metric(row_type, dataset_spec, model_key):
    if row_type in {"iol", "soi"}:
        return eval_efficiency(row_type, dataset_spec["grid"], model_key)
    return eval_efficiency(row_type, dataset_spec["single"], model_key)


def build_table():
    header1 = ["Model", "Type"]
    header2 = ["", ""]
    header3 = ["", ""]
    columns = []

    for scenario_name, datasets in SCENARIOS:
        scenario_span = (len(datasets) + 1) * len(METRICS)
        header1.extend([scenario_name] + [""] * (scenario_span - 1))
        for dataset_label, dataset_spec in datasets:
            columns.append((scenario_name, dataset_label, dataset_spec))
            header2.extend([dataset_label] + [""] * (len(METRICS) - 1))
            header3.extend(METRICS)
        total_spec = {
            "grid": [name for _, spec in datasets for name in ensure_list(spec["grid"])],
            "single": [name for _, spec in datasets for name in ensure_list(spec["single"])],
        }
        columns.append((scenario_name, "Total", total_spec))
        header2.extend(["Total"] + [""] * (len(METRICS) - 1))
        header3.extend(METRICS)

    rows = [header1, header2, header3]
    for model_display, model_key in MODEL_SPECS:
        first_row = True
        for row_display, row_type in ROW_TYPES:
            row = [model_display if first_row else "", row_display]
            first_row = False
            for _, _, dataset_spec in columns:
                row.extend(metric_cells(collect_metric(row_type, dataset_spec, model_key)))
            rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)


def spans_from_header(row, start_col=2):
    starts = [(idx, value) for idx, value in enumerate(row[start_col:], start_col) if value]
    spans = []
    for pos, (idx, label) in enumerate(starts):
        next_idx = starts[pos + 1][0] if pos + 1 < len(starts) else len(row)
        spans.append((idx, next_idx - idx, label))
    return spans


def cmidrules(spans):
    rules = []
    for idx, span, _ in spans:
        first = idx + 1
        last = idx + span
        rules.append(rf"\cmidrule(lr){{{first}-{last}}}")
    return " ".join(rules)


def write_latex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(rows[0])
    scenario_spans = spans_from_header(rows[0])
    dataset_spans = spans_from_header(rows[1])
    align = "cc" + "c" * (ncols - 2)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    top = [r"\multirow{3}{*}{Model}", r"\multirow{3}{*}{Type}"]
    for _, span, label in scenario_spans:
        top.append(rf"\multicolumn{{{span}}}{{c}}{{{base.latex_escape(label)}}}")
    lines.append(" & ".join(top) + r" \\")
    lines.append(cmidrules(scenario_spans))

    mid = ["", ""]
    for _, span, label in dataset_spans:
        mid.append(rf"\multicolumn{{{span}}}{{c}}{{{base.latex_escape(label)}}}")
    lines.append(" & ".join(mid) + r" \\")
    lines.append(cmidrules(dataset_spans))
    lines.append(" & ".join(base.latex_escape(cell) for cell in rows[2]) + r" \\")
    lines.append(r"\midrule")

    body_rows = rows[3:]
    for row_idx, row in enumerate(body_rows):
        model_cell = base.latex_escape(row[0])
        if row_idx % len(ROW_TYPES) == 0:
            model_cell = rf"\multirow{{{len(ROW_TYPES)}}}{{*}}{{{model_cell}}}"
        cells = [model_cell, base.latex_escape(row[1])] + [base.latex_escape(x) for x in row[2:]]
        lines.append(" & ".join(cells) + r" \\")
        if row_idx % len(ROW_TYPES) == len(ROW_TYPES) - 1 and row_idx != len(body_rows) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\label{tab:real_scene_efficiency_metrics}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-image average inference time and completion-token tables."
    )
    default_csv = PROJECT_ROOT / "report_generation" / "merged_reports" / "real_scene_efficiency_metrics.csv"
    default_tex = (
        PROJECT_ROOT
        / "report_generation"
        / "merged_reports"
        / "latex_tables"
        / "real_scene_efficiency_metrics.tex"
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--tex", type=Path, default=default_tex)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_table()
    write_csv(args.csv, rows)
    write_latex(args.tex, rows)
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote LaTeX: {args.tex}")


if __name__ == "__main__":
    main()
