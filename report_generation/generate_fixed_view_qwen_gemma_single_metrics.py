#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_real_scene_big_models_single_metrics as base


MODEL_SPECS = [
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
]

DATASETS = [
    (
        "MVTec AD",
        {
            "grid": "MVTEC_output",
            "single": "mvtec",
            "eff_grid": "MVTEC",
            "eff_single": {"candidates": ["mvtec", "MVTEC"]},
        },
    ),
    ("VisA", {"grid": "VisA_output", "single": "VisA", "eff_grid": "VisA", "eff_single": "VisA"}),
    (
        "BTAD",
        {
            "grid": "BTech_output",
            "single": "BTech_Dataset_transformed",
            "eff_grid": "BTech",
            "eff_single": {"candidates": ["BTech_Dataset_transformed", "BTech"]},
        },
    ),
]

ROW_TYPES = [
    ("Grid-base", "iol"),
    ("Sequence-base", "soi"),
    ("Zero-shot", "zero-shot"),
    ("One-shot", "one-example"),
    ("Two-shot", "two-examples"),
]

METRICS = ["Acc$\\uparrow$", "Prec$\\uparrow$", "Rec$\\uparrow$", "F1$\\uparrow$", "FPS$\\uparrow$", "AvgTok$\\downarrow$"]
PERFORMANCE_METRIC_COUNT = 4
EFFECTIVE_ROOT = base.PROJECT_ROOT / "Effective"
EFFECTIVE_DIRS = {
    "iol": EFFECTIVE_ROOT / "iol_output",
    "soi": EFFECTIVE_ROOT / "soi_output",
    "zero-shot": EFFECTIVE_ROOT / "single_results_zero-shot",
    "one-example": EFFECTIVE_ROOT / "single_results_one-example",
    "two-examples": EFFECTIVE_ROOT / "single_results_two-examples",
}
EFF_CACHE = {}
WARNED_EFF_MISSING = set()


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
    def avg_seconds(self):
        return self.total_time / self.total_images if self.total_images else None

    @property
    def avg_tokens(self):
        if not self.total_images or not self.has_tokens:
            return None
        return self.total_tokens / self.total_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fixed-view Qwen ablation metrics with per-image efficiency columns."
    )
    default_csv = (
        base.PROJECT_ROOT
        / "report_generation"
        / "merged_reports"
        / "fixed_view_qwen_single_metrics.csv"
    )
    default_tex = (
        base.PROJECT_ROOT
        / "report_generation"
        / "merged_reports"
        / "latex_tables"
        / "fixed_view_qwen_single_metrics.tex"
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--tex", type=Path, default=default_tex)
    return parser.parse_args()


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


def effective_num_images(item, row_type):
    if row_type in {"zero-shot", "one-example", "two-examples"}:
        return 1
    if row_type == "iol":
        return parse_grid_cell_count(item.get("grid_size", "[1,1]"))
    if row_type == "soi":
        return int(item.get("image_num") or item.get("total_images") or 1)
    return 1


def process_efficiency_file(path, row_type):
    stat = EfficiencyStat()
    for item in read_json(path):
        inference_time = item.get("inference_time")
        if inference_time is None:
            continue
        stat.total_time += float(inference_time)
        stat.total_images += int(effective_num_images(item, row_type))
        completion_tokens = item.get("completion_tokens")
        if completion_tokens is not None:
            stat.total_tokens += float(completion_tokens)
            stat.has_tokens = True
    return stat if stat.total_images else None


def effective_filename(row_type, dataset_name, model_key):
    if row_type in {"zero-shot", "one-example", "two-examples"}:
        return f"iol_{dataset_name}_{model_key}.json"
    return f"{dataset_name}_{model_key}.json"


def iter_effective_entries(dataset_key):
    return dataset_key if isinstance(dataset_key, list) else [dataset_key]


def effective_candidates(entry):
    if isinstance(entry, dict):
        return entry.get("candidates", [])
    return base.ensure_list(entry)


def resolve_efficiency_path(row_type, entry, model_key):
    candidates = effective_candidates(entry)
    paths = [EFFECTIVE_DIRS[row_type] / effective_filename(row_type=row_type, dataset_name=dataset_name, model_key=model_key) for dataset_name in candidates]
    for path in paths:
        if path.exists():
            return path
    warn_key = (row_type, tuple(str(path) for path in paths))
    if warn_key not in WARNED_EFF_MISSING:
        WARNED_EFF_MISSING.add(warn_key)
        joined = " or ".join(str(path) for path in paths)
        print(f"[WARN] missing efficiency file {joined}")
    return None


def collect_efficiency(row_type, dataset_spec, model_key):
    key_name = "eff_grid" if row_type in {"iol", "soi"} else "eff_single"
    dataset_key = dataset_spec[key_name]
    stat = EfficiencyStat()
    for entry in iter_effective_entries(dataset_key):
        path = resolve_efficiency_path(row_type, entry, model_key)
        if path is None:
            continue
        cache_key = (row_type, path)
        if cache_key not in EFF_CACHE:
            EFF_CACHE[cache_key] = process_efficiency_file(path, row_type)
        current = EFF_CACHE[cache_key]
        if current is not None:
            stat.add(current)
    return stat if stat.total_images else None


def metric_cells(stat):
    if stat is None:
        return ["", "", "", ""]
    return [
        base.format_pct(stat.acc),
        base.format_pct(stat.precision),
        base.format_pct(stat.recall),
        base.format_pct(stat.f1),
    ]


def efficiency_cells(stat):
    if stat is None:
        return ["", ""]
    fps = "" if not stat.avg_seconds else f"{1.0 / stat.avg_seconds:.2f}"
    avg_tokens = "" if stat.avg_tokens is None else f"{stat.avg_tokens:.2f}"
    return [fps, avg_tokens]


def collect_metric(row_type, dataset_spec, model_key):
    if row_type == "iol":
        return base.eval_grid("iol", dataset_spec["grid"], model_key)
    if row_type == "soi":
        return base.eval_grid("soi", dataset_spec["grid"], model_key)
    return base.eval_ablation(row_type, dataset_spec["single"], model_key)


def combined_cells(row_type, dataset_spec, model_key):
    return metric_cells(collect_metric(row_type, dataset_spec, model_key)) + efficiency_cells(
        collect_efficiency(row_type, dataset_spec, model_key)
    )


def build_columns():
    columns = list(DATASETS)
    total_spec = {
        "grid": [name for _, spec in DATASETS for name in base.ensure_list(spec["grid"])],
        "single": [name for _, spec in DATASETS for name in base.ensure_list(spec["single"])],
        "eff_grid": [name for _, spec in DATASETS for name in base.ensure_list(spec["eff_grid"])],
        "eff_single": [name for _, spec in DATASETS for name in base.ensure_list(spec["eff_single"])],
    }
    columns.append(("Total", total_spec))
    return columns


def build_table():
    columns = build_columns()
    header1 = ["Model", "Type"]
    header2 = ["", ""]
    for dataset_label, _ in columns:
        header1.extend([dataset_label] + [""] * (len(METRICS) - 1))
        header2.extend(METRICS)

    rows = [header1, header2]
    for model_display, model_key in MODEL_SPECS:
        first_row = True
        for row_display, row_type in ROW_TYPES:
            row = [model_display if first_row else "", row_display]
            first_row = False
            for _, dataset_spec in columns:
                row.extend(combined_cells(row_type, dataset_spec, model_key))
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


def colored_extreme_cell(value, color_name):
    return rf"\textcolor{{{color_name}}}{{{base.latex_escape(value)}}}"


def color_extreme_metric_cells(body_rows):
    colored = [row[:] for row in body_rows]
    group_size = len(ROW_TYPES)
    metric_start = 2

    for group_start in range(0, len(body_rows), group_size):
        group = colored[group_start:group_start + group_size]
        for col in range(metric_start, len(body_rows[0]), len(METRICS)):
            for metric_offset in range(len(METRICS)):
                metric_col = col + metric_offset
                values = []
                for row in group:
                    try:
                        values.append(float(row[metric_col]))
                    except (TypeError, ValueError):
                        values.append(None)
                valid = [value for value in values if value is not None]
                if not valid:
                    continue
                max_value = max(valid)
                min_value = min(valid)
                if max_value == min_value:
                    continue

                lower_is_better = metric_offset == PERFORMANCE_METRIC_COUNT + 1
                for offset, value in enumerate(values):
                    if value is None:
                        continue
                    if lower_is_better:
                        if value == min_value:
                            group[offset][metric_col] = colored_extreme_cell(group[offset][metric_col], "MaxValueText")
                        elif value == max_value:
                            group[offset][metric_col] = colored_extreme_cell(group[offset][metric_col], "MinValueText")
                    else:
                        if value == max_value:
                            group[offset][metric_col] = colored_extreme_cell(group[offset][metric_col], "MaxValueText")
                        elif value == min_value:
                            group[offset][metric_col] = colored_extreme_cell(group[offset][metric_col], "MinValueText")
    return colored


def write_latex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(rows[0])
    dataset_spans = spans_from_header(rows[0])
    align = "cc" + "c" * (ncols - 2)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\definecolor{MaxValueText}{HTML}{B00020}",
        r"\definecolor{MinValueText}{HTML}{1F5FA8}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    top = [r"\multirow{2}{*}{Model}", r"\multirow{2}{*}{Type}"]
    for _, span, label in dataset_spans:
        top.append(rf"\multicolumn{{{span}}}{{c}}{{{base.latex_escape(label)}}}")
    lines.append(" & ".join(top) + r" \\")
    lines.append(cmidrules(dataset_spans))
    lines.append(" & ".join(rows[1]) + r" \\")
    lines.append(r"\midrule")

    body_rows = color_extreme_metric_cells(rows[2:])
    for row_idx, row in enumerate(body_rows):
        model_cell = base.latex_escape(row[0])
        if row_idx % len(ROW_TYPES) == 0:
            model_cell = rf"\multirow{{{len(ROW_TYPES)}}}{{*}}{{{model_cell}}}"
        cells = [model_cell, base.latex_escape(row[1])] + row[2:]
        lines.append(" & ".join(cells) + r" \\")
        if row_idx % len(ROW_TYPES) == len(ROW_TYPES) - 1 and row_idx != len(body_rows) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        (
            r"\caption{Fixed-view Qwen model comparison with accuracy metrics and per-image efficiency. "
            r"FPS denotes processed images per second, and AvgTok denotes average completion tokens per image. "
            r"Metric arrows indicate optimization direction. Red marks the best value and blue marks the worst value "
            r"within each model block and metric column.}"
        ),
        r"\label{tab:fixed_view_qwen_single_metrics}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    base.METRIC_CACHE = {}
    base.WARNED_MISSING = set()
    rows = build_table()
    write_csv(args.csv, rows)
    write_latex(args.tex, rows)
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote LaTeX: {args.tex}")


if __name__ == "__main__":
    main()
