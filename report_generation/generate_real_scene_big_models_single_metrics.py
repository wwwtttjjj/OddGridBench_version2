#!/usr/bin/env python3
import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

MODEL_SPECS = [
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    # The local result files use Qwen3.5-27B for the large Qwen3.5 model.
    ("Qwen3.5-27B", "Qwen3.5-27B"),
    ("InternVL3.5-38B", "InternVL3_5-38B"),
    ("Gemma4-31B-it", "gemma-4-31B-it"),
]

ROW_TYPES = [
    ("IOL", "iol"),
    ("SOI", "soi"),
    ("zeroshot", "zero-shot"),
    ("oneshot", "one-example"),
    ("twoshot", "two-examples"),
]

SCENARIOS = [
    (
        "Fixed-View Industrial Scenario",
        [
            ("MVTec AD", {"grid": "MVTEC_output", "single": "mvtec"}),
            ("VisA", {"grid": "VisA_output", "single": "VisA"}),
            ("BTAD", {"grid": "BTech_output", "single": "BTech_Dataset_transformed"}),
        ],
    ),
    (
        "View-Variant Industrial Scenario",
        [
            ("MPDD/RAD", {"grid": ["MPDD_output", "RAD_output"], "single": ["MPDD", "RAD"]}),
            ("GoodsAD", {"grid": "GOODADS_output", "single": "GOODADS"}),
        ],
    ),
]

ABLATION_DIRS = {
    "zero-shot": PROJECT_ROOT / "Ablation" / "results_zero-shot",
    "one-example": PROJECT_ROOT / "Ablation" / "results_one-example",
    "two-examples": PROJECT_ROOT / "Ablation" / "results_two-examples",
}

IOL_COORD_RE = re.compile(r"^\((\d+),(\d+)\)$")
SOI_IMAGE_RE = re.compile(r"image\s*(\d+)", re.IGNORECASE)
METRIC_CACHE = {}
WARNED_MISSING = set()


@dataclass
class Confusion:
    tp: float = 0.0
    fp: float = 0.0
    tn: float = 0.0
    fn: float = 0.0
    samples: int = 0
    units: float = 0.0
    invalid_pred: float = 0.0

    def add(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        self.samples += other.samples
        self.units += other.units
        self.invalid_pred += other.invalid_pred

    @property
    def acc(self):
        return (self.tp + self.tn) / self.units if self.units else None

    @property
    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def ensure_list(value):
    return value if isinstance(value, list) else [value]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def normalize_iol_gt(answer):
    out = set()
    for item in answer or []:
        coord = normalize_iol_coord(item)
        if coord is not None:
            out.add(coord)
    return out


def normalize_iol_coord(value):
    if isinstance(value, dict):
        if value.get("row") is not None and value.get("col") is not None:
            value = [value.get("row"), value.get("col")]
        else:
            for key in ("grid_pos", "coord", "coords", "coordinate", "position"):
                if key in value:
                    value = value.get(key)
                    break
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    row, col = value
    if row is None or col is None:
        return None
    try:
        return (int(row), int(col))
    except (TypeError, ValueError):
        return None


def normalize_iol_pred(extract_answer):
    if extract_answer == "" or extract_answer is None:
        return None
    if not isinstance(extract_answer, list):
        return None
    coords = set()
    for item in extract_answer:
        if not isinstance(item, str):
            return None
        match = IOL_COORD_RE.match(item.strip())
        if not match:
            return None
        coords.add((int(match.group(1)), int(match.group(2))))
    return coords


def parse_grid_size(grid_size):
    if grid_size is None:
        return [3, 3]
    if isinstance(grid_size, str):
        try:
            grid_size = json.loads(grid_size)
        except Exception:
            return [3, 3]
    if isinstance(grid_size, list) and len(grid_size) == 2:
        return [int(grid_size[0]), int(grid_size[1])]
    return [3, 3]


def sample_keys(sample):
    keys = []
    if sample.get("id") is not None:
        keys.append(("id", str(sample["id"])))
    if sample.get("image") is not None:
        keys.append(("image", str(sample["image"])))
    return keys


def get_embedded_total_count(sample):
    valid_cells = get_embedded_valid_iol_cells(sample)
    if valid_cells is not None:
        return len(valid_cells)
    if sample.get("total_count") is not None:
        return int(sample["total_count"])
    source_cells = sample.get("source_cells")
    if isinstance(source_cells, list):
        return sum(1 for cell in source_cells if cell.get("original_name") is not None)
    return None


def get_embedded_valid_iol_cells(sample):
    source_cells = sample.get("source_cells")
    if not isinstance(source_cells, list):
        return None

    has_coord_metadata = False
    cells = []
    for cell in source_cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("original_name") is None:
            continue
        if (
            cell.get("row") is not None
            or cell.get("col") is not None
            or any(key in cell for key in ("grid_pos", "coord", "coords", "coordinate", "position"))
        ):
            has_coord_metadata = True
        coord = normalize_iol_coord(cell)
        if coord is not None:
            cells.append(coord)

    return cells if has_coord_metadata else None


def build_iol_meta_lookup(dataset_dir):
    lookup = {}
    for json_path in sorted(Path(dataset_dir).glob("*.json")):
        try:
            samples = read_json(json_path)
        except Exception:
            continue
        for sample in samples:
            total_count = get_embedded_total_count(sample)
            valid_cells = get_embedded_valid_iol_cells(sample)
            if total_count is None and valid_cells is None:
                continue
            meta = {"total_count": total_count, "valid_cells": valid_cells}
            for key in sample_keys(sample):
                lookup.setdefault(key, meta)
    return lookup


def get_valid_iol_cells(sample, grid_size, iol_meta_lookup=None):
    valid_cells = get_embedded_valid_iol_cells(sample)
    if valid_cells is not None:
        return valid_cells

    total_count = get_embedded_total_count(sample)
    if iol_meta_lookup:
        for key in sample_keys(sample):
            if key in iol_meta_lookup:
                meta = iol_meta_lookup[key]
                if meta.get("valid_cells") is not None:
                    return meta["valid_cells"]
                if total_count is None and meta.get("total_count") is not None:
                    total_count = int(meta["total_count"])
                break
    if total_count is None:
        total_count = int(grid_size[0]) * int(grid_size[1])
    return valid_iol_cells_from_total_count(grid_size, total_count)


def valid_iol_cells_from_total_count(grid_size, total_count):
    rows, cols = grid_size
    cells = []
    for idx in range(total_count):
        row = idx // cols + 1
        col = idx % cols + 1
        if row <= rows:
            cells.append((row, col))
    return cells


def eval_iol_file(path, iol_meta_lookup=None):
    stat = Confusion()
    for sample in read_json(path):
        gt = normalize_iol_gt(sample.get("answer", []))
        pred = normalize_iol_pred(sample.get("extract_answer", ""))
        grid_size = parse_grid_size(sample.get("grid_size", [3, 3]))
        current_pred = pred if pred is not None else set()
        valid_cells = get_valid_iol_cells(sample, grid_size, iol_meta_lookup)
        valid_set = set(valid_cells)

        fp = len(current_pred - valid_set)
        tp = tn = fn = 0
        for cell in valid_cells:
            is_pred = cell in current_pred
            is_gt = cell in gt
            if is_gt and is_pred:
                tp += 1
            elif is_gt and not is_pred:
                fn += 1
            elif not is_gt and is_pred:
                fp += 1
            else:
                tn += 1
        stat.add(Confusion(tp=tp, fp=fp, tn=tn, fn=fn, samples=1, units=len(valid_cells)))
    return stat


def normalize_soi_gt(answer):
    return {int(x) for x in (answer or [])}


def normalize_soi_pred(extract_answer):
    if extract_answer is None:
        return None
    parts = extract_answer if isinstance(extract_answer, list) else str(extract_answer).split(",")
    indices = set()
    for item in parts:
        item = str(item).strip()
        if not item:
            continue
        match = SOI_IMAGE_RE.search(item)
        if not match:
            return None
        indices.add(int(match.group(1)))
    return indices


def eval_soi_file(path):
    stat = Confusion()
    for sample in read_json(path):
        gt = normalize_soi_gt(sample.get("answer", []))
        pred = normalize_soi_pred(sample.get("extract_answer", ""))
        current_pred = pred if pred is not None else set()
        total_count = int(sample.get("total_images") or sample.get("image_num") or 9)
        tp = fp = tn = fn = 0
        for idx in range(1, total_count + 1):
            is_pred = idx in current_pred
            is_gt = idx in gt
            if is_gt and is_pred:
                tp += 1
            elif is_gt and not is_pred:
                fn += 1
            elif not is_gt and is_pred:
                fp += 1
            else:
                tn += 1
        stat.add(Confusion(tp=tp, fp=fp, tn=tn, fn=fn, samples=1, units=tp + fp + tn + fn))
    return stat


def eval_ablation_file(path):
    stat = Confusion()
    pos_labels = {"1", "true", "yes", "correct"}
    neg_labels = {"0", "false", "no", "incorrect"}
    for item in read_json(path):
        weight = float(item.get("original_count") or item.get("count") or 1)
        pred = str(item.get("extract_answer", "")).strip().lower()
        gt = str(item.get("gt", "")).strip().lower()
        is_pred_pos = pred in pos_labels
        is_pred_neg = pred in neg_labels
        is_gt_pos = gt in pos_labels
        is_gt_neg = gt in neg_labels
        if not is_gt_pos and not is_gt_neg:
            continue

        tp = fp = tn = fn = invalid = 0.0
        if not is_pred_pos and not is_pred_neg:
            invalid = weight
            if is_gt_pos:
                fn = weight
            else:
                fp = weight
        elif is_pred_pos and is_gt_pos:
            tp = weight
        elif is_pred_pos and is_gt_neg:
            fp = weight
        elif is_pred_neg and is_gt_pos:
            fn = weight
        elif is_pred_neg and is_gt_neg:
            tn = weight
        stat.add(Confusion(tp=tp, fp=fp, tn=tn, fn=fn, samples=1, units=weight, invalid_pred=invalid))
    return stat


def eval_grid(task, dataset_key, model_key):
    stat = Confusion()
    base = PROJECT_ROOT / ("IOL_type" if task == "iol" else "SOI_type") / "eval"
    evaluator = eval_iol_file if task == "iol" else eval_soi_file
    for dataset_dir in ensure_list(dataset_key):
        path = base / dataset_dir / f"{model_key}.json"
        if path.exists():
            dataset_path = base / dataset_dir
            cache_key = (task, path)
            if cache_key not in METRIC_CACHE:
                if task == "iol":
                    lookup_key = ("iol_meta_lookup", dataset_path)
                    if lookup_key not in METRIC_CACHE:
                        METRIC_CACHE[lookup_key] = build_iol_meta_lookup(dataset_path)
                    METRIC_CACHE[cache_key] = evaluator(path, METRIC_CACHE[lookup_key])
                else:
                    METRIC_CACHE[cache_key] = evaluator(path)
            stat.add(METRIC_CACHE[cache_key])
        else:
            if path not in WARNED_MISSING:
                WARNED_MISSING.add(path)
                print(f"[WARN] missing {path}")
    return stat if stat.units else None


def eval_ablation(mode, dataset_key, model_key):
    stat = Confusion()
    base = ABLATION_DIRS[mode]
    for dataset_name in ensure_list(dataset_key):
        path = base / f"iol_{dataset_name}_{model_key}.json"
        if path.exists():
            cache_key = (mode, path)
            if cache_key not in METRIC_CACHE:
                METRIC_CACHE[cache_key] = eval_ablation_file(path)
            stat.add(METRIC_CACHE[cache_key])
        else:
            if path not in WARNED_MISSING:
                WARNED_MISSING.add(path)
                print(f"[WARN] missing {path}")
    return stat if stat.units else None


def format_int(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def format_pct(value):
    return "" if value is None else f"{value * 100:.2f}"


def metric_cells(stat):
    if stat is None:
        return ["", "", "", "", ""]
    return [
        format_int(stat.units),
        format_pct(stat.acc),
        format_pct(stat.precision),
        format_pct(stat.recall),
        format_pct(stat.f1),
    ]


def collect_metric(row_type, dataset_spec, model_key):
    if row_type == "iol":
        return eval_grid("iol", dataset_spec["grid"], model_key)
    if row_type == "soi":
        return eval_grid("soi", dataset_spec["grid"], model_key)
    return eval_ablation(row_type, dataset_spec["single"], model_key)


def build_table():
    header1 = ["Model", "Type"]
    header2 = ["", ""]
    header3 = ["", ""]
    columns = []

    for scenario_name, datasets in SCENARIOS:
        scenario_span = (len(datasets) + 1) * 5
        header1.extend([scenario_name] + [""] * (scenario_span - 1))
        for dataset_label, dataset_spec in datasets:
            columns.append((scenario_name, dataset_label, dataset_spec))
            header2.extend([dataset_label] + [""] * 4)
            header3.extend(["N", "Acc", "Precision", "Recall", "F1"])
        total_spec = {
            "grid": [name for _, spec in datasets for name in ensure_list(spec["grid"])],
            "single": [name for _, spec in datasets for name in ensure_list(spec["single"])],
        }
        columns.append((scenario_name, "Total", total_spec))
        header2.extend(["Total"] + [""] * 4)
        header3.extend(["N", "Acc", "Precision", "Recall", "F1"])

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


def latex_row(cells):
    return " & ".join(latex_escape(cell) for cell in cells) + r" \\"


def write_latex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(rows[0])
    scenario_spans = spans_from_header(rows[0])
    dataset_spans = spans_from_header(rows[1])
    align = "ll" + "r" * (ncols - 2)
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
        top.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(label)}}}")
    lines.append(" & ".join(top) + r" \\")
    lines.append(cmidrules(scenario_spans))

    mid = ["", ""]
    for _, span, label in dataset_spans:
        mid.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(label)}}}")
    lines.append(" & ".join(mid) + r" \\")
    lines.append(cmidrules(dataset_spans))
    lines.append(latex_row(rows[2]))
    lines.append(r"\midrule")

    for row_idx, row in enumerate(rows[3:]):
        out = row[:]
        if row_idx % len(ROW_TYPES) == 0:
            out[0] = rf"\multirow{{{len(ROW_TYPES)}}}{{*}}{{{latex_escape(row[0])}}}"
        lines.append(" & ".join(out[:1] + [latex_escape(x) for x in out[1:]]) + r" \\")
        if row_idx % len(ROW_TYPES) == len(ROW_TYPES) - 1 and row_idx != len(rows[3:]) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Cell-level metrics on real-world scenarios. N denotes the number of valid small images/cells; IOL missing/None cells are excluded.}",
        r"\label{tab:real_scene_big_models_single_metrics}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CSV and LaTeX tables for large models on real-world scenarios."
    )
    default_csv = PROJECT_ROOT / "report_generation" / "merged_reports" / "real_scene_big_models_single_metrics.csv"
    default_tex = PROJECT_ROOT / "report_generation" / "merged_reports" / "latex_tables" / "real_scene_big_models_single_metrics.tex"
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
