#!/usr/bin/env python3
import argparse
from pathlib import Path

import generate_fixed_view_qwen_gemma_single_metrics as base_script


MODEL_SPECS = [
    ("Qwen3-VL-4B", "Qwen3-VL-4B-Instruct"),
    ("Qwen3.5-4B", "Qwen3.5-4B"),
    ("Qwen3-VL-32B", "Qwen3-VL-32B-Instruct"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fixed-view Qwen metrics including 4B models, using the same logic as the base script."
    )
    default_csv = (
        base_script.base.PROJECT_ROOT
        / "report_generation"
        / "merged_reports"
        / "fixed_view_qwen_acc_pre_rec_f1.csv"
    )
    default_tex = (
        base_script.base.PROJECT_ROOT
        / "report_generation"
        / "merged_reports"
        / "latex_tables"
        / "fixed_view_qwen_acc_pre_rec_f1.tex"
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--tex", type=Path, default=default_tex)
    return parser.parse_args()


def main():
    args = parse_args()
    base_script.MODEL_SPECS = MODEL_SPECS
    base_script.base.METRIC_CACHE = {}
    base_script.base.WARNED_MISSING = set()
    base_script.EFF_CACHE = {}
    base_script.WARNED_EFF_MISSING = set()

    rows = base_script.build_table()
    base_script.write_csv(args.csv, rows)
    base_script.write_latex(args.tex, rows)
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote LaTeX: {args.tex}")


if __name__ == "__main__":
    main()
