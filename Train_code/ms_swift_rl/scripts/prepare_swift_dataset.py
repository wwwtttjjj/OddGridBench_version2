#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MS_SWIFT_ROOT = SCRIPT_DIR.parent
TRAIN_CODE_ROOT = MS_SWIFT_ROOT.parent
PROJECT_ROOT = TRAIN_CODE_ROOT.parent
DEFAULT_OUTPUT_DIR = MS_SWIFT_ROOT / "data"


def as_posix(path):
    return str(path).replace(os.sep, "/")


def remap_legacy_absolute(path: Path, project_root: Path) -> Path:
    """Map old /oddgrid_task/{IOL,SOI}_type paths into OddGridBench_clean."""
    parts = path.parts
    for data_root in ("IOL_type", "SOI_type"):
        if data_root in parts and "OddGridBench_clean" not in parts:
            idx = parts.index(data_root)
            return project_root.joinpath(*parts[idx:])
    return path


def normalize_image_path(image, relative_base: Path, project_root: Path, image_base_dir: Path | None):
    path = Path(str(image))
    if path.is_absolute():
        path = remap_legacy_absolute(path, project_root)
        return as_posix(os.path.relpath(path, relative_base))

    if image_base_dir is not None:
        path = image_base_dir / path
        return as_posix(os.path.relpath(path, relative_base))

    return as_posix(path)


def normalize_images(images, relative_base: Path, project_root: Path, image_base_dir: Path | None):
    return [normalize_image_path(image, relative_base, project_root, image_base_dir) for image in images or []]


def output_path_for(input_path: Path, explicit_output: Path | None, output_dir: Path) -> Path:
    if explicit_output is not None:
        return explicit_output
    return output_dir / input_path.name


def convert_file(input_path: Path, output_path: Path, limit: int, relative_base: Path, project_root: Path,
                 image_base_dir: Path | None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if limit >= 0 and count >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if messages is None:
                messages = [{"role": "user", "content": item["problem"]}]
            out = {
                "messages": messages,
                "images": normalize_images(item.get("images", []), relative_base, project_root, image_base_dir),
                "solution": item.get("answer", item.get("solution", "")),
                "data_type": item.get("data_type", ""),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def add_legacy_jobs(args, jobs):
    if args.train_input is None and args.val_input is None:
        return
    if args.train_input is None or args.val_input is None:
        raise ValueError("--train-input and --val-input must be provided together.")
    jobs.append((args.train_input, output_path_for(args.train_input, args.train_output, args.output_dir),
                 args.train_limit))
    jobs.append((args.val_input, output_path_for(args.val_input, args.val_output, args.output_dir), args.val_limit))


def main():
    parser = argparse.ArgumentParser(description="Convert EasyR1 OddGrid JSONL to ms-swift GRPO JSONL.")
    parser.add_argument("--inputs", nargs="+", type=Path, default=[],
                        help="One or more JSONL files to convert. Output filenames mirror input filenames.")
    parser.add_argument("--limit", type=int, default=-1, help="Sample limit for --inputs. Use -1 for all samples.")
    parser.add_argument("--train-input", type=Path, default=None)
    parser.add_argument("--val-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory. Filenames default to the corresponding input filename.")
    parser.add_argument("--train-output", type=Path, default=None,
                        help="Optional explicit train output path; overrides --output-dir/input_name.")
    parser.add_argument("--val-output", type=Path, default=None,
                        help="Optional explicit val output path; overrides --output-dir/input_name.")
    parser.add_argument("--train-limit", type=int, default=16, help="Use -1 for all samples.")
    parser.add_argument("--val-limit", type=int, default=8, help="Use -1 for all samples.")
    parser.add_argument("--relative-base", type=Path, default=MS_SWIFT_ROOT,
                        help="Base directory used when converting absolute image paths to relative paths.")
    parser.add_argument("--image-base-dir", type=Path, default=None,
                        help="Optional base for resolving relative image paths before re-relativizing them.")
    args = parser.parse_args()

    relative_base = args.relative_base.resolve()
    project_root = PROJECT_ROOT.resolve()
    image_base_dir = args.image_base_dir.resolve() if args.image_base_dir is not None else None

    jobs = [(input_path, args.output_dir / input_path.name, args.limit) for input_path in args.inputs]
    try:
        add_legacy_jobs(args, jobs)
    except ValueError as exc:
        parser.error(str(exc))
    if not jobs:
        parser.error("Provide --inputs, or provide --train-input and --val-input.")

    for input_path, output_path, limit in jobs:
        count = convert_file(input_path, output_path, limit, relative_base, project_root, image_base_dir)
        print(f"Wrote {count} samples to {output_path}")


if __name__ == "__main__":
    main()
