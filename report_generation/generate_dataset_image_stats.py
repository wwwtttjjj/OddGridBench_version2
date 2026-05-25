#!/usr/bin/env python3
"""Generate dataset image-count statistics for OddGridBench.

The script is read-only with respect to dataset/code inputs. It writes summary
reports to merged_reports/dataset_image_stats by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SKIP_DATASET_DIRS = {"__pycache__", "total_data_iol", "total_data_soi", "ELPV"}
FIELDNAMES = [
    "dataset",
    "split",
    "initial_image_files",
    "iol_samples",
    "iol_generated_image_files",
    "iol_source_image_slots",
    "soi_samples",
    "soi_generated_image_files",
    "soi_total_icons",
    "notes",
]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if is_image(p))


def image_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [p for p in path.iterdir() if p.is_dir() and p.name == "images"]


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def records_from_json(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        values = list(data.values())
        if values and all(isinstance(v, dict) for v in values):
            return values
        return [data]
    return []


def first_existing_json(paths: Iterable[Path]) -> tuple[Path | None, Any | None]:
    for path in paths:
        data = load_json(path)
        if data is not None:
            return path, data
    return None, None


def count_combined_records(dataset_dir: Path, task: str) -> tuple[int, list[Any], str]:
    if task == "iol":
        candidates = [
            dataset_dir / "A_iol_type_data" / "all_iol_combined_metadata.json",
            dataset_dir / "iol_test_data" / "iol_test_data.json",
        ]
    else:
        candidates = [
            dataset_dir / "A_soi_type_data" / "all_soi_combined_metadata.json",
            dataset_dir / "soi_test_data" / "soi_test_data.json",
        ]

    source_path, data = first_existing_json(candidates)
    if data is None:
        return 0, [], ""
    records = records_from_json(data)
    return len(records), records, str(source_path.relative_to(dataset_dir))


def iol_source_slots(records: list[Any]) -> int:
    total = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        source_cells = item.get("source_cells")
        if isinstance(source_cells, list):
            total += len(source_cells)
            continue
        grid_size = item.get("grid_size")
        if (
            isinstance(grid_size, list)
            and len(grid_size) >= 2
            and isinstance(grid_size[0], int)
            and isinstance(grid_size[1], int)
        ):
            total += grid_size[0] * grid_size[1]
    return total


def soi_icon_slots(records: list[Any]) -> int:
    total = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        total_icons = item.get("total_icons")
        if isinstance(total_icons, int):
            total += total_icons
            continue
        details = item.get("source_images_details")
        if isinstance(details, list):
            total += len(details)
    return total


def unique_sources_from_records(records: Iterable[Any]) -> set[str]:
    sources: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        for key in ("source_cells", "source_images_details"):
            values = item.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict):
                    for source_key in ("source_image", "image", "path", "filename", "file_name"):
                        source_value = value.get(source_key)
                        if source_value:
                            sources.add(str(source_value))
                            break
                elif value:
                    sources.add(str(value))
    return sources


def count_task_images(dataset_dir: Path, task: str) -> int:
    if task == "iol":
        primary = dataset_dir / "iol_test_data" / "images"
        if primary.exists():
            return count_images(primary)
        return sum(count_images(p) for p in image_dirs(dataset_dir / "A_iol_type_data"))

    primary = dataset_dir / "soi_test_data" / "images"
    if primary.exists():
        return count_images(primary)
    return sum(count_images(p) for p in image_dirs(dataset_dir / "A_soi_type_data"))


def count_initial_images(dataset_dir: Path, all_records: list[Any]) -> tuple[int, str]:
    cropped_dir = dataset_dir / "A_cropped_images"
    cropped_count = count_images(cropped_dir)
    if cropped_count:
        return cropped_count, "initial=A_cropped_images"

    synthetic_dirs = [
        "mnist_png",
        "hanzi_png",
        "Raw_data",
        "SimplifiedChinese",
        "A_original_images",
        "original_images",
        "images",
    ]
    for dirname in synthetic_dirs:
        candidate = dataset_dir / dirname
        candidate_count = count_images(candidate)
        if candidate_count:
            return candidate_count, f"initial={dirname}"

    source_count = len(unique_sources_from_records(all_records))
    if source_count:
        return source_count, "initial=unique_sources_in_metadata"
    return 0, "initial=not_found"


def summarize_dataset(dataset_dir: Path, split: str) -> dict[str, Any]:
    iol_samples, iol_records, iol_source = count_combined_records(dataset_dir, "iol")
    soi_samples, soi_records, soi_source = count_combined_records(dataset_dir, "soi")
    initial_count, initial_note = count_initial_images(dataset_dir, iol_records + soi_records)

    notes = [initial_note]
    if iol_source:
        notes.append(f"iol={iol_source}")
    if soi_source:
        notes.append(f"soi={soi_source}")

    return {
        "dataset": dataset_dir.name,
        "split": split,
        "initial_image_files": initial_count,
        "iol_samples": iol_samples,
        "iol_generated_image_files": count_task_images(dataset_dir, "iol"),
        "iol_source_image_slots": iol_source_slots(iol_records),
        "soi_samples": soi_samples,
        "soi_generated_image_files": count_task_images(dataset_dir, "soi"),
        "soi_total_icons": soi_icon_slots(soi_records),
        "notes": "; ".join(notes),
    }


def count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*.json") if p.is_file())


def summarize_create_data(root: Path, task: str, split: str) -> dict[str, Any]:
    split_dir = root / f"{task.upper()}_type" / "create_data" / f"{split}_data"
    metadata_dir = split_dir / "metadata"
    image_dir = split_dir / "image"
    metadata_count = count_json_files(metadata_dir)
    image_count = count_images(image_dir)

    row = {name: 0 for name in FIELDNAMES}
    row["dataset"] = f"Synthetic_{task.upper()}"
    row["split"] = split
    row["initial_image_files"] = ""
    row["notes"] = "create_data; no explicit initial source-image pool"
    if task == "iol":
        row["iol_samples"] = metadata_count
        row["iol_generated_image_files"] = image_count
        row["iol_source_image_slots"] = ""
        row["soi_samples"] = ""
        row["soi_generated_image_files"] = ""
        row["soi_total_icons"] = ""
    else:
        row["iol_samples"] = ""
        row["iol_generated_image_files"] = ""
        row["iol_source_image_slots"] = ""
        row["soi_samples"] = metadata_count
        row["soi_generated_image_files"] = image_count
        row["soi_total_icons"] = image_count
    return row


def numeric(value: Any) -> int:
    return value if isinstance(value, int) else 0


def add_total_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)

    totals: list[dict[str, Any]] = []
    for split, split_rows in sorted(grouped.items()):
        total_row: dict[str, Any] = {"dataset": "TOTAL", "split": split, "notes": "sum of numeric columns"}
        for field in FIELDNAMES:
            if field in {"dataset", "split", "notes"}:
                continue
            total_row[field] = sum(numeric(row.get(field)) for row in split_rows)
        totals.append(total_row)

    all_total: dict[str, Any] = {"dataset": "TOTAL", "split": "all", "notes": "sum of numeric columns"}
    for field in FIELDNAMES:
        if field in {"dataset", "split", "notes"}:
            continue
        all_total[field] = sum(numeric(row.get(field)) for row in rows)
    totals.append(all_total)
    return rows + totals


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, dirname in (("train", "Train_data"), ("test", "Test_data")):
        split_dir = root / dirname
        if not split_dir.exists():
            continue
        for dataset_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if dataset_dir.name in SKIP_DATASET_DIRS:
                continue
            rows.append(summarize_dataset(dataset_dir, split))

    for split in ("train", "val", "test"):
        for task in ("iol", "soi"):
            split_dir = root / f"{task.upper()}_type" / "create_data" / f"{split}_data"
            if split_dir.exists():
                rows.append(summarize_create_data(root, task, split))

    return add_total_rows(rows)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Dataset Image Statistics\n\n")
        f.write("Count definitions:\n")
        f.write("- `initial_image_files`: raw/source images found in the dataset split, preferably under `A_cropped_images`.\n")
        f.write("- `iol_samples`: number of IOL metadata records; one record normally corresponds to one grid image.\n")
        f.write("- `iol_generated_image_files`: actual generated IOL image files on disk.\n")
        f.write("- `iol_source_image_slots`: total source-cell slots used by IOL samples.\n")
        f.write("- `soi_samples`: number of SOI metadata records; one record can contain multiple images/icons.\n")
        f.write("- `soi_generated_image_files`: actual SOI image files on disk, counted recursively.\n")
        f.write("- `soi_total_icons`: sum of `total_icons`/source image details in SOI metadata.\n\n")
        f.write("| " + " | ".join(FIELDNAMES) + " |\n")
        f.write("| " + " | ".join(["---"] * len(FIELDNAMES)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in FIELDNAMES) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "merged_reports" / "dataset_image_stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    rows = collect_rows(root)
    csv_path = out_dir / "dataset_image_stats.csv"
    md_path = out_dir / "dataset_image_stats.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
