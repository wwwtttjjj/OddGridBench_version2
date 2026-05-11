import json
from pathlib import Path


def count_real_source_images(data):
    if not isinstance(data, list):
        return 0

    total = 0
    for entry in data:
        if "source_cells" in entry:
            for cell in entry.get("source_cells", []):
                if cell.get("original_name") is not None:
                    total += 1
        elif "source_images_details" in entry:
            for image_detail in entry.get("source_images_details", []):
                if image_detail.get("original_filename") is not None:
                    total += 1

    return total


def count_combined_statistics(root_dir="."):
    root_path = Path(root_dir)
    total_iol = 0
    total_soi = 0
    total_iol_images = 0
    total_soi_images = 0

    task_map = {
        "IOL": "iol_test_data/iol_test_data.json",
        "SOI": "soi_test_data/soi_test_data.json",
    }

    header = (
        f"{'Dataset Name':<25} | "
        f"{'IOL Count':<12} | {'IOL Images':<12} | "
        f"{'SOI Count':<12} | {'SOI Images':<12}"
    )
    print(header)
    print("-" * len(header))

    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    for subdir in subdirs:
        counts = {"IOL": 0, "SOI": 0}
        image_counts = {"IOL": 0, "SOI": 0}
        has_data = False

        for task_label, rel_path in task_map.items():
            target_json = subdir / rel_path

            if target_json.exists():
                has_data = True
                try:
                    with target_json.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    current_count = len(data) if isinstance(data, list) else 0
                    current_image_count = count_real_source_images(data)
                    counts[task_label] = current_count
                    image_counts[task_label] = current_image_count

                    if task_label == "IOL":
                        total_iol += current_count
                        total_iol_images += current_image_count
                    else:
                        total_soi += current_count
                        total_soi_images += current_image_count
                except Exception:
                    counts[task_label] = "Error"
                    image_counts[task_label] = "Error"
            else:
                counts[task_label] = "-"
                image_counts[task_label] = "-"

        if has_data:
            print(
                f"{subdir.name:<25} | "
                f"{str(counts['IOL']):<12} | {str(image_counts['IOL']):<12} | "
                f"{str(counts['SOI']):<12} | {str(image_counts['SOI']):<12}"
            )

    print("-" * len(header))
    print(
        f"{'TOTAL SAMPLES':<25} | "
        f"{total_iol:<12} | {total_iol_images:<12} | "
        f"{total_soi:<12} | {total_soi_images:<12}"
    )


if __name__ == "__main__":
    count_combined_statistics()
