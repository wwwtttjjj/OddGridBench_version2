import json
from pathlib import Path


ABNORMAL_LABELS = {"abnormal", "anomaly", "defect", "defective", "ng", "odd"}
NORMAL_LABELS = {"normal", "good", "ok"}


def empty_image_counts():
    return {"normal": 0, "abnormal": 0, "total": 0}


def add_image_count(counts, label):
    counts["total"] += 1
    normalized = str(label or "").strip().lower()
    if normalized in NORMAL_LABELS:
        counts["normal"] += 1
    elif normalized in ABNORMAL_LABELS:
        counts["abnormal"] += 1


def count_real_source_images(data):
    counts = empty_image_counts()
    if not isinstance(data, list):
        return counts

    for entry in data:
        if "source_cells" in entry:
            for cell in entry.get("source_cells", []):
                if cell.get("original_name") is not None:
                    add_image_count(counts, cell.get("label"))
        elif "source_images_details" in entry:
            for image_detail in entry.get("source_images_details", []):
                if image_detail.get("original_filename") is not None:
                    add_image_count(counts, image_detail.get("label"))

    return counts


def add_counts(total_counts, current_counts):
    for key in ("normal", "abnormal", "total"):
        total_counts[key] += current_counts[key]


def format_image_counts(counts):
    if isinstance(counts, dict):
        return "{:<12} | {:<12} | {:<12}".format(
            counts["normal"], counts["abnormal"], counts["total"]
        )
    return "{:<12} | {:<12} | {:<12}".format(str(counts), str(counts), str(counts))


def count_combined_statistics(root_dir="."):
    root_path = Path(root_dir)
    total_iol = 0
    total_soi = 0
    total_iol_images = empty_image_counts()
    total_soi_images = empty_image_counts()

    task_map = {
        "IOL": "iol_test_data/iol_test_data.json",
        "SOI": "soi_test_data/soi_test_data.json",
    }

    header = (
        "{:<25} | ".format("Dataset Name")
        + "{:<12} | ".format("IOL Count")
        + "{:<12} | {:<12} | {:<12} | ".format("IOL Normal", "IOL Abnormal", "IOL Images")
        + "{:<12} | ".format("SOI Count")
        + "{:<12} | {:<12} | {:<12}".format("SOI Normal", "SOI Abnormal", "SOI Images")
    )
    print(header)
    print("-" * len(header))

    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    for subdir in subdirs:
        counts = {"IOL": 0, "SOI": 0}
        image_counts = {"IOL": empty_image_counts(), "SOI": empty_image_counts()}
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
                        add_counts(total_iol_images, current_image_count)
                    else:
                        total_soi += current_count
                        add_counts(total_soi_images, current_image_count)
                except Exception:
                    counts[task_label] = "Error"
                    image_counts[task_label] = "Error"
            else:
                counts[task_label] = "-"
                image_counts[task_label] = "-"

        if has_data:
            print(
                "{:<25} | ".format(subdir.name)
                + "{:<12} | {} | ".format(str(counts["IOL"]), format_image_counts(image_counts["IOL"]))
                + "{:<12} | {}".format(str(counts["SOI"]), format_image_counts(image_counts["SOI"]))
            )

    print("-" * len(header))
    print(
        "{:<25} | ".format("TOTAL SAMPLES")
        + "{:<12} | {} | ".format(total_iol, format_image_counts(total_iol_images))
        + "{:<12} | {}".format(total_soi, format_image_counts(total_soi_images))
    )


if __name__ == "__main__":
    count_combined_statistics()
