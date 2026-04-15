import os
import json
import ast
import csv
from glob import glob

BASE_DIR = "./"
TYPES = ["single_results", "iol_output", "soi_output"]
SAVE_DIR = "./results"


def parse_grid_size(grid_str):
    try:
        nums = ast.literal_eval(grid_str)
        return nums[0] * nums[1]
    except:
        return 1


def get_num_images(item, data_type):
    if data_type.startswith("single"):
        return 1
    elif data_type.startswith("iol"):
        return parse_grid_size(item.get("grid_size", "[1,1]"))
    elif data_type.startswith("soi"):
        return item.get("image_num", 1)
    return 1


def process_file(file_path, data_type):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_time = 0
    total_images = 0

    for item in data:
        t = item.get("inference_time", None)
        if t is None:
            continue

        num_images = get_num_images(item, data_type)

        total_time += t
        total_images += num_images

    if total_images == 0:
        return None

    return {
        "file": os.path.basename(file_path),
        "samples": len(data),
        "total_images": total_images,
        "total_time": round(total_time, 4),
        "avg_per_image_time": round(total_time / total_images, 6)
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for t in TYPES:
        dir_path = os.path.join(BASE_DIR, t)
        files = glob(os.path.join(dir_path, "*.json"))

        rows = []

        print(f"\n===== Processing {t} =====")

        for fpath in files:
            res = process_file(fpath, t)
            if res is None:
                continue

            rows.append(res)

            print(f"{res['file']} -> {res['avg_per_image_time']}")

        # ===== 写该类型的 CSV =====
        csv_path = os.path.join(SAVE_DIR, f"{t}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file",
                "samples",
                "total_images",
                "total_time",
                "avg_per_image_time"
            ])
            writer.writeheader()
            writer.writerows(rows)

        print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()