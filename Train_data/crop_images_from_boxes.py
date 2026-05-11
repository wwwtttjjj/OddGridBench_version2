import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ======================
# 基础工具函数
# ======================
def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))


def looks_like_qwen1000(box):
    """Heuristic: Qwen3-VL 0~1000 relative coords"""
    try:
        xs = list(map(float, box))
    except Exception:
        return False
    return all(0.0 <= v <= 1000.0 for v in xs)


def looks_like_norm01(box):
    """Heuristic: 0~1 normalized coords"""
    try:
        xs = list(map(float, box))
    except Exception:
        return False
    return all(0.0 <= v <= 1.0 for v in xs)


def convert_box_to_pixels(box, W, H, mode="auto"):
    """
    box: [x1,y1,x2,y2] in pixel | qwen1000 | norm01
    return: (x1,y1,x2,y2, used_mode) in pixel (int)
    """
    x1, y1, x2, y2 = map(float, box)

    if mode == "auto":
        if looks_like_norm01([x1, y1, x2, y2]):
            mode_eff = "norm01"
        elif looks_like_qwen1000([x1, y1, x2, y2]) and (W != 1000 or H != 1000):
            mode_eff = "qwen1000"
        else:
            mode_eff = "pixel"
    else:
        mode_eff = mode

    if mode_eff == "qwen1000":
        x1 = x1 * W / 1000.0
        x2 = x2 * W / 1000.0
        y1 = y1 * H / 1000.0
        y2 = y2 * H / 1000.0
    elif mode_eff == "norm01":
        x1 = x1 * W
        x2 = x2 * W
        y1 = y1 * H
        y2 = y2 * H
    elif mode_eff == "pixel":
        pass
    else:
        raise ValueError(f"Unknown box mode: {mode_eff}")

    x1, y1, x2, y2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))
    return x1, y1, x2, y2, mode_eff


# ======================
# bbox 尺寸统计
# ======================
def collect_valid_boxes(data, image_root, box_mode="auto", padding=0):
    """
    第一遍：收集所有有效 bbox 的像素坐标和 crop 尺寸。
    """
    records = []

    for item_idx, item in enumerate(data):
        rel_image_path = item.get("image")
        if not rel_image_path:
            continue

        abs_image_path = os.path.join(image_root, rel_image_path)
        if not os.path.exists(abs_image_path):
            print(f"[WARN] Image not found: {abs_image_path}")
            continue

        boxes = item.get("boxes", [])
        if not boxes:
            continue

        try:
            img = Image.open(abs_image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image: {abs_image_path}, err={e}")
            continue

        W, H = img.size

        for box_idx, obj in enumerate(boxes):
            box = obj.get("box")
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2, used_mode = convert_box_to_pixels(
                box, W, H, mode=box_mode
            )

            # padding
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

            # clamp
            x1 = clamp(x1, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            x2 = clamp(x2, x1 + 1, W)
            y2 = clamp(y2, y1 + 1, H)

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w <= 1 or crop_h <= 1:
                continue

            records.append({
                "item_idx": item_idx,
                "rel_image_path": rel_image_path,
                "box_idx": box_idx,
                "num_boxes": len(boxes),
                "xyxy": (x1, y1, x2, y2),
                "img_size": (W, H),
                "crop_size": (crop_w, crop_h),
                "used_mode": used_mode,
            })

    return records


def median(values):
    """
    简单 median，避免额外依赖 numpy。
    """
    if not values:
        return None

    values = sorted(values)
    n = len(values)

    if n % 2 == 1:
        return values[n // 2]
    else:
        return (values[n // 2 - 1] + values[n // 2]) / 2.0


def compute_ref_size(records, outlier_ratio=2.5, min_keep_ratio=0.5):
    """
    根据所有 crop 尺寸计算合理 ref_size。

    outlier_ratio:
        允许尺寸相对中位数的最大倍数。
        例如 2.5 表示：
        width / median_width 需要在 [1/2.5, 2.5] 内；
        height / median_height 也需要在 [1/2.5, 2.5] 内。

    min_keep_ratio:
        如果过滤后剩余太少，则自动回退到全部 records，避免误删太多。
    """
    if not records:
        return None, []

    widths = [r["crop_size"][0] for r in records]
    heights = [r["crop_size"][1] for r in records]

    med_w = median(widths)
    med_h = median(heights)

    if med_w is None or med_h is None:
        return None, []

    filtered = []

    for r in records:
        w, h = r["crop_size"]

        rw = w / max(med_w, 1)
        rh = h / max(med_h, 1)

        valid_w = (1.0 / outlier_ratio) <= rw <= outlier_ratio
        valid_h = (1.0 / outlier_ratio) <= rh <= outlier_ratio

        if valid_w and valid_h:
            filtered.append(r)

    min_keep_num = max(1, int(len(records) * min_keep_ratio))

    if len(filtered) < min_keep_num:
        print(
            f"[WARN] Too many boxes filtered: keep {len(filtered)}/{len(records)}, "
            f"fallback to all records."
        )
        filtered = records

    ref_w = max(r["crop_size"][0] for r in filtered)
    ref_h = max(r["crop_size"][1] for r in filtered)

    print(f"[INFO] Median crop size: ({med_w:.2f}, {med_h:.2f})")
    print(f"[INFO] Filtered boxes: {len(filtered)} / {len(records)}")
    print(f"[INFO] Reference crop size set to: ({ref_w}, {ref_h})")

    return (ref_w, ref_h), filtered


def expand_box_from_center(x1, y1, x2, y2, ref_w, ref_h, W, H):
    """
    以原 bbox 中心为中心，扩展到 ref_w x ref_h。
    如果越界，则整体平移回来。

    注意：
    - 不 resize；
    - 只是扩大裁剪区域；
    - 保证目标物体比例不变。
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_x1 = int(round(cx - ref_w / 2.0))
    new_y1 = int(round(cy - ref_h / 2.0))
    new_x2 = new_x1 + ref_w
    new_y2 = new_y1 + ref_h

    # 如果 ref_w 比原图还大，只能取原图完整宽度
    if ref_w >= W:
        new_x1, new_x2 = 0, W
    else:
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0

        if new_x2 > W:
            shift = new_x2 - W
            new_x1 -= shift
            new_x2 = W

    # 如果 ref_h 比原图还大，只能取原图完整高度
    if ref_h >= H:
        new_y1, new_y2 = 0, H
    else:
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0

        if new_y2 > H:
            shift = new_y2 - H
            new_y1 -= shift
            new_y2 = H

    new_x1 = clamp(new_x1, 0, W - 1)
    new_y1 = clamp(new_y1, 0, H - 1)
    new_x2 = clamp(new_x2, new_x1 + 1, W)
    new_y2 = clamp(new_y2, new_y1 + 1, H)

    return new_x1, new_y1, new_x2, new_y2


# ======================
# 核心裁剪逻辑
# ======================
def crop_from_json(
    json_path,
    image_root,
    save_root,
    box_mode="auto",
    padding=0,
    outlier_ratio=2.5,
    min_keep_ratio=0.5,
):
    """
    新裁剪逻辑：

    1. 第一遍统计所有 bbox 的 crop size；
    2. 用中位数作为中心尺寸；
    3. 剔除和中心尺寸差距很大的异常值；
    4. 从剩下的合理尺寸中取最大 width/height 作为 ref_size；
    5. 第二遍裁剪时，以 bbox 中心为中心，把小 crop 扩展到 ref_size；
    6. 不使用 resize，避免改变物体比例。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} records")
    print(f"[INFO] box_mode={box_mode}, padding={padding}px")
    print(f"[INFO] outlier_ratio={outlier_ratio}, min_keep_ratio={min_keep_ratio}")

    records = collect_valid_boxes(
        data=data,
        image_root=image_root,
        box_mode=box_mode,
        padding=padding,
    )

    if not records:
        print(f"[WARN] No valid boxes found in {json_path}")
        return

    ref_size, filtered_records = compute_ref_size(
        records=records,
        outlier_ratio=outlier_ratio,
        min_keep_ratio=min_keep_ratio,
    )

    if ref_size is None:
        print(f"[WARN] Failed to compute ref_size for {json_path}")
        return

    ref_w, ref_h = ref_size

    done_count = 0

    for r in records:
        rel_image_path = r["rel_image_path"]
        abs_image_path = os.path.join(image_root, rel_image_path)

        if not os.path.exists(abs_image_path):
            print(f"[WARN] Image not found during crop: {abs_image_path}")
            continue

        try:
            img = Image.open(abs_image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image: {abs_image_path}, err={e}")
            continue

        W, H = img.size
        x1, y1, x2, y2 = r["xyxy"]

        ex1, ey1, ex2, ey2 = expand_box_from_center(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            ref_w=ref_w,
            ref_h=ref_h,
            W=W,
            H=H,
        )

        crop = img.crop((ex1, ey1, ex2, ey2))

        out_dir = os.path.join(save_root, os.path.dirname(rel_image_path))
        os.makedirs(out_dir, exist_ok=True)

        base_name, ext = os.path.splitext(os.path.basename(rel_image_path))

        if r["num_boxes"] > 1:
            save_name = f"{base_name}_box{r['box_idx']}{ext}"
        else:
            save_name = f"{base_name}{ext}"

        save_path = os.path.join(out_dir, save_name)
        crop.save(save_path)

        done_count += 1

    print(f"[INFO] Saved {done_count} crops.")
    print("[INFO] Finished one dataset.")


# ======================
# 单个子目录处理函数
# ======================
def process_one_image_path(
    image_path,
    output_root="A_cropped_images",
    box_mode="qwen1000",
    padding=0,
    outlier_ratio=2.5,
    min_keep_ratio=0.5,
):
    image_path = Path(image_path)
    image_root = image_path
    image_name = image_path.name

    save_root = Path(output_root) / image_name
    json_path = image_path / f"{image_name}.json"

    if not json_path.exists():
        print(f"[WARN] JSON not found, skip: {json_path}")
        return

    # 清空输出目录
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Processing dataset: {image_path}")

    crop_from_json(
        json_path=str(json_path),
        image_root=str(image_root),
        save_root=str(save_root),
        box_mode=box_mode,
        padding=padding,
        outlier_ratio=outlier_ratio,
        min_keep_ratio=min_keep_ratio,
    )


def find_image_path_list(input_root):
    """
    自动扫描 input_root 下所有一级子目录。

    默认要求：
        manual_images/pcb1/pcb1.json
        manual_images/pcb2/pcb2.json
        manual_images/cashew/cashew.json

    只有存在同名 json 的子目录才会被处理。
    """
    input_root = Path(input_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    image_path_list = []

    for p in sorted(input_root.iterdir()):
        if not p.is_dir():
            continue

        json_path = p / f"{p.name}.json"
        if json_path.exists():
            image_path_list.append(str(p))
        else:
            print(f"[WARN] Skip directory without same-name json: {p}")

    return image_path_list


# ======================
# 多线程主入口
# ======================
def main():
    print(f"[INFO] Scanning image roots...")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--box_mode",
        type=str,
        default="qwen1000",
        choices=["auto", "qwen1000", "norm01", "pixel"],
        help="Bounding box coordinate mode."
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding size in pixels."
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of worker threads."
    )

    parser.add_argument(
        "--outlier_ratio",
        type=float,
        default=2.5,
        help=(
            "Filter crop sizes far from median. "
            "Larger means looser filtering. "
            "Example: 2.5 keeps sizes within [median/2.5, median*2.5]."
        )
    )

    parser.add_argument(
        "--min_keep_ratio",
        type=float,
        default=0.5,
        help=(
            "If filtered records are fewer than this ratio, "
            "fallback to all records."
        )
    )

    args = parser.parse_args()

    image_roots = [
        Path("RAD/manual_images"),
        Path("MPDD/manual_images"),
        Path("GOODADS/manual_images"),
        Path("mvtec_ad2/manual_images"),
    ]

    all_tasks = []

    for input_root in image_roots:
        if not input_root.exists():
            print(f"[WARN] Input root not found, skip: {input_root}")
            continue

        # RAD/manual_images -> RAD/A_cropped_images
        output_root = input_root.parent / "A_cropped_images"

        image_path_list = find_image_path_list(input_root)

        if not image_path_list:
            print(f"[WARN] No valid dataset directories found under: {input_root}")
            continue

        print(f"[INFO] Found {len(image_path_list)} datasets under {input_root}:")
        for p in image_path_list:
            print(f"  - {p}")

        for image_path in image_path_list:
            all_tasks.append(
                {
                    "image_path": image_path,
                    "output_root": output_root,
                }
            )

    if not all_tasks:
        print("[WARN] No valid datasets found in all image_roots.")
        return

    print(f"\n[INFO] Total datasets to process: {len(all_tasks)}")

    max_workers = min(args.max_workers, len(all_tasks))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_image_path,
                task["image_path"],
                str(task["output_root"]),
                args.box_mode,
                args.padding,
                args.outlier_ratio,
                args.min_keep_ratio,
            )
            for task in all_tasks
        ]

        for fut in as_completed(futures):
            fut.result()

    print("\n[INFO] All datasets processed.")


if __name__ == "__main__":
    main()