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
        if looks_like_qwen1000([x1, y1, x2, y2]) and (W != 1000 or H != 1000):
            mode_eff = "qwen1000"
        elif looks_like_norm01([x1, y1, x2, y2]):
            mode_eff = "norm01"
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
# 核心裁剪逻辑（不变）
# ======================
def crop_from_json(json_path, image_root, save_root, box_mode="auto", padding=0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} records")
    print(f"[INFO] box_mode={box_mode}, padding={padding}px")

    ref_size = None  # (w, h) from first crop

    for item in data:
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

        img = Image.open(abs_image_path).convert("RGB")
        W, H = img.size

        out_dir = os.path.join(save_root, os.path.dirname(rel_image_path))
        os.makedirs(out_dir, exist_ok=True)

        base_name, ext = os.path.splitext(os.path.basename(rel_image_path))

        for obj in boxes:
            box = obj.get("box")
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2, _ = convert_box_to_pixels(
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

            crop = img.crop((x1, y1, x2, y2))

            # unify crop size
            if ref_size is None:
                ref_size = crop.size
                print(f"[INFO] Reference crop size set to {ref_size}")
            elif crop.size != ref_size:
                crop = crop.resize(ref_size, Image.BILINEAR)

            save_name = f"{base_name}{ext}"
            crop.save(os.path.join(out_dir, save_name))

        print(f"[OK] {rel_image_path} -> {len(boxes)} crops")

    print("[INFO] Finished one dataset.")


# ======================
# 单个 image_name 处理函数（不变逻辑）
# ======================
def process_one_image_name(image_path, box_mode="qwen1000", padding=0):
    image_root = image_path
    image_name = os.path.basename(image_path)
    save_root = Path("A_cropped_images") / image_name
    json_path = Path(image_path) / f"{image_name}.json"

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
    )


# ======================
# 多线程主入口
# ======================
def main():
    image_name_list = ["capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor"]
    image_path_list = ["manual_images/" + name for name in image_name_list]
    
    box_mode = "qwen1000"
    padding = 0

    max_workers = min(8, len(image_path_list))  # 保守一点，IO-bound

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_image_name,
                image_path,
                box_mode,
                padding,
            )
            for image_path in image_path_list
        ]

        for fut in as_completed(futures):
            # 如果某个线程异常，这里会直接抛出来
            fut.result()

    print("\n[INFO] All datasets processed.")


if __name__ == "__main__":
    main()
