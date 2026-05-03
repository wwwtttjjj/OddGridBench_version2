from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import os

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}
MAX_SIDE = 600
NUM_THREADS = 8


def resize_image_keep_ratio(src_path, dst_path, max_side=MAX_SIDE):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        w, h = img.size
        current_max = max(w, h)

        if current_max <= max_side:
            shutil.copy2(src_path, dst_path)
            return

        scale = max_side / current_max
        new_size = (round(w * scale), round(h * scale))

        if img.mode in ("RGBA", "LA"):
            out = img.copy()
        else:
            out = img.convert("RGB")

        out = out.resize(new_size, Image.Resampling.LANCZOS)

        save_kwargs = {}
        if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": 95}

        out.save(dst_path, **save_kwargs)


def process_one_file(src_path, src_root, dst_root, max_side=MAX_SIDE):
    rel_path = src_path.relative_to(src_root)
    dst_path = dst_root / rel_path

    if src_path.suffix.lower() in IMAGE_EXTS:
        resize_image_keep_ratio(src_path, dst_path, max_side)
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def resize_directory(src_root, dst_root, max_side=MAX_SIDE, num_threads=NUM_THREADS):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    all_files = [p for p in src_root.rglob("*") if p.is_file()]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_one_file,
                src_path,
                src_root,
                dst_root,
                max_side
            )
            for src_path in all_files
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Resizing images",
            unit="file"
        ):
            future.result()


if __name__ == "__main__":
    jobs = [
        ("vial", "revised_data/mvtec_ad2_vial_resized"),
        # ("MVTEC_LOCO/", "revised_data/MVTEC_LOCO_resized"),
    ]

    for src_dir, dst_dir in jobs:
        print(f"\n[INFO] Processing: {src_dir} -> {dst_dir}")

        resize_directory(
            src_root=src_dir,
            dst_root=dst_dir,
            max_side=600,
            num_threads=8
        )
