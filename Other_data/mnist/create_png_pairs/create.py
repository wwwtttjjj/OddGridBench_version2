import random
import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# =========================
# 可调参数
# =========================

MNIST_ROOT = Path("../mnist_png")
OUTPUT_ROOT = Path("./mnist_pairs")

TOTAL_PAIRS = 100

INIT_MIN_SSIM = 0.80
MIN_SSIM_FLOOR = 0.70
SSIM_DECAY = 0.02

MAX_TRIES_PER_ROUND = 3000
MAX_STAGNATION = 4

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

SEED = random.randint(0, 10000)
GRAYSCALE = True
RESIZE_TO = None


# =========================
# 工具函数
# =========================

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    if GRAYSCALE:
        img = img.convert("L")
    if RESIZE_TO is not None:
        img = img.resize(RESIZE_TO)
    return np.array(img)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, data_range=img2.max() - img2.min())


def collect_images(digit_dir: Path) -> List[Path]:
    return [
        p for p in digit_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]


# =========================
# 主逻辑
# =========================

def main():
    random.seed(SEED)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    digit_dirs = [MNIST_ROOT / str(i) for i in range(10)]
    digit_dirs = [d for d in digit_dirs if d.exists()]
    digits = [d.name for d in digit_dirs]

    digit_to_images = {
        d.name: collect_images(d)
        for d in digit_dirs
    }

    PAIRS_PER_DIGIT = TOTAL_PAIRS // len(digits)
    digit_pair_count = {d: 0 for d in digits}

    collected = 0

    print(f"[INFO] seed={SEED}")
    print(f"[INFO] target={PAIRS_PER_DIGIT} pairs per digit")

    for digit in digits:
        img_paths = digit_to_images[digit]
        cur_min_ssim = INIT_MIN_SSIM
        stagnation = 0

        while (
            digit_pair_count[digit] < PAIRS_PER_DIGIT
            and len(img_paths) >= 2
            and cur_min_ssim >= MIN_SSIM_FLOOR
        ):
            found = False

            for _ in range(MAX_TRIES_PER_ROUND):
                p1, p2 = random.sample(img_paths, 2)
                score = compute_ssim(load_image(p1), load_image(p2))

                if score >= cur_min_ssim:
                    img_paths.remove(p1)
                    img_paths.remove(p2)

                    pair_id = f"{digit}_{uuid.uuid4().hex[:6]}"
                    out_dir = OUTPUT_ROOT / pair_id
                    out_dir.mkdir(parents=True)

                    shutil.copy(p1, out_dir / p1.name)
                    shutil.copy(p2, out_dir / p2.name)

                    digit_pair_count[digit] += 1
                    collected += 1
                    found = True
                    stagnation = 0
                    break

            if not found:
                stagnation += 1
                random.shuffle(img_paths)

                if stagnation >= MAX_STAGNATION:
                    cur_min_ssim -= SSIM_DECAY
                    stagnation = 0
                    print(
                        f"[WARN] digit {digit}: "
                        f"lower MIN_SSIM -> {cur_min_ssim:.2f}"
                    )

        print(
            f"[INFO] digit {digit}: "
            f"{digit_pair_count[digit]}/{PAIRS_PER_DIGIT} pairs"
        )

    print(f"[DONE] total collected: {collected}/{TOTAL_PAIRS}")


if __name__ == "__main__":
    main()
