import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.8
SEED = 42

ROOT = Path(__file__).parent          # ELPV/Raw_data
LABELS_PATH = ROOT / "labels.csv"

# 正确的输出目录（与 Raw_data 同级）
OUT_ROOT = ROOT.parent / "ELPV_split"

random.seed(SEED)


def main():
    # ===== 清空输出目录 =====
    if OUT_ROOT.exists():
        print(f"[INFO] Clearing existing directory: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    samples = []

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            img_rel, label_str = parts[0], parts[1]
            if label_str not in {"0", "1", "0.0", "1.0"}:
                continue

            samples.append((img_rel, int(float(label_str))))

    print(f"[INFO] Valid samples: {len(samples)}")

    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_RATIO)

    splits = {
        "train": samples[:split_idx],
        "test": samples[split_idx:],
    }

    for split_name, split_samples in splits.items():
        for img_rel, label in split_samples:
            cls = "normal" if label == 1 else "anomaly"
            src = ROOT / img_rel
            dst = OUT_ROOT / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)

            if src.exists():
                shutil.copy2(src, dst / src.name)

    print(f"[DONE] Saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
