import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resize_image(img, max_size):
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def process_one(task):
    src_path, dst_path, max_size = task
    try:
        img = Image.open(src_path).convert("RGB")
        img = resize_image(img, max_size)
        img.save(dst_path, quality=95)
        return True, None
    except Exception as e:
        return False, str(e)


def process_dataset(input_root, output_root, max_size=600, num_workers=4, verbose=True):
    stats = {
        "total": 0,
        "Normal": 0,
        "Anomaly": 0,
        "error": 0
    }

    counters = {}
    lock = threading.Lock()
    tasks = []

    # ===== 收集任务 =====
    for category in os.listdir(input_root):
        category_path = os.path.join(input_root, category)
        if not os.path.isdir(category_path):
            continue

        if verbose:
            print(f"Scanning category: {category}")

        for split in ["train", "test"]:
            split_path = os.path.join(category_path, split)
            if not os.path.exists(split_path):
                continue

            for sub in os.listdir(split_path):
                sub_path = os.path.join(split_path, sub)
                if not os.path.isdir(sub_path):
                    continue

                label = "Normal" if sub == "good" else "Anomaly"
                out_dir = os.path.join(output_root, category, label)
                os.makedirs(out_dir, exist_ok=True)

                key = (category, label)
                if key not in counters:
                    counters[key] = 0

                for root, _, files in os.walk(sub_path):
                    for file in files:
                        if not file.lower().endswith(IMG_EXTS):
                            continue

                        src_path = os.path.join(root, file)

                        # 🔒 生成唯一文件名（加锁）
                        with lock:
                            counters[key] += 1
                            idx = counters[key]

                        new_name = f"{category}_{label}_{idx:06d}.jpg"
                        dst_path = os.path.join(out_dir, new_name)

                        tasks.append((src_path, dst_path, max_size))
                        stats[label] += 1

    if verbose:
        print(f"Total tasks: {len(tasks)}")

    # ===== 多线程执行 =====
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, t) for t in tasks]

        for future in as_completed(futures):
            success, err = future.result()
            stats["total"] += 1
            if not success:
                stats["error"] += 1
                if verbose:
                    print("Error:", err)

    return stats

import shutil
import os


def safe_clear_dir(path, verbose=True):
    """
    安全清空目录：
    - 必须存在
    - 不能是根目录或空字符串
    """

    if not path or path in ["/", "\\"]:
        raise ValueError(f"危险路径，拒绝删除: {path}")

    if os.path.exists(path):
        if verbose:
            print(f"[INFO] 清空目录: {path}")
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    
    safe_clear_dir("MPDD/manual_images")
    stats = process_dataset(
        input_root="MPDD/Raw_data",
        output_root="MPDD/manual_images",
        max_size=1024,
        num_workers=6,  # 👉 可以调
        verbose=True
    )
    print("\nDone!")
    
    # safe_clear_dir("RAD/manual_images")
    # stats = process_dataset(
    #     input_root="RAD/Raw_data",
    #     output_root="RAD/manual_images",
    #     max_size=1500,
    #     num_workers=6,  # 👉 可以调
    #     verbose=True
    # )
    # print("\nDone!")
