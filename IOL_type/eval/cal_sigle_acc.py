import json
import csv
import re
from pathlib import Path


# =========================
# 工具函数：标准化坐标
# =========================
def normalize_gt(answer):
    if not answer:
        return set()

    out = set()
    for item in answer:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.add((int(item[0]), int(item[1])))

    return out


def normalize_pred(extract_answer):
    """
    返回:
    - set(): 合法预测坐标集合，包括空集合
    - None: 非法格式
    """
    if extract_answer == "" or extract_answer is None:
        return None

    if isinstance(extract_answer, list):
        if len(extract_answer) == 0:
            return set()

        coords = set()
        coord_re = re.compile(r'^\((\d+),(\d+)\)$')

        for s in extract_answer:
            if not isinstance(s, str):
                return None

            m = coord_re.match(s.strip())
            if not m:
                return None

            coords.add((int(m.group(1)), int(m.group(2))))

        return coords

    return None


def parse_grid_size(grid_size):
    """
    兼容:
    - [3, 3]
    - "[3, 3]"
    - 缺失时默认 [3, 3]
    """
    if grid_size is None:
        return [3, 3]

    if isinstance(grid_size, str):
        try:
            grid_size = json.loads(grid_size)
        except Exception:
            return [3, 3]

    if isinstance(grid_size, list) and len(grid_size) == 2:
        return [int(grid_size[0]), int(grid_size[1])]

    return [3, 3]


def get_total_count(sample, grid_size):
    """
    优先使用 sample["total_count"]。
    如果没有 total_count，则尝试从 source_cells 里统计 original_name 不为 null 的数量。
    如果还没有，则回退到 grid_size[0] * grid_size[1]。
    """
    if sample.get("total_count") is not None:
        return int(sample["total_count"])

    source_cells = sample.get("source_cells", None)
    if isinstance(source_cells, list):
        return sum(
            1 for cell in source_cells
            if cell.get("original_name") is not None
        )

    return int(grid_size[0]) * int(grid_size[1])


def valid_cells_from_total_count(grid_size, total_count):
    """
    按 row-major 顺序生成有效 cell。
    例如 grid_size=[3,3], total_count=7:
    有效 cell 为:
    (1,1),(1,2),(1,3),
    (2,1),(2,2),(2,3),
    (3,1)
    """
    rows, cols = grid_size
    valid_cells = []

    for idx in range(total_count):
        r = idx // cols + 1
        c = idx % cols + 1

        # 防止 total_count 异常超过完整 grid
        if r <= rows:
            valid_cells.append((r, c))

    return valid_cells


# =========================
# 核心计算逻辑：基于有效单元格的二分类指标
# =========================
def compute_cell_confusion_matrix(pred_set, gt_set, grid_size, total_count):
    """
    只统计 total_count 个有效 cell，而不是完整 grid_size。
    """
    tp, fp, tn, fn = 0, 0, 0, 0

    is_invalid = pred_set is None
    current_pred = pred_set if not is_invalid else set()

    valid_cells = valid_cells_from_total_count(grid_size, total_count)
    valid_cell_set = set(valid_cells)

    # 如果模型预测了缺失 cell，应该算 FP
    # 例如 total_count=7，但预测了 (3,3)，这个位置不存在，算错误预测
    extra_pred_cells = current_pred - valid_cell_set
    fp += len(extra_pred_cells)

    for cell in valid_cells:
        is_pred_odd = cell in current_pred
        is_gt_odd = cell in gt_set

        if is_gt_odd and is_pred_odd:
            tp += 1
        elif is_gt_odd and not is_pred_odd:
            fn += 1
        elif not is_gt_odd and is_pred_odd:
            fp += 1
        else:
            tn += 1

    return tp, fp, tn, fn


# =========================
# 文件与目录处理
# =========================
def eval_json_file(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    img_count = len(data)

    for sample in data:
        gt_set = normalize_gt(sample.get("answer", []))
        pred_set = normalize_pred(sample.get("extract_answer", ""))

        grid_size = parse_grid_size(sample.get("grid_size", [3, 3]))
        total_count = get_total_count(sample, grid_size)

        tp, fp, tn, fn = compute_cell_confusion_matrix(
            pred_set=pred_set,
            gt_set=gt_set,
            grid_size=grid_size,
            total_count=total_count
        )

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    total_cells = total_tp + total_fp + total_tn + total_fn

    acc = (total_tp + total_tn) / total_cells if total_cells > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "images": img_count,
        "total_cells": total_cells,
        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def eval_all(json_dir_list, out_csv):
    results = []

    for dir_path in json_dir_list:
        path_obj = Path(dir_path)
        json_files = sorted(path_obj.glob("*.json"))

        for jp in json_files:
            m = eval_json_file(jp)

            print(f"Model: {jp.stem:30}")
            print(
                f" ├─ Cells: {m['total_cells']} "
                f"(Odd GT: {m['tp'] + m['fn']} | Normal GT: {m['tn'] + m['fp']})"
            )
            print(
                f" └─ Metrics: "
                f"Acc: {m['acc'] * 100:.2f}% | "
                f"P: {m['precision'] * 100:.2f}% | "
                f"R: {m['recall'] * 100:.2f}% | "
                f"F1: {m['f1'] * 100:.2f}%"
            )
            print(
                f" └─ Detail: "
                f"TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}\n"
            )

            results.append({
                "model": jp.stem,
                "Images": m["images"],
                "Total_Cells": m["total_cells"],
                "Acc": f"{m['acc'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall": f"{m['recall'] * 100:.2f}%",
                "F1": f"{m['f1'] * 100:.2f}%",
                "TP": m["tp"],
                "FP": m["fp"],
                "TN": m["tn"],
                "FN": m["fn"]
            })

    fieldnames = [
        "model",
        "Images",
        "Total_Cells",
        "Acc",
        "Precision",
        "Recall",
        "F1",
        "TP",
        "FP",
        "TN",
        "FN"
    ]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# =========================
# 执行部分
# =========================
if __name__ == "__main__":
    target_dirs = [
        "VisA_output",
        "MVTEC_output",
        "BTech_output",
        "ELPV_output",
        "hanzi_output",
        "icon_output",
        "mnist_output",
        "MVTEC_loco_output",
        "GOODADS_output",
        "RAD_output",
        "MPDD_output"
    ]

    out_root = Path("results_single")
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== 开始全量指标评估 (Cell-level, using total_count) ===\n")

    for json_dir in target_dirs:
        folder_name = json_dir.rstrip("/").replace("/", "_")
        out_csv_path = out_root / f"{folder_name}_cell_metrics.csv"

        print(f"--- 正在处理目录: {json_dir} ---")
        eval_all([json_dir], str(out_csv_path))
        print(f"✅ 该目录评估完成，结果保存至: {out_csv_path}\n")

    print(f"🚀 所有任务处理完毕，请在 '{out_root}' 文件夹下查看结果。")