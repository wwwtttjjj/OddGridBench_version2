import json
import csv
import re
from pathlib import Path


# =========================
# Normalize GT: answer -> List[int]
# =========================
def normalize_gt(answer):
    """
    []        -> []
    [2, 5]    -> [2, 5]
    """
    if not answer:
        return []
    return [int(x) for x in answer]


# =========================
# Normalize Pred: extract_answer -> List[int] | None
# None 表示格式错误
# =========================
_IMAGE_RE = re.compile(r"^image(\d+)$")

def normalize_pred(extract_answer):
    """
    合法：
      "image2" -> [2]
      "image2, image5" -> [2,5]
      ["image2", "image5"] -> [2,5]
      [] -> []
    非法：
      "" / None / 乱格式 -> None
    """
    if extract_answer is None:
        return None

    # 空 list：合法，表示无异常
    if extract_answer == []:
        return []

    indices = []

    # -------- case 1: list[str] --------
    if isinstance(extract_answer, list):
        for p in extract_answer:
            if not isinstance(p, str):
                return None
            m = _IMAGE_RE.match(p.strip())
            if not m:
                return None
            indices.append(int(m.group(1)))
        return indices

    # -------- case 2: str --------
    if not isinstance(extract_answer, str):
        return None

    s = extract_answer.strip()
    if s == "":
        return None

    parts = [p.strip() for p in s.split(",")]
    for p in parts:
        m = _IMAGE_RE.match(p)
        if not m:
            return None
        indices.append(int(m.group(1)))

    return indices


# =========================
# EM / F1
# =========================
def compute_em_f1(pred, gt):
    """
    pred: List[int] | None
    gt:   List[int]
    """
    # 格式错误
    if pred is None:
        return 0, 0.0

    pred_set = set(pred)
    gt_set = set(gt)

    # 无异常 & 判断正确
    if len(gt_set) == 0 and len(pred_set) == 0:
        return 1, 1.0

    # EM
    em = int(pred_set == gt_set)

    # F1
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return em, f1


# =========================
# Evaluate one json file
# =========================
def eval_json_file(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    em_sum = 0.0
    f1_sum = 0.0
    n = 0

    for sample in data:
        gt = normalize_gt(sample.get("answer", []))
        pred = normalize_pred(sample.get("extract_answer", ""))

        em, f1 = compute_em_f1(pred, gt)
        print(f"ID: {sample.get('id')}, EM: {em}, F1: {f1:.3f}, GT: {gt}, Pred: {pred}")
        em_sum += em
        f1_sum += f1
        n += 1

    return (
        em_sum / n if n else 0.0,
        f1_sum / n if n else 0.0,
    )


# =========================
# Evaluate a directory
# =========================
def eval_json_dir(json_dir: str, out_csv: str):
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        return

    results = {}

    for jp in json_files:
        em, f1 = eval_json_file(jp)
        results[jp.stem] = {"EM": em, "F1": f1}

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "EM", "F1"])
        for model_name, m in results.items():
            writer.writerow([
                model_name,
                f"{m['EM']*100:.2f}",
                f"{m['F1']*100:.2f}",
            ])

    print(f"✅ Saved CSV to: {out_csv}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    json_dirs = [
        # "mnist_output/",
        # "hanzi_output/",
        # "icon_output/",
        "VisA_output/",
        "BTech_output",
        "MVTEC_output/",
        
    ]

    out_root = Path("results_emf1")
    out_root.mkdir(parents=True, exist_ok=True)

    for json_dir in json_dirs:
        json_dir = json_dir.rstrip("/")
        out_csv = out_root / f"{json_dir}_results.csv"
        print(f"\n=== Evaluating {json_dir}/ ===")
        eval_json_dir(f"{json_dir}/", str(out_csv))
