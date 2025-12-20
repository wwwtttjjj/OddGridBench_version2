import json
import os
import sys
import random

# 允许从上级目录 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.utils import build_prompt


def convert_dataset(
    in_json_path: str,
    image_root: str,
    out_json_path: str,
    max_num: int = None
):
    """
    将原始数据转换为 sharegpt 格式并保存

    参数:
    - in_json_path: 输入 json 路径 (train / test 均可)
    - image_root: 图片根目录
    - out_json_path: 输出 json 保存路径
    - max_num: 最多转换多少条（None 表示不限制）
    """

    with open(in_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    processed = []

    for item in samples:
        answer = item.get("answer", "")
        image = item.get("image", "")
        odd_rows_cols = item.get("odd_rows_cols", [])

        # ✅ 打乱顺序
        random.shuffle(odd_rows_cols)

        # ✅ 生成最终 boxed 答案
        answer = ",".join([f"({r},{c})" for r, c in odd_rows_cols])

        # ✅ 构造 prompt
        prompt = build_prompt(item)

        # ✅ 转换为绝对路径
        image_abs = os.path.abspath(
            os.path.join(image_root, os.path.basename(image))
        )

        processed.append({
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{prompt}"
                },
                {
                    "from": "gpt",
                    "value": f"\\boxed{{{answer}}}"
                }
            ],
            "image": image_abs
        })

        if max_num is not None and len(processed) >= max_num:
            break

    # ✅ 保存
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成: {out_json_path}, 共 {len(processed)} 条数据")

if __name__ == "__main__":
    train_json_path = "../create_data/train_data.json"
    test_json_path  = "../create_data/test_data.json"

    train_image_dir = "/data/wengtengjin/colorsense/create_data/train_data/image"
    test_image_dir  = "/data/wengtengjin/colorsense/create_data/test_data/image"

    train_out = "./train_sft_qa.json"
    test_out  = "./test_sft_qa.json"

    convert_dataset(train_json_path, train_image_dir, train_out, max_num=15000000)
    convert_dataset(test_json_path,  test_image_dir,  test_out,  max_num=None)
