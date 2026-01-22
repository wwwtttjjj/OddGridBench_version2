import json
import os
import sys
# 允许从上级目录 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.utils import build_prompt


def convert_and_save_dataset(json_path: str, image_dir: str, out_path: str, num: int = None):
    """
    将原始 JSON 数据转换为 EasyR1 / geo3k 格式，并保存为 JSONL 文件。

    Args:
        json_path (str): 输入 JSON 文件路径。
        image_dir (str): 对应图片目录。
        out_path (str): 输出 JSONL 文件路径。
        num (int, optional): 限制输出样本数量（None 表示全部）。

    输出文件格式示例：
    {
        "images": ["/abs/path/to/image.png"],
        "problem": "<image> 这是生成的题目 prompt",
        "answer": "[5,9]--Row 3, Column 2"
    }
    """
    # 读取输入数据
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    processed = []
    for item in samples:
        odd_rows_cols = item.get("odd_rows_cols", [])
        image = item.get("image", "")
        grid_size = item.get("grid_size")
        prompt = build_prompt(item)

        # 转换为绝对路径
        image_abs = os.path.join(image_dir, os.path.basename(image))
        # ✅ 生成最终 boxed 答案
        answer = ",".join([f"({r},{c})" for r, c in odd_rows_cols])
        
        processed.append({
            "images": [image_abs],
            "problem": f"\n <image> {prompt}",
            "answer": f"{answer.strip()}",
            "data_type": "IOL_type"
        })

        if num and len(processed) >= num:
            break

    # 写入 JSONL 文件
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in processed:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成: {out_path}，共 {len(processed)} 条数据")
    return processed


# ===== 示例用法 =====
if __name__ == "__main__":
    train_json = "../create_data/train_data.json"
    train_img_dir = "../../IOL_type/create_data/train_data/image"
    train_out = "./train_rl_data.jsonl"

    val_json = "../create_data/test_data.json"
    val_img_dir = "../../IOL_type/create_data/test_data/image"
    val_out = "./test_rl_data.jsonl"
    
    

    convert_and_save_dataset(train_json, train_img_dir, train_out, num=30000)
    convert_and_save_dataset(val_json, val_img_dir, val_out)
    
    
    # 图片路径（所有难度共用）
    # IMAGE_DIR = "/data/wengtengjin/colorsense/create_data/train_data/image"

    # # 输入目录（包含 easy.json / medium.json / hard.json 等）
    # INPUT_DIR = "rl_data"
    # OUTPUT_DIR = "./rl_ready"
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    # # 遍历并逐个转换
    # for filename in os.listdir(INPUT_DIR):
    #     if filename.endswith(".json"):
    #         json_path = os.path.join(INPUT_DIR, filename)
    #         out_path = os.path.join(OUTPUT_DIR, filename.replace(".json", ".jsonl"))
    #         convert_and_save_dataset(json_path, IMAGE_DIR, out_path)

    # print(f"📂 所有文件已输出到: {os.path.abspath(OUTPUT_DIR)}")
