import json
import os
import sys
# 允许从上级目录 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.utils import build_prompt_same_angle_real, build_prompt_different_angle, build_prompt_same_angle_synthesis
from pathlib import Path


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
        
        data_source = item.get("source", "")
        if data_source in ["icon", "minst","hanzi"]:
            prompt = build_prompt_same_angle_synthesis(item)
            image_abs = os.path.join(image_dir, os.path.basename(image))
            
        elif data_source in ["RAD", "MPDD", "GOODADS"]:
            prompt = build_prompt_different_angle(item)
            image_abs = image
        else:
            prompt = build_prompt_same_angle_real(item)
            image_abs = image
            

        # 转换为绝对路径
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
    train_out = "./train_icon_rl_data.jsonl"

        
    val_json = "../create_data/val_data.json"
    val_img_dir = "../../IOL_type/create_data/val_data/image"
    val_out = "./test_rl_data.jsonl"
    
    train_real_json = "../../Train_data/total_data_iol/iol_test_data.json"
    train_real_img_dir = "../../Train_data/total_data_iol/image"
    train_real_out = "./train_real_rl_data.jsonl"
    


    convert_and_save_dataset(train_json, train_img_dir, train_out)
    convert_and_save_dataset(train_real_json, train_real_img_dir, train_real_out)
    convert_and_save_dataset(val_json, val_img_dir, val_out)
    
