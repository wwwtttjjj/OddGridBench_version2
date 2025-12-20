import json
import os
import re

def write_json(save_json_path, save_json_data):
    if os.path.exists(save_json_path):
        with open(save_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(save_json_data)
        else:
            data = [data, save_json_data]
    else:
        data = [save_json_data]

    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def extract_answer_from_response(response_text):
    if response_text is None:
        return ""
    m = re.search(r'\\boxed\{([^}]*)\}', response_text)
    return m.group(1) if m else ""

def build_prompt(image_paths: list):
    # 生成图片编号说明（image1对应第1张，依此类推）
    image_count = len(image_paths)
    image_desc = (
        f"You are given {image_count} images, numbered as image1, image2, ..., image{image_count} "
        "(corresponding to the 1st, 2nd, ..., {image_count}-th image in the input list respectively).\n\n"
    )

def build_prompt(image_paths: list):
    # 生成图片编号说明（image1对应第1张，依此类推）
    image_count = len(image_paths)
    image_desc = (
        f"你将收到 {image_count} 张图片，编号为 image1、image2、……、image{image_count} "
        "（分别对应输入列表中第1张、第2张、……、第{image_count}张图片）。\n\n"
    )

    prompt = (
        f"{image_desc}"
        "你的核心任务是识别出其中与大多数图片不一样的内容（即与其他图片在内容/特征/模式上存在偏离，颜色，形状，位置等，最多三张）。\n\n"
        "严格遵守以下输出规则（不符合将被判定为错误）：\n"
        "1. 异常图片的编号必须使用「imageX」格式（X为图片的数字序号，例如第3张图片写为image3）。\n"
        "2. 将你觉得的最终答案的异常图片编号包裹在且仅包裹在一个 \\boxed{ } 标签内\n"
    )

    return prompt

    return prompt

