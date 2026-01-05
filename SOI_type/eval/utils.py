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

_IMAGE_PATTERN = re.compile(r'image\d+')

def extract_answer_from_response(response_text):
    if response_text is None:
        return ""

    m = re.search(r'\\boxed\{([^}]*)\}', response_text)
    if not m:
        # ❌ 没有 boxed，格式错误
        return ""

    content = m.group(1).strip()

    # ✅ 合法：无异常图像
    if content == "":
        return []

    # 提取所有合法 imageX
    images = _IMAGE_PATTERN.findall(content)

    # boxed 里有内容，但没有合法 imageX → 格式错误
    if not images:
        return ""

    return images


def build_prompt(image_paths: list):
    # Generate image index description (image1 corresponds to the 1st image, etc.)
    image_count = len(image_paths)
    image_tokens = "\n".join([f"<image> image{i}" for i in range(1, image_count + 1)])

    # prompt = (
    #     f"{image_desc}"
    #     "Your main task is to identify the images that are different from the majority. "
    #     "An image should be considered anomalous if it deviates from the others in terms of content."
    #     "You may first carefully observe and describe the images to support your results. "
    #     "However, your final answer must strictly follow the output rules below. "
    #     "1. The indices of anomalous images must be written in the format \"imageX\", where X is the image index.\n"
    #     "2. The final answer must be enclosed in exactly one \\boxed{ } block.\n"
    #     "3. Only the anomalous image indices should appear inside the \\boxed{ } block.\n"
    #     "4. Any explanation or observation must appear outside the \\boxed{ } block.\n\n"
    #     "Example output for final answer:\n"
    #     "\\boxed{image2, image5}\n"
    # )
    prompt = (
        f"{image_tokens}\n\n"
        f"你将看到 {image_count} 张图像，分别标记为 image1、image2、……、image{image_count}，"
        f"它们分别对应输入列表中的第 1 张、第 2 张、……、第 {image_count} 张图像。\n\n"

        "绝大多数图像中的物体遵循一致的视觉模式，只有少部分图像是异常的（一个或多个），"
        "但也有可能所有图像均为正常。"
        "若存在异常图像，它们通常在视觉上偏离了大多数正常图像，"
        "例如在外观、结构或其他可感知属性上存在明显差异。\n\n"

        "你的核心任务是：找出这些图像中【所有】异常图像。\n\n"

        "你可以在回答前进行必要的观察与分析，但【最终答案】必须且只能使用一个 \\boxed{ }。"
        "若不存在任何异常图像，请输出 \\boxed{}。\n\n"

        "存在一个异常图像的输出示例：……\\boxed{image2}\n"
        "存在多个异常图像的输出示例：……\\boxed{image2,image3}\n"
        "不存在异常图像的输出示例：……\\boxed{}\n"
    )

    return prompt
