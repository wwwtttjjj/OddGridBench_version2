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

# def build_prompt(image_paths: list):
#     # 生成图片编号说明（image1对应第1张，依此类推）
#     image_count = len(image_paths)
#     image_desc = (
#         f"You are given {image_count} images, numbered as image1, image2, ..., image{image_count} "
#         "(corresponding to the 1st, 2nd, ..., {image_count}-th image in the input list respectively).\n\n"
#     )

def build_prompt(image_paths: list):
    # Generate image index description (image1 corresponds to the 1st image, etc.)
    image_count = len(image_paths)
    image_tokens = "\n".join([f"<image> image{i}" for i in range(1, image_count + 1)])
    image_desc = (
        f"{image_tokens}\n\nYou will be given {image_count} images, labeled as image1, image2, …, image{image_count} "
        f"(corresponding to the 1st, 2nd, …, and {image_count}th images in the input list, respectively).\n\n"
    )

    prompt = (
        f"{image_desc}"
        "Your main task is to identify the images that are different from the majority. "
        "An image should be considered anomalous if it deviates from the others in terms of content."
        "You may first carefully observe and describe the images to support your results. "
        "However, your final answer must strictly follow the output rules below. "
        "1. The indices of anomalous images must be written in the format \"imageX\", where X is the image index.\n"
        "2. The final answer must be enclosed in exactly one \\boxed{ } block.\n"
        "3. Only the anomalous image indices should appear inside the \\boxed{ } block.\n"
        "4. Any explanation or observation must appear outside the \\boxed{ } block.\n\n"
        "Example output for final answer:\n"
        "\\boxed{image2, image5}\n"
    )

    return prompt
