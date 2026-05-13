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

    m = re.search(r'\\boxed\s*\{+([^}]*)\}+', response_text)
    
    if not m:
        return ""

    content = m.group(1).replace('{', '').replace('}', '').strip()

    if content == "":
        return []

    images = _IMAGE_PATTERN.findall(content)

    if not images:
        return ""

    return images


def build_prompt_different_angle(image_paths: list):
    # Generate image index description (image1 corresponds to the 1st image, etc.)
    image_count = len(image_paths)
    image_tokens = "\n".join([f"<image> image{i}" for i in range(1, image_count + 1)])
    
    prompt = (
        f"{image_tokens}\n\n"
        f"You are presented with {image_count} images, labeled image1, image2, ..., image{image_count}. "
        f"These labels correspond to the 1st, 2nd, ..., {image_count}-th images in the input sequence respectively.\n\n"

        "All images show the same type of object, possibly captured from different viewpoints, "
        "orientations, scales, or lighting conditions. These normal variations should not be considered anomalies.\n\n"

        "Your task is to identify ALL anomalous images in this set. "
        "Anomalous images are those that show visible surface defects or abnormal conditions, "
        "such as damage, cracks, scratches, stains, dirt, contamination, missing parts, "
        "deformation, discoloration, or other defects that cannot be explained by viewpoint or lighting changes.\n\n"

        "Strictly adhere to the following output rules:\n"
        "1. You may perform observation and comparative analysis before answering.\n"
        "2. The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block.\n"
        "   - Inside the box, list the labels of all anomalous images (e.g., image1, image2).\n"
        "   - Separate multiple labels using commas with no spaces.\n"
        "3. If no anomalous images are found, output \\boxed{{}}.\n\n"

        "Examples:\n"
        "- One anomaly: \\boxed{{image2}}\n"
        "- Multiple anomalies: \\boxed{{image2,image3}}\n"
        "- No anomalies: \\boxed{{}}\n"
    )

    return prompt

def build_prompt_same_angle_synthesis(image_paths: list):
    # Generate image index description (image1 corresponds to the 1st image, etc.)
    image_count = len(image_paths)
    image_tokens = "\n".join([f"<image> image{i}" for i in range(1, image_count + 1)])

    prompt = (
        f"{image_tokens}\n\n"
        f"You are presented with {image_count} images, labeled image1, image2, ..., image{image_count}. "
        f"These labels correspond to the 1st, 2nd, ..., {image_count}-th images in the input sequence respectively.\n\n"

        "The vast majority of these images follow a consistent visual pattern. "
        "A small number of images (one or more) may be anomalous, though it is also possible that all images are normal. "
        "Anomalous images deviate clearly from the majority in terms of appearance, structure, "
        "or other perceptible visual attributes. \n\n"

        "Your core task is to identify ALL anomalous images in this set.\n\n"

        "Strictly adhere to the following output rules:\n"
        "1. You may perform observation and comparative analysis before answering.\n"
        "2. The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block.\n"
        "   - Inside the box, list the labels of all anomalous images (e.g., image1, image2).\n"
        "   - Separate multiple labels using commas with no spaces.\n"
        "3. If no anomalous images are found, output \\boxed{{}}.\n\n"

        "Examples:\n"
        "- One anomaly: \\boxed{{image2}}\n"
        "- Multiple anomalies: \\boxed{{image2,image3}}\n"
        "- No anomalies: \\boxed{{}}\n"
    )
    return prompt
    
def build_prompt_same_angle_real(image_paths: list):
    # Generate image index description (image1 corresponds to the 1st image, etc.)
    image_count = len(image_paths)
    image_tokens = "\n".join([f"<image> image{i}" for i in range(1, image_count + 1)])

    prompt = (
        f"{image_tokens}\n\n"
        f"You are presented with {image_count} images, labeled image1, image2, ..., image{image_count}. "
        f"These labels correspond to the 1st, 2nd, ..., {image_count}-th images in the input sequence respectively.\n\n"

        "The vast majority of these images follow a consistent visual pattern. "
        "A small number of images (one or more) may be anomalous, though it is also possible that all images are normal. "
        "Anomalous images deviate clearly from the majority in terms of appearance, structure, "
        "or other perceptible visual attributes, such as damage, cracks, scratches, stains, dirt, contamination, or missing parts. \n\n"

        "Your core task is to identify ALL anomalous images in this set.\n\n"

        "Strictly adhere to the following output rules:\n"
        "1. You may perform observation and comparative analysis before answering.\n"
        "2. The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block.\n"
        "   - Inside the box, list the labels of all anomalous images (e.g., image1, image2).\n"
        "   - Separate multiple labels using commas with no spaces.\n"
        "3. If no anomalous images are found, output \\boxed{{}}.\n\n"

        "Examples:\n"
        "- One anomaly: \\boxed{{image2}}\n"
        "- Multiple anomalies: \\boxed{{image2,image3}}\n"
        "- No anomalies: \\boxed{{}}\n"
    )

    return prompt
