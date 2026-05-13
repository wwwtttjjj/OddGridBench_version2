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
        
_COORD_PATTERN = re.compile(r'\(\d+,\d+\)')

def extract_answer_from_response(response_text):
    if response_text is None:
        return ""

    m = re.search(r'\\boxed\s*\{+([^}]*)\}+', response_text)
    
    if not m:
        return ""

    # 清洗提取到的内容，把多出来的 { 或 } 彻底去掉
    content = m.group(1).replace('{', '').replace('}', '').strip()

    # ✅ 合法：无异常（现在 content 为空字符串了）
    if content == "":
        return []

    # 提取所有合法坐标
    # 假设 _COORD_PATTERN 定义正确
    coords = _COORD_PATTERN.findall(content)

    if not coords:
        return ""

    return coords
def build_prompt_different_angle(data):
    grid_size = data.get("grid_size", [0, 0])

    # 兼容 grid_size = [rows, cols] 的情况
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        rows, cols = grid_size
        grid_str = f"{rows} × {cols}"
    else:
        # 兜底（防止脏数据）
        grid_str = str(grid_size)

    prompt = (
        f"The image displays a {grid_str} (there may be some missing at the end). Each cell shows the same type of object, "
        "possibly from different viewpoints, orientations, scales, or lighting conditions. "
        "These normal variations should not be considered anomalies.\n\n"

        "Your task is to identify ALL anomalous objects in the grid. "
        "Anomalous objects are those with visible surface defects or abnormal conditions, "
        "such as damage, cracks, scratches, stains, dirt, contamination, missing parts, "
        "deformation, discoloration, or other defects that cannot be explained by viewpoint or lighting changes.\n\n"

        "Strictly follow these output rules:\n"
        "1. For each anomalous object, provide its coordinates in the format (row,column).\n"
        "   - The top-left cell is defined as (1,1).\n"
        "   - Row indices increase from top to bottom; column indices increase from left to right.\n"
        "2. You may perform observation and analysis before answering. "
        "The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block, "
        "including the complete list of anomalous object coordinates.\n"
        "3. If no anomalous objects are found, output \\boxed{{}}.\n\n"

        "Examples:\n"
        "- One anomaly: \\boxed{{(2,2)}}\n"
        "- Multiple anomalies: \\boxed{{(2,2),(3,1)}}\n"
        "- No anomalies: \\boxed{{}}\n"
    )
    return prompt


def build_prompt_same_angle_synthesis(data):
    grid_size = data.get("grid_size", [0, 0])

    # 兼容 grid_size = [rows, cols] 的情况
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        rows, cols = grid_size
        grid_str = f"{rows} × {cols}"
    else:
        # 兜底（防止脏数据）
        grid_str = str(grid_size)

    prompt = (
        f"The image displays a {grid_str} grid, where each cell contains an object. "
        "Most objects follow a consistent visual pattern; a small number may be anomalous, "
        "and it is also possible that all objects are normal. "
        "Anomalous objects deviate clearly from the majority in appearance, structure, "
        "or other perceptible visual attributes. \n\n"

        "Your core task is to identify ALL anomalous objects in the grid.\n\n"

        "Strictly adhere to the following output rules:\n"
        "1. For each anomalous object, provide its coordinates in the format (row,column).\n"
        "   - The top-left cell is defined as (1,1).\n"
        "   - Row indices increase from top to bottom; column indices increase from left to right.\n"
        "2. You may perform observation and analysis before answering. "
        "The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block, "
        "which includes the complete list of coordinates for all anomalous objects.\n"
        "3. If no anomalous objects are found, output \\boxed{{}}.\n\n"

        "Examples:\n"
        "- One anomaly: \\boxed{{(2,2)}}\n"
        "- Multiple anomalies: \\boxed{{(2,2),(3,1)}}\n"
        "- No anomalies: \\boxed{{}}\n"
    )
    return prompt


def build_prompt_same_angle_real(data):
    grid_size = data.get("grid_size", [0, 0])

    # 兼容 grid_size = [rows, cols] 的情况
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        rows, cols = grid_size
        grid_str = f"{rows} × {cols}"
    else:
        # 兜底（防止脏数据）
        grid_str = str(grid_size)

    prompt = (
        f"The image displays a {grid_str} grid (there may be some missing at the end), where each cell contains an object. "
        "Most objects follow a consistent visual pattern; a small number may be anomalous, "
        "and it is also possible that all objects are normal. "
        "Anomalous objects deviate clearly from the majority in appearance, structure, "
        "or other perceptible visual attributes, such as damage, cracks, scratches, stains, dirt, contamination, or missing parts. \n\n"

        "Your core task is to identify ALL anomalous objects in the grid.\n\n"

        "Strictly adhere to the following output rules:\n"
        "1. For each anomalous object, provide its coordinates in the format (row,column).\n"
        "   - The top-left cell is defined as (1,1).\n"
        "   - Row indices increase from top to bottom; column indices increase from left to right.\n"
        "2. You may perform observation and analysis before answering. "
        "The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block, "
        "which includes the complete list of coordinates for all anomalous objects.\n"
        "3. If no anomalous objects are found, output \\boxed{{}}.\n\n"

        "Examples:\n"
        "- One anomaly: \\boxed{{(2,2)}}\n"
        "- Multiple anomalies: \\boxed{{(2,2),(3,1)}}\n"
        "- No anomalies: \\boxed{{}}\n"
    )
    return prompt
