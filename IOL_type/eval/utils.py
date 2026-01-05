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

    m = re.search(r'\\boxed\{([^}]*)\}', response_text)
    if not m:
        # ❌ 没有 boxed，格式错误
        return ""

    content = m.group(1).strip()

    # ✅ 合法：无异常
    if content == "":
        return []

    # 提取所有合法坐标
    coords = _COORD_PATTERN.findall(content)

    # 如果提取不到坐标，说明 boxed 里是垃圾 → 格式错误
    if not coords:
        return ""

    return coords

def build_prompt(data):
    grid_size = data.get("grid_size", [0, 0])

    # 兼容 grid_size = [rows, cols] 的情况
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        rows, cols = grid_size
        grid_str = f"{rows} × {cols}"
    else:
        # 兜底（防止脏数据）
        grid_str = str(grid_size)

    # prompt = (
    #     f"The image displays a {grid_str} grid of objects. "
    #     "The vast majority of objects adhere to a consistent pattern, while a small number of objects are anomalous—"
    #     "these anomalies deviate from the normal objects in one or more aspects: shape, color, size, or position.\n\n"

    #     "Your core task is to identify ALL anomalous objects in the grid.\n\n"
    #     "Strictly follow these output rules (non-compliance will be considered an error):\n"
    #     "1. For each anomalous object, specify its coordinate in the format (row, column). "
    #     "   - The top-left cell is defined as (1, 1). "
    #     "   - Rows increase in the downward direction, columns increase to the right.\n"
    #     "2. Compile ALL anomalous coordinates into a single list, with coordinates separated by commas (no spaces).\n"
    #     "3. Enclose the entire list of coordinates in exactly one \\boxed{ } tag—NO additional text, explanations, or formatting.\n"
    #     # "4. Output NOTHING except the \\boxed{ } block containing the coordinates.\n\n"

    #     "Example of valid output: \\boxed{(1,4),(2,1),(2,2),(1,5)}\n"
    #     "Examples of invalid output (ALL are unacceptable):\n"
    #     " - (1,4),(2,1) (missing \\boxed{ })\n"
    #     " - \\boxed{(1,4)} \\boxed{(2,1)} (multiple \\boxed{ } tags)\n"
    #     " - Anomalous objects: \\boxed{(1,4),(2,1)} (extra text)\n"
    #     " - \\boxed{(1, 4), (2, 1)}\n"
    # )
    prompt = (
    f"图像展示了一个 {grid_str} 的网格，每个单元格中包含一个物体。"
    "绝大多数物体遵循一致的视觉模式，只有少部分物体是异常的（一个或者多个），但也有可能所有物体均为正常。"
    "若存在异常物体，它们通常在视觉上偏离了大多数正常物体，"
    "例如在外观、结构或其他可感知属性上存在明显差异。\n\n"

    "你的核心任务是：找出网格中【所有】异常物体。\n\n"

    "请严格遵守以下输出规则：\n"
    "1. 对于每一个异常物体，给出其坐标，格式为 (行,列)。\n"
    "   - 左上角的单元格定义为 (1,1)。\n"
    "   - 行号从上到下递增，列号从左到右递增。\n"
    "2. 你可以在回答前进行必要的观察与分析，但【最终答案】必须且只能使用一个 \\boxed{ }，"
    "将所有异常物体的坐标列表完整包裹。\n"
    "3. 坐标之间使用英文逗号分隔，不要包含空格。\n"
    "4. 若不存在任何异常物体，请输出 \\boxed{}。\n\n"

    "存在一个异常物体的输出示例：……\\boxed{(2,2)}\n"
    "存在多个异常物体的输出示例：……\\boxed{(2,2),(3,1)，... ，}\n"
    "不存在异常物体的输出示例：……\\boxed{}\n"
)

    return prompt

