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

def build_prompt(data):
    grid_size = data.get("grid_size", [0, 0])

    # 兼容 grid_size = [rows, cols] 的情况
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        rows, cols = grid_size
        grid_str = f"{rows} × {cols}"
    else:
        # 兜底（防止脏数据）
        grid_str = str(grid_size)

    prompt = (
        f"The image displays a {grid_str} grid of objects. "
        "The vast majority of objects adhere to a consistent pattern, while a small number of objects are anomalous—"
        "these anomalies deviate from the normal objects in one or more aspects: shape, color, size, or position.\n\n"

        "Your core task is to identify ALL anomalous objects in the grid.\n\n"
        "Strictly follow these output rules (non-compliance will be considered an error):\n"
        "1. For each anomalous object, specify its coordinate in the format (row, column). "
        "   - The top-left cell is defined as (1, 1). "
        "   - Rows increase in the downward direction, columns increase to the right.\n"
        "2. Compile ALL anomalous coordinates into a single list, with coordinates separated by commas (no spaces).\n"
        "3. Enclose the entire list of coordinates in exactly one \\boxed{ } tag—NO additional text, explanations, or formatting.\n"
        # "4. Output NOTHING except the \\boxed{ } block containing the coordinates.\n\n"

        "Example of valid output: \\boxed{(1,4),(2,1),(2,2),(1,5)}\n"
        "Examples of invalid output (ALL are unacceptable):\n"
        " - (1,4),(2,1) (missing \\boxed{ })\n"
        " - \\boxed{(1,4)} \\boxed{(2,1)} (multiple \\boxed{ } tags)\n"
        " - Anomalous objects: \\boxed{(1,4),(2,1)} (extra text)\n"
        " - \\boxed{(1, 4), (2, 1)}\n"
    )

    return prompt

