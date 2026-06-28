# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
import os
import json
import re
from mathruler.grader import extract_boxed_content, grade_answer
import re
import ast
import math


def extract_boxed_content_odd(text: str) -> str:
    """
    支持多层嵌套提取，例如 \boxed{{(1,2)}} -> (1,2)
    如果未提取到，返回 "None"
    """
    start_str = r"\boxed{"
    start_pos = text.rfind(start_str)
    if start_pos == -1:
        return "None"
    
    # 找到第一个左括号的位置开始截取
    brace_start = start_pos + len(start_str) - 1
    content = text[brace_start:]
    
    depth = 0
    end_pos = -1
    for i, char in enumerate(content):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            end_pos = i
            break
            
    if end_pos != -1:
        res = content[1:end_pos].strip()
        # 递归剥壳，处理 {{...}} 这种多层括号
        while res.startswith("{") and res.endswith("}"):
            res = res[1:-1].strip()
        return res
    return "None"

def parse_row_col(text: str):
    """
    从字符串结尾提取 Row 和 Column。
    要求：
      - 文本最后部分必须是 \boxed{Row X, Column Y}（可带句号、空格或换行）
      - 忽略前面任何解释性文字
    """
    pattern = re.compile(
        r"\\boxed\{\s*Row\s*(\d+)\s*,\s*Column\s*(\d+)\s*\}\s*[\.\s]*$",
        re.IGNORECASE
    )
    match = re.search(pattern, text.strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_gt(text: str):
    """
    从字符串中提取 Row 和 Column 数字。
    """
    pattern = re.compile(r"Row\s*(\d+)\s*,\s*Column\s*(\d+)")
    match = re.search(pattern, text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def accuracy_oddgrid_reward(gridsize: str, response: str, ground_truth: str, sigma_scale: float = 0.25) -> float:
    """
    计算基于高斯距离的离散准确率奖励。
    - 完全相等 → 1.0
    - 距离越远 → 按高斯衰减
    参数:
      gridsize: 如 "[3, 3]" 的字符串
      sigma_scale: 控制高斯分布宽度 (相对网格大小)
    """
    # 解析网格大小
    size = ast.literal_eval(gridsize)
    n_rows, n_cols = int(size[0]), int(size[1])


    # 提取坐标
    pred_row, pred_col = parse_row_col(response)
    gt_row, gt_col = parse_gt(ground_truth)
    
    if None in [pred_row, pred_col, gt_row, gt_col]:
        return 0.0,0,0
    if pred_row == gt_row and pred_col==gt_col:
        return 1.0, pred_row, pred_col
    # 曼哈顿或欧氏距离（这里用欧氏）
    distance = math.sqrt((pred_row - gt_row) ** 2 + (pred_col - gt_col) ** 2)

    # σ 随 grid 大小自适应，例如取 grid 对角线长度 × scale
    grid_diag = math.sqrt(n_rows ** 2 + n_cols ** 2)
    sigma = grid_diag * sigma_scale

    # 高斯衰减：e^(-(d^2)/(2σ^2))
    score = max(math.exp(- (distance ** 2) / (2 * sigma ** 2)) - 0.3, 0.0)  # 减去一个偏置，确保稍远的点奖励为0
    pred_row = 0 if pred_row is None else pred_row
    pred_col = 0 if pred_col is None else pred_col
    return float(score), pred_row, pred_col


def parse_to_set(raw_str):
    """
    Parse the boxed answer into a set of coordinates/items.
    Examples: "(2,2),(3,1)" -> {"(2,2)", "(3,1)"}; "" -> set().
    """
    if not raw_str or raw_str.strip() in ["", "None", "{}"]:
        return set()

    raw_str = raw_str.replace("，", ",")
    pattern = r"(\(\d+,\d+\)|image\d+|[^,\s]+)"
    items = re.findall(pattern, raw_str)
    return {item.strip() for item in items if item.strip()}


def format_reward(response: str) -> float:
    pattern = re.compile(r"\\boxed\{.*\}\s*$", re.DOTALL)
    return 1.0 if pattern.search(response.strip()) else 0.0


def parse_prediction_set(prediction_text):
    """
    Return (is_valid_final_answer, parsed_set). Invalid format is not the same
    as a valid empty answer: only \boxed{{}} at the end parses as an empty set.
    """
    if format_reward(prediction_text) == 0.0:
        return False, set()

    pred_content = extract_boxed_content_odd(prediction_text)
    return True, parse_to_set(pred_content)


def accuracy_reward(prediction_text, ground_truth_text):
    """
    Exact match over the predicted answer set.
    """
    valid_pred, pred_set = parse_prediction_set(prediction_text)
    if not valid_pred:
        return 0.0

    gt_set = parse_to_set(ground_truth_text)
    return 1.0 if pred_set == gt_set else 0.0


def f1_reward(prediction_text, ground_truth_text):
    """
    Set F1 over predicted and ground-truth coordinates.
    """
    valid_pred, pred_set = parse_prediction_set(prediction_text)
    if not valid_pred:
        return 0.0

    gt_set = parse_to_set(ground_truth_text)

    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0

    tp = len(pred_set.intersection(gt_set))
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_reward(prediction_text, ground_truth_text):
    """
    Reward matching the number of predicted anomalies.
    Invalid format receives 0; valid \boxed{{}} can match an empty GT.
    """
    valid_pred, pred_set = parse_prediction_set(prediction_text)
    if not valid_pred:
        return 0.0

    pred_count = len(pred_set)
    gt_count = len(parse_to_set(ground_truth_text))

    denom = max(pred_count, gt_count, 1)
    return max(1.0 - abs(pred_count - gt_count) / denom, 0.0)

def compute_score_EM(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    format_weight = 0.2
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

def compute_score_F1(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    format_weight = 0.2
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        f1_score = f1_reward(response, reward_input["ground_truth"])

        scores.append(
            {
                "overall": (1 - format_weight) * f1_score + format_weight * format_score,
                "format": format_score,
                "accuracy": f1_score,
            }
        )

    return scores



def compute_score_OURS(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.10,
    count_weight: float = 0.20,
    em_weight: float = 0.30,
    f1_weight: float = 0.40,
) -> list[dict[str, float]]:
    """
    Composite OddGrid reward for 200-step training.
    Components: format + count + exact match + set F1.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    total_weight = format_weight + count_weight + em_weight + f1_weight
    if total_weight <= 0:
        raise ValueError("Reward weights must sum to a positive value.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        ground_truth = reward_input["ground_truth"]

        format_score = format_reward(response)
        count_score = count_reward(response, ground_truth)
        em_score = accuracy_reward(response, ground_truth)
        f1_score = f1_reward(response, ground_truth)

        overall = (
            format_weight * format_score
            + count_weight * count_score
            + em_weight * em_score
            + f1_weight * f1_score
        ) / total_weight

        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "count": count_score,
                "em": em_score,
                "f1": f1_score,
                "accuracy": em_score,
            }
        )

    return scores

def compute_odd_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    format_weight = 0.2
    scores = []

    for reward_input in reward_inputs:
        gridsize = reward_input["ground_truth"].split("--")[0]

        reward_input["ground_truth"] = reward_input["ground_truth"].split("--")[1]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score, pred_row, pred_col = accuracy_oddgrid_reward(gridsize, response, reward_input["ground_truth"])

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores


# LOG_PATH = "./reward_debug_outputs.jsonl"  # 输出日志文件路径

# def compute_odd_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     scores = []
#     os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

#     with open(LOG_PATH, "a", encoding="utf-8") as f:  # 以追加模式写入
#         for reward_input in reward_inputs:
#             gridsize = reward_input["ground_truth"].split("--")[0]
    
#             reward_input["ground_truth"] = reward_input["ground_truth"].split("--")[1]
#             response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
#             format_score = format_reward(response)
#             accuracy_score, pred_row, pred_col = accuracy_oddgrid_reward(gridsize, response, reward_input["ground_truth"])

#             # 写入调试信息
#             log_item = {
#                 "response": response,
#                 "ground_truth": reward_input["ground_truth"],
#                 "format_score": format_score,
#                 "accuracy_score": accuracy_score,
#                 "pred_row": pred_row,
#                 "pred_col": pred_col,
#             }
#             f.write(json.dumps(log_item, ensure_ascii=False) + "\n")

#             scores.append(
#                 {
#                     "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
#                     "format": format_score,
#                     "accuracy": accuracy_score,
#                 }
#             )

#     return scores