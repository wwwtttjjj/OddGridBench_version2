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

# def format_reward(response: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     format_match = re.fullmatch(pattern, response)
#     return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def format_reward(response: str) -> float:
    # 去掉空白
    response = response.strip()
    # 匹配：结尾是 \boxed{Row X, Column Y}（可带结尾句号）
    pattern = re.compile(r"\\boxed\{Row\s+\d+,\s*Column\s+\d+\}\.?$")

    format_match = re.search(pattern, response)
    return 1.0 if format_match else 0.0



def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    format_weight = 0.2
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        reward_input["ground_truth"] = reward_input["ground_truth"].split("--")[1]
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