import os
import json
import base64
import requests
import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
import re

import os
# models_dir = "../models/"
max_new_tokens = 2048

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models")

# MODEL_PATH = r"/nfsdata4/wengtengjin/oddgrid_task/models/Qwen3-VL-8B-Instruct"
BASE_DATA_DIR = "../Ablation/single_data"
SAVE_DIR = "./single_results"  # 保存目录

# ================= 配置区 =================
API_URL = "http://localhost:8081/v1/chat/completions"
# =========================================

def extract_answer(predict_answer):
    match = re.search(r'box\{(Yes|No)\}', predict_answer, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def build_custom_prompt():
    return f"Please analyze the image first, then determine if there are any defects and output the final answer as box{{Yes}} or box{{No}}."

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vllm_api(prompt, img_path, model_name):
    image_base64 = encode_image(img_path)
    full_model_path = os.path.join(MODEL_PATH, model_name)

    payload = {
        "model": full_model_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt.strip()
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"\n[ERROR] API 请求失败 {os.path.basename(img_path)}: {e}")
        return None


def run_inference(data_type, dataset_name, model_name, sample_num):
    # ===== 全局计时 =====
    global_start = time.time()

    # ===== 路径 =====
    json_input_path = os.path.join(BASE_DATA_DIR, f"{dataset_name}_{data_type}.json")
    print(json_input_path)
    safe_model_name = model_name.replace('/', '_')
    save_path = f"{SAVE_DIR}/{data_type}_{dataset_name}_{safe_model_name}.json"

    if not os.path.exists(json_input_path):
        print(f"[ERROR] 找不到索引文件: {json_input_path}")
        return

    with open(json_input_path, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    # ===== 断点续传 =====
    all_results = []
    processed_paths = set()

    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
                processed_paths = {item["path"] for item in all_results}
                print(f"[INFO] 载入进度，跳过已处理 {len(processed_paths)} 条")
        except Exception as e:
            print(f"[WARNING] 读取输出 JSON 失败: {e}")

    # ===== 过滤有效数据 =====
    valid_items = []
    for k, v in image_metadata.items():
        img_path = "../Ablation/" + v.get("physical_path")
        label = v.get("label", "").lower()

        if not img_path or not os.path.exists(img_path):
            continue
        if label not in ["anomaly", "normal"]:
            continue
        if img_path in processed_paths:
            continue

        valid_items.append((k, v))

    print(f"[INFO] 可用样本数: {len(valid_items)}")

    # ===== 随机采样 =====
    random.seed(42)
    if len(valid_items) > sample_num:
        valid_items = random.sample(valid_items, sample_num)

    print(f"[INFO] 实际采样数: {len(valid_items)}")

    # ===== 推理 =====
    print(f"\n[START] {dataset_name} | {data_type} | {model_name}")

    time_list = []

    for meta_key, info in tqdm(valid_items):
        img_path = "../Ablation/" + info["physical_path"]

        prompt = build_custom_prompt()

        # ===== 单条计时 =====
        start_time = time.time()

        predict_answer = call_vllm_api(prompt, img_path, model_name)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if predict_answer is None:
            continue

        extract_ans = extract_answer(predict_answer)

        # ===== 记录结果 =====
        res_item = {
            "filename": info.get("filename"),
            "path": img_path,
            "gt_label": info.get("label"),
            "predict": predict_answer,
            "extract_answer": extract_ans,
            "gt": "no" if info.get("label") == "normal" else "yes",
            "data_type": data_type,
            "dataset_name": dataset_name,
            "model_name": model_name,

            # 🔥 核心新增
            "inference_time": elapsed_time,

            # 原字段
            "resize_scale": info.get("resize_scale"),
            "original_count": info.get("count")
        }

        all_results.append(res_item)
        time_list.append(elapsed_time)

        # ===== 实时保存 =====
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        processed_paths.add(img_path)

    # ===== 全局统计 =====
    total_time = time.time() - global_start

    if len(time_list) > 0:
        avg_time = sum(time_list) / len(time_list)
        p50 = sorted(time_list)[len(time_list)//2]
        p95 = sorted(time_list)[int(len(time_list)*0.95)]

        print("\n===== 性能统计 =====")
        print(f"样本数: {len(time_list)}")
        print(f"平均耗时: {avg_time:.3f}s")
        print(f"P50: {p50:.3f}s")
        print(f"P95: {p95:.3f}s")
        print(f"总耗时: {total_time:.2f}s")

    print(f"\n[SUCCESS] 保存路径: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="iol")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")

    # 🔥 新增参数
    parser.add_argument("--sample_num", type=int, default=100)

    args = parser.parse_args()

    run_inference(
        args.type,
        args.dataset,
        args.model_name,
        args.sample_num
    )