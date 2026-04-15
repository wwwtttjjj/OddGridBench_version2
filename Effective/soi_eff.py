import os
import argparse
import json
import base64
import requests
import random
import time
from tqdm import tqdm

from configs_soi import get_configs, max_new_tokens
from utils_soi import *
from PIL import Image

API_URL = "http://localhost:8081/v1/chat/completions"


def call_vllm_server(prompt, image_paths, model_path):
    messages = [{"role": "user", "content": []}]

    # ===== ❌ 去掉 resize，直接原图 =====
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    messages[0]["content"].append({
        "type": "text",
        "text": prompt.strip()
    })

    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }

    resp = None
    try:
        resp = requests.post(API_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        msg = resp.text[:300] if resp is not None else str(e)
        print(f"[ERROR] vLLM request failed: {msg}")
        return None


def run_vllm_http(args):
    configs_para = get_configs(args)

    json_path = configs_para["json_path"]
    model_path = configs_para["model_path"]
    save_json_path = configs_para["save_path"]

    if not json_path or not model_path:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    # ===== 全局计时 =====
    global_start = time.time()

    # ===== 断点续传 =====
    processed_ids = set()
    all_results = []

    if os.path.exists(save_json_path):
        with open(save_json_path, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
                processed_ids = {item["id"] for item in all_results}
                print(f"[INFO] Resume: {len(processed_ids)} processed")
            except:
                print("[WARNING] Failed to load existing results")

    # ===== 读取数据 =====
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    print(f"[INFO] Total samples: {len(json_data)}")

    # ===== 过滤未处理 =====
    valid_data = [d for d in json_data if d.get("id") not in processed_ids]
    print(f"[INFO] Remaining: {len(valid_data)}")

    # ===== 随机采样 =====
    random.seed(42)
    if len(valid_data) > args.sample_num:
        valid_data = random.sample(valid_data, args.sample_num)

    print(f"[INFO] Sampled: {len(valid_data)}")

    # ===== 推理 =====
    time_list = []

    for data in tqdm(valid_data):
        id = data.get("id")

        image_names = [
            os.path.join(data.get("image"), f"{i}.png")
            for i in range(1, data.get("total_icons") + 1)
        ]
        image_paths = [
            os.path.join(configs_para["image_dir"], img_name)
            for img_name in image_names
        ]

        prompt = build_prompt(image_paths)

        # ===== 单条计时 =====
        start_time = time.time()

        predict_answer = call_vllm_server(prompt, image_paths, model_path)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if predict_answer is None:
            continue

        extract_answer = extract_answer_from_response(predict_answer)

        save_item = {
            "id": id,
            "image": data.get("image"),
            "image_num": data.get("total_icons"),
            "prompt": prompt,
            "predict_answer": predict_answer,
            "extract_answer": extract_answer,
            "answer": data.get("odd_indices", []),
            "odd_count": data.get("num_odds"),

            # 🔥 新增
            "inference_time": elapsed_time
        }

        all_results.append(save_item)
        time_list.append(elapsed_time)

        # ===== 实时保存 =====
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        processed_ids.add(id)

    # ===== 性能统计 =====
    total_time = time.time() - global_start

    if len(time_list) > 0:
        time_list_sorted = sorted(time_list)
        avg_time = sum(time_list) / len(time_list)
        p50 = time_list_sorted[len(time_list)//2]
        p95 = time_list_sorted[int(len(time_list)*0.95)]

        print("\n===== 性能统计 =====")
        print(f"样本数: {len(time_list)}")
        print(f"平均耗时: {avg_time:.3f}s")
        print(f"P50: {p50:.3f}s")
        print(f"P95: {p95:.3f}s")
        print(f"总耗时: {total_time:.2f}s")

    print(f"\n[SUCCESS] Saved to {save_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multimodal inference via vLLM HTTP API")

    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--image_type", type=str, default="normal")
    parser.add_argument("--data_type", type=str, default="MVTEC")

    # 🔥 新增
    parser.add_argument("--sample_num", type=int, default=100)

    args = parser.parse_args()

    run_vllm_http(args)


if __name__ == "__main__":
    main()