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
import sys
# 允许从上级目录 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Ablation.configs import extract_answer

# ================= 基础配置 =================
max_new_tokens = 2048
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models")
BASE_DATA_DIR = "../Ablation/single_data"
SAVE_DIR_BASE = "./single_results"  # 结果保存根目录
EXAMPLE_DIR = "../Ablation/examples"         # 示例图片根目录
API_URL = "http://localhost:8081/v1/chat/completions"
# ================= 工具函数 =================

def build_multimodal_prompt(mode, img_path, current_img_base64):
    """构建多模态对话消息列表"""
    messages = []
    
    # 1. 正常参考
    if mode in ["one-example", "two-examples"]:
        pos_path = find_example_image(img_path, target_type="Normal")
        pos_b64 = encode_image(pos_path)
        if pos_b64:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pos_b64}"}},
                {"type": "text", "text": "This is a [Standard Normal Sample] without any defects. It serves as your quality baseline."}
            ]})
            messages.append({"role": "assistant", "content": "Understood. I have analyzed the normal sample and will use it as a reference."})

    # 2. 异常参考
    if mode == "two-examples":
        neg_path = find_example_image(img_path, target_type="Anomaly")
        neg_b64 = encode_image(neg_path)
        if neg_b64:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{neg_b64}"}},
                {"type": "text", "text": "This is an [Anomalous Sample] that contains defects. Please note these irregular features."}
            ]})
            messages.append({"role": "assistant", "content": "Understood. I have identified the defective features for comparison."})

    output_relu = f"""Strictly adhere to the following output rules\n
    1.You may perform observation and comparative analysis before answering.
    2. The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block.\n"""
    # 3. 最终指令
    if mode == "zero-shot":
        instruction = f"""{output_relu}
        3. Determine if there are any defects, and finally output the answer as boxed{{Yes}} or boxed{{No}}."""
    elif mode == "one-example":
        instruction = f"""
        {output_relu}
        3.Compared to the [Standard Normal Sample] provided earlier, does this image show any deviations or defects? 
        4. Based on your analysis, output the final answer as boxed{{Yes}} or boxed{{No}}."""
    else: # two-examples
        instruction = f"""
        {output_relu} 
        3. By comparing it with both the [Standard Normal Sample] and the [Anomalous Sample] above, determine if this image is defective. 
        4. Based on your comparison, output the final answer as boxed{{Yes}} or boxed{{No}}."""

    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_base64}"}},
            {"type": "text", "text": instruction}
        ]
    })
    return messages

def encode_image(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def find_example_image(original_path, target_type="Normal"):
    """将推理路径映射到 example 路径"""
    if not original_path: return None
    # 统一路径分隔符并分割
    parts = original_path.replace("\\", "/").split('/')
    
    new_parts = []
    # 从路径中寻找类别标识并替换
    found_type = False
    for p in parts:
        if p.lower() in ["normal", "anomaly", "abnormal"]:
            new_parts.append(target_type)
            found_type = True
        else:
            new_parts.append(p)
    
    if not found_type: return None

    # 构建目标目录 (去掉文件名)
    sub_dir = "/".join(new_parts[:-1])
    # 移除路径中可能存在的相对路径前缀以匹配 example 结构
    sub_dir = sub_dir.replace("../Ablation/single_data/", "")
    
    target_dir = os.path.join(EXAMPLE_DIR, sub_dir)
    
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
    for ext in extensions:
        potential_path = os.path.join(target_dir, f"example{ext}")
        if os.path.exists(potential_path):
            return potential_path
    return None


def call_vllm_api(messages, model_name):
    """通过 API 发送消息列表"""
    full_model_path = os.path.join(MODEL_PATH, model_name)
    payload = {
        "model": full_model_path,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"\n[ERROR] API 请求失败: {e}")
        return None

# ================= 主推理函数 =================

def run_inference(data_type, dataset_name, model_name, sample_num, mode):
    global_start = time.time()

    # 根据 mode 区分保存路径
    current_save_dir = f"{SAVE_DIR_BASE}_{mode}"
    if not os.path.exists(current_save_dir):
        os.makedirs(current_save_dir)

    json_input_path = os.path.join(BASE_DATA_DIR, f"{dataset_name}_{data_type}.json")
    safe_model_name = model_name.replace('/', '_')
    save_path = os.path.join(current_save_dir, f"{data_type}_{dataset_name}_{safe_model_name}.json")

    if not os.path.exists(json_input_path):
        print(f"[ERROR] 找不到索引文件: {json_input_path}")
        return

    with open(json_input_path, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    all_results = []
    processed_paths = set()

    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
                processed_paths = {item["path"] for item in all_results}
                print(f"[INFO] 载入进度，跳过已处理 {len(processed_paths)} 条")
        except: pass

    # 过滤与路径修复
    valid_items = []
    for k, v in image_metadata.items():
        # 保持你原始代码中的路径拼接逻辑
        img_path = "../Ablation/" + v.get("physical_path")
        label = v.get("label", "").lower()

        if not os.path.exists(img_path) or label not in ["anomaly", "normal"] or img_path in processed_paths:
            continue
        valid_items.append((k, v))

    random.seed(42)
    if len(valid_items) > sample_num:
        valid_items = random.sample(valid_items, sample_num)

    print(f"\n[START] Mode: {mode} | {dataset_name} | {model_name} | Samples: {len(valid_items)}")

    time_list = []
    for meta_key, info in tqdm(valid_items):
        img_path = "../Ablation/" + info["physical_path"]
        print(f"\n[INFO] Processing: {img_path}")
        # 构建当前图片的 Base64
        current_b64 = encode_image(img_path)
        if not current_b64: continue

        # 根据模式构建多图消息体
        messages = build_multimodal_prompt(mode, img_path, current_b64)

        start_time = time.time()
        predict_answer = call_vllm_api(messages, model_name)
        elapsed_time = time.time() - start_time

        if predict_answer is None: continue

        extract_ans = extract_answer(predict_answer)

        res_item = {
            "filename": info.get("filename"),
            "path": img_path,
            "gt_label": info.get("label"),
            "predict": predict_answer,
            "extract_answer": extract_ans,
            "gt": "no" if info.get("label") == "normal" else "yes",
            "mode": mode,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "inference_time": elapsed_time,
            "resize_scale": info.get("resize_scale"),
            "original_count": info.get("count")
        }

        all_results.append(res_item)
        time_list.append(elapsed_time)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    # 统计逻辑保持不变...
    total_time = time.time() - global_start
    if time_list:
        print(f"\n[DONE] Avg: {sum(time_list)/len(time_list):.3f}s | Total: {total_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="iol")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--sample_num", type=int, default=100)
    # 新增模式参数
    parser.add_argument("--mode", type=str, default="one-example", 
                        choices=["zero-shot", "one-example", "two-examples"])

    args = parser.parse_args()

    run_inference(
        args.type,
        args.dataset,
        args.model_name,
        args.sample_num,
        args.mode
    )