import os
import json
import base64
import requests
import argparse
from tqdm import tqdm
from pathlib import Path
from configs import BASE_DATA_DIR, SAVE_DIR, MODEL_PATH, max_new_tokens, build_multimodal_prompt, extract_answer

# ================= 配置区 =================
API_URL = "http://localhost:8081/v1/chat/completions"
# =========================================

def encode_image(image_path):
    """将图像转换为 base64"""
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_vllm_api(messages, model_name):
    """通过 HTTP API 调用部署好的 vLLM 服务"""
    # 这里的 model 字段需要与你启动 vllm serve 时指定的名称一致
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

def run_inference(data_type, dataset_name, model_name, mode):
    # --- 1. 构建输入与输出路径 ---
    json_input_path = os.path.join(BASE_DATA_DIR, f"{dataset_name}_{data_type}.json")
    
    # 结果文件名包含模式，并确保存储目录存在
    safe_model_name = model_name.replace('/', '_')
    current_save_dir = f"{SAVE_DIR}_{mode}"
    if not os.path.exists(current_save_dir):
        os.makedirs(current_save_dir)
        
    save_path = os.path.join(current_save_dir, f"{data_type}_{dataset_name}_{safe_model_name}.json")
    
    if not os.path.exists(json_input_path):
        print(f"[ERROR] 找不到索引文件: {json_input_path}")
        return

    with open(json_input_path, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    # --- 2. 处理断点续传 ---
    all_results = []
    processed_paths = set()
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
                processed_paths = {item["path"] for item in all_results}
                print(f"[INFO] 载入进度，跳过已处理的 {len(processed_paths)} 条记录")
        except Exception as e:
            print(f"[WARNING] 读取输出 JSON 失败: {e}")

    # --- 3. 遍历索引进行 API 推理 ---
    print(f"\n[START] 模式: {mode} | 任务: {dataset_name} | 类型: {data_type}")

    for meta_key, info in tqdm(image_metadata.items()):
        img_path = info.get("physical_path")
        if not img_path or not os.path.exists(img_path) or img_path in processed_paths:
            continue
            
        label = info.get("label", "").lower()
        if label not in ["anomaly", "normal"]:
            continue

        try:
            # 编码当前待测图片
            current_b64 = encode_image(img_path)
            
            # 使用与本地脚本相同的 build_multimodal_prompt 构建消息体
            # 注意：build_multimodal_prompt 内部会调用 find_example_image
            messages = build_multimodal_prompt(mode, img_path, current_b64)
            
            # 调用 API (传入完整的 messages 列表)
            predict_answer = call_vllm_api(messages, model_name)
            
            if predict_answer is None:
                continue
                
            extract_ans = extract_answer(predict_answer)
            
            # 构造结果项
            res_item = {
                "filename": info.get("filename"),
                "path": img_path,
                "gt_label": info.get("label"),
                "predict": predict_answer,
                "mode": mode,
                "data_type": data_type,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "extract_answer": extract_ans,
                "gt": "no" if info.get("label") == "normal" else "yes",
                "resize_scale": info.get("resize_scale"),
                "original_count": info.get("count")
            }
            
            # --- 4. 实时保存 ---
            all_results.append(res_item)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            processed_paths.add(img_path)
            
        except Exception as e:
            print(f"\n[ERROR] 处理失败 {img_path}: {e}")
            continue
    
    print(f"\n[SUCCESS] 处理完成，结果保存在: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="soi", help="iol or soi")
    parser.add_argument("--dataset", type=str, default="Nanfang", help="e.g., mvtec, VisA")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-4B-Instruct")
    # 增加 mode 参数
    parser.add_argument("--mode", type=str, default="two-examples", 
                        choices=["zero-shot", "one-example", "two-examples"])
    args = parser.parse_args()

    run_inference(args.type, args.dataset, args.model_name, args.mode)