import os
import json
import base64
import requests
import argparse
from tqdm import tqdm
from configs import BASE_DATA_DIR, SAVE_DIR,MODEL_PATH, max_new_tokens

# ================= 配置区 =================
# vllm serve 启动的 API 地址
API_URL = "http://localhost:8081/v1/chat/completions"
# =========================================

def build_custom_prompt(image_path, label):
    """保持原有的 Prompt 逻辑"""
    return f"This image is labeled as {label}. Please analyze it."

def encode_image(image_path):
    """将图像转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_vllm_api(prompt, img_path, model_name):
    """通过 HTTP API 调用部署好的 vLLM 服务"""
    image_base64 = encode_image(img_path)
    
    payload = {
        "model": os.path.join(MODEL_PATH, model_name),  # 必须与 vllm serve 启动时的 --model 或 --served-model-name 一致
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
        print(f"\n[ERROR] API request failed for {os.path.basename(img_path)}: {e}")
        return None

def run_inference(data_type, dataset_name, model_name):
    # 构建路径与保存文件名
    target_dir = os.path.join(BASE_DATA_DIR, data_type, dataset_name)
    save_path = f"{SAVE_DIR}/{data_type}_{dataset_name}_{model_name.replace('/', '_')}.json"
    
    if not os.path.exists(target_dir):
        print(f"路径不存在: {target_dir}")
        return

    # --- 1. 处理断点续传：加载已有结果 ---
    all_results = []
    processed_paths = set()
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
                processed_paths = {item["path"] for item in all_results}
                print(f"[INFO] 检测到已有文件，已跳过 {len(processed_paths)} 条记录")
        except Exception as e:
            print(f"[WARNING] 读取已有 JSON 失败: {e}")
            all_results = []

    supported_extensions = ('.bmp', '.JPG', '.png', '.jpg', '.jpeg')

    # 遍历目录结构
    for root, dirs, files in os.walk(target_dir):
        label = os.path.basename(root)
        if label not in ["Anomaly", "Normal"]:
            continue

        print(f"\n正在扫描路径: {root} (Label: {label})")
        
        for file in tqdm(files):
            if not file.lower().endswith(supported_extensions):
                continue
            
            img_path = os.path.join(root, file)

            # --- 2. 判断是否已处理 ---
            if img_path in processed_paths:
                continue
            
            prompt = build_custom_prompt(img_path, label)
            
            # --- 3. 调用 API 进行推理 ---
            predict_answer = call_vllm_api(prompt, img_path, model_name)
            
            if predict_answer is None:
                continue

            # 构造保存项（保持原有格式）
            res_item = {
                "filename": file,
                "path": img_path,
                "gt_label": label,
                "predict": predict_answer,
                "data_type": data_type,
                "dataset_name": dataset_name,
                "model_name": model_name,
            }
            
            # --- 4. 实时写入结果 ---
            all_results.append(res_item)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            processed_paths.add(img_path)
    
    print(f"\n[SUCCESS] 任务完成！结果已更新至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="iol", help="iol or soi")
    parser.add_argument("--dataset", type=str, default="mvtec", help="e.g., mvtec, VisA")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    run_inference(args.type, args.dataset, args.model_name)