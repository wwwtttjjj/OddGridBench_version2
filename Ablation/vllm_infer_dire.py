import os
import json
import base64
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from configs import MODEL_PATH, BASE_DATA_DIR,SAVE_DIR, max_new_tokens

# =========================================

_VLLM_MODEL = None

def get_vllm_model(model_name):
    global _VLLM_MODEL
    if _VLLM_MODEL is None:
        tp = torch.cuda.device_count()
        _VLLM_MODEL = LLM(
            model=os.path.join(MODEL_PATH, model_name),
            max_model_len=4096,
            trust_remote_code=True,
            tensor_parallel_size=tp,
        )
    return _VLLM_MODEL

def build_custom_prompt(image_path, label):
    """
    在这里编写你的 Prompt 逻辑
    """
    return f"This image is labeled as {label}. Please analyze it."

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_inference(data_type, dataset_name, model_name):
    llm = get_vllm_model(model_name)
    
    # 构建路径: single_data/{iol|soi}/{dataset_name}
    target_dir = os.path.join(BASE_DATA_DIR, data_type, dataset_name)
    save_path = f"{SAVE_DIR}/{data_type}_{dataset_name}_{model_name}.json"
    
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
                # 记录所有已处理过的图片路径
                processed_paths = {item["path"] for item in all_results}
                print(f"[INFO] 检测到已有文件，已跳过 {len(processed_paths)} 条记录")
        except Exception as e:
            print(f"[WARNING] 读取已有 JSON 失败，可能格式损坏: {e}")
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
            
            # vLLM 推理
            try:
                image_base64 = encode_image(img_path)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                outputs = llm.chat(
                    messages=messages, 
                    sampling_params=SamplingParams(temperature=0, max_tokens=max_new_tokens)
                )
                predict_answer = outputs[0].outputs[0].text

                # 构造保存项
                res_item = {
                    "filename": file,
                    "path": img_path,
                    "gt_label": label,
                    "predict": predict_answer,
                    "data_type": data_type,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                }
                
                # --- 3. 实时写入结果 ---
                all_results.append(res_item)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                # 更新已处理集合
                processed_paths.add(img_path)

            except Exception as e:
                print(f"\n[ERROR] 推理失败 {file}: {e}")
                continue
    
    print(f"\n[SUCCESS] 任务完成！结果已更新至: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["iol", "soi"], required=True, help="iol or soi")
    parser.add_argument("--dataset", type=str, required=True, help="e.g., mvtec， BTech_Dataset_transformed, VisA")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct", help="Qwen3-VL-2B-Instruct  Qwen3-VL-4B-Instruct,Qwen3-VL-8B-Instruct,Qwen3-VL-32B-Instruct")
    args = parser.parse_args()

    run_inference(args.type, args.dataset, args.model_name)