
import os
import json
import base64
import torch
import re
import glob
from tqdm import tqdm
from vllm import LLM, SamplingParams
from configs import MODEL_PATH, BASE_DATA_DIR, SAVE_DIR, max_new_tokens, encode_image, extract_answer, build_multimodal_prompt

# 配置路径
EXAMPLE_DIR = "./examples"

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

def run_inference(data_type, dataset_name, model_name, mode):
    llm = get_vllm_model(model_name)
    
    json_input_path = os.path.join(BASE_DATA_DIR, f"{dataset_name}_{data_type}.json")
    # 结果文件名区分模式
    save_path = f"{SAVE_DIR}_{mode}/{data_type}_{dataset_name}_{model_name}.json"
    
    if not os.path.exists(f"{SAVE_DIR}_{mode}"):
        os.makedirs(f"{SAVE_DIR}_{mode}")

    if not os.path.exists(json_input_path):
        print(f"[ERROR] 找不到 JSON 索引: {json_input_path}")
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
                print(f"[INFO] 跳过已处理的 {len(processed_paths)} 条记录")
        except: pass

    print(f"\n[INFO] 模式: {mode} | 任务: {dataset_name} ({data_type})")
    
    for meta_key, info in tqdm(image_metadata.items()):
        img_path = info.get("physical_path")
        if not img_path or not os.path.exists(img_path) or img_path in processed_paths:
            continue

        label = info.get("label", "").lower()
        if label not in ["anomaly", "normal"]:
            continue

        try:
            image_base64 = encode_image(img_path)
            # 动态构建包含示例的消息
            messages = build_multimodal_prompt(mode, img_path, image_base64)
            outputs = llm.chat(
                messages=messages, 
                sampling_params=SamplingParams(temperature=0, max_tokens=max_new_tokens)
            )
            
            predict_answer = outputs[0].outputs[0].text
            extract_ans = extract_answer(predict_answer)
            
            res_item = {
                "filename": info.get("filename"),
                "path": img_path,
                "gt_label": info.get("label"),
                "predict": predict_answer,
                "data_type": data_type,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "extract_answer": extract_ans,
                "gt": "no" if info.get("label") == "normal" else "yes",
                # 保留新模式中的扩展字段
                "resize_scale": info.get("resize_scale"),
                "original_count": info.get("count")
            }
            
            all_results.append(res_item)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            processed_paths.add(img_path)

        except Exception as e:
            print(f"\n[ERROR] {img_path}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="iol")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-4B-Instruct")
    # 新增模式参数
    parser.add_argument("--mode", type=str, default="two-examples", 
                        choices=["zero-shot", "one-example", "two-examples"],
                        help="Prompting mode")
    args = parser.parse_args()

    run_inference(args.type, args.dataset, args.model_name, args.mode)