
import os
import json
import base64
import torch
import re
import glob
from tqdm import tqdm
from vllm import LLM, SamplingParams
from configs import MODEL_PATH, BASE_DATA_DIR, SAVE_DIR, max_new_tokens

# 配置路径
EXAMPLE_DIR = "./examples"

_VLLM_MODEL = None





# def build_multimodal_prompt(mode, img_path, current_img_base64):
#     messages = []
    
#     # --- 第一部分：提供参考示例 ---
#     if mode in ["one-example", "two-examples"]:
#         pos_path = find_example_image(img_path, target_type="Normal") # 假设逻辑能找到 Normal 文件夹下的 example
#         print(f"[DEBUG] 正例路径: {pos_path}")
#         if pos_path:
#             pos_b64 = encode_image(pos_path)
#             messages.append({"role": "user", "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pos_b64}"}},
#                 {"type": "text", "text": "The image above shows a [Standard Normal Sample]. It has no defects and serves as your reference for quality."}
#             ]})
#             messages.append({"role": "assistant", "content": "I have recorded the features of the normal sample for reference."})

#     if mode == "two-examples":
#         # 注意：这里你可能需要修改 find_example_image 逻辑，让它能区分找 Normal 还是 Anomaly
#         # 假设这里逻辑已经处理好能找到负例
#         neg_path = find_example_image(img_path, target_type="Anomaly")
#         print(f"[DEBUG] 负例路径: {neg_path}")

#         if neg_path:
#             neg_b64 = encode_image(neg_path)
#             messages.append({"role": "user", "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{neg_b64}"}},
#                 {"type": "text", "text": "The image above shows an [Anomalous Sample]. It contains typical defects. Please observe its appearance carefully."}
#             ]})
#             messages.append({"role": "assistant", "content": "I have recorded the features of the anomalous sample."})

#         # --- 第二部分：构建动态指令 (关键改动) ---
#     if mode == "zero-shot":
#             instruction = f"Please analyze the image first, then determine if there are any defects and output the final answer as box{{Yes}} or box{{No}}."
#     elif mode == "one-example":
#         instruction = "Please analyze the current image first. Compared to the [Standard Normal Sample] provided earlier, determine if this image is defective. Based on your comparison, output the final answer as box{{Yes}} or box{{No}}."
#     elif mode == "two-examples":
#         instruction = "Please analyze the current image first. By comparing it with both the [Standard Normal Sample] and the [Anomalous Sample] above, determine if this image is defective. Based on your comparison, output the final answer as box{{Yes}} or box{{No}}."

#     # --- 第三部分：添加当前图 ---
#     messages.append({
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_base64}"}},
#             {"type": "text", "text": f"{instruction}"}
#         ]
#     })
    
#     return messages

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
            
            # res_item = {
            #     "filename": info.get("filename"),
            #     "path": img_path,
            #     "gt_label": info.get("label"),
            #     "predict": predict_answer,
            #     "mode": mode,
            #     "dataset_name": dataset_name,
            #     "extract_answer": extract_ans,
            #     "gt": "no" if info.get("label") == "normal" else "yes"
            # }
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
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    # 新增模式参数
    parser.add_argument("--mode", type=str, default="two-examples", 
                        choices=["zero-shot", "one-example", "two-examples"],
                        help="Prompting mode")
    args = parser.parse_args()

    run_inference(args.type, args.dataset, args.model_name, args.mode)