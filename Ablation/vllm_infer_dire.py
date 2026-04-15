# import os
# import json
# import base64
# import torch
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# from configs import MODEL_PATH, BASE_DATA_DIR, SAVE_DIR, max_new_tokens
# import re

# # =========================================
# _VLLM_MODEL = None

# def extract_answer(predict_answer):
#     match = re.search(r'box\{(Yes|No)\}', predict_answer, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()
#     return None

# def build_custom_prompt():
#     return f"Please analyze the image first, then determine if there are any defects and output the final answer as box{{Yes}} or box{{No}}."


# def get_vllm_model(model_name):
#     global _VLLM_MODEL
#     if _VLLM_MODEL is None:
#         tp = torch.cuda.device_count()
#         _VLLM_MODEL = LLM(
#             model=os.path.join(MODEL_PATH, model_name),
#             max_model_len=4096,
#             trust_remote_code=True,
#             tensor_parallel_size=tp,
#         )
#     return _VLLM_MODEL


# def encode_image(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def run_inference(data_type, dataset_name, model_name):
#     llm = get_vllm_model(model_name)
    
#     # --- 1. 构建输入 JSON 路径 (按新模式命名) ---
#     # 假设 JSON 存在 single_data/VisA_iol.json
#     json_input_path = os.path.join(BASE_DATA_DIR, f"{dataset_name}_{data_type}.json")
#     save_path = f"{SAVE_DIR}/{data_type}_{dataset_name}_{model_name}.json"
    
#     if not os.path.exists(json_input_path):
#         print(f"[ERROR] 找不到对应的 JSON 索引文件: {json_input_path}")
#         return

#     # 读取索引数据
#     with open(json_input_path, "r", encoding="utf-8") as f:
#         image_metadata = json.load(f)

#     # --- 2. 处理断点续传 ---
#     all_results = []
#     processed_paths = set()
#     if os.path.exists(save_path):
#         try:
#             with open(save_path, "r", encoding="utf-8") as f:
#                 all_results = json.load(f)
#                 processed_paths = {item["path"] for item in all_results}
#                 print(f"[INFO] 检测到已有文件，已跳过 {len(processed_paths)} 条记录")
#         except Exception as e:
#             print(f"[WARNING] 读取已有的输出 JSON 失败: {e}")

#     # --- 3. 遍历 JSON 中的图片条目 ---
#     print(f"\n[INFO] 开始推理任务: {dataset_name} ({data_type})")
    
#     # meta_key 通常是图片的相对路径
#     for meta_key, info in tqdm(image_metadata.items()):
        
#         # 获取物理路径 (从 JSON 字段中取)
#         img_path = info.get("physical_path")
#         if not img_path or not os.path.exists(img_path):
#             print(f"[SKIP] 物理文件不存在: {img_path}")
#             continue

#         # 断点续传判断
#         if img_path in processed_paths:
#             continue
        
#         label = info.get("label", "").lower()
#         if label not in ["anomaly", "normal"]:
#             continue

#         prompt = build_custom_prompt()
        
#         try:
#             image_base64 = encode_image(img_path)
#             messages = [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
#                     {"type": "text", "text": prompt}
#                 ]
#             }]
            
#             outputs = llm.chat(
#                 messages=messages, 
#                 sampling_params=SamplingParams(temperature=0, max_tokens=max_new_tokens)
#             )
            
#             predict_answer = outputs[0].outputs[0].text
#             extract_ans = extract_answer(predict_answer)
            
#             # 构造结果项 (融合原有的 info 信息)
#             res_item = {
#                 "filename": info.get("filename"),
#                 "path": img_path,
#                 "gt_label": info.get("label"),
#                 "predict": predict_answer,
#                 "data_type": data_type,
#                 "dataset_name": dataset_name,
#                 "model_name": model_name,
#                 "extract_answer": extract_ans,
#                 "gt": "no" if info.get("label") == "normal" else "yes",
#                 # 保留新模式中的扩展字段
#                 "resize_scale": info.get("resize_scale"),
#                 "original_count": info.get("count")
#             }
            
#             # 实时保存
#             all_results.append(res_item)
#             with open(save_path, "w", encoding="utf-8") as f:
#                 json.dump(all_results, f, ensure_ascii=False, indent=4)
            
#             processed_paths.add(img_path)

#         except Exception as e:
#             print(f"\n[ERROR] 推理失败 {img_path}: {e}")
#             continue
    
#     print(f"\n[SUCCESS] 任务完成！结果已更新至: {save_path}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--type", type=str, default="iol", help="iol or soi")
#     parser.add_argument("--dataset", type=str, default="mvtec", help="e.g., mvtec, VisA")
#     parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
#     args = parser.parse_args()

#     run_inference(args.type, args.dataset, args.model_name)


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

def extract_answer(predict_answer):
    match = re.search(r'box\{(Yes|No)\}', predict_answer, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_example_image(original_path):
    """
    将原始路径映射到 example 路径。
    输入示例: "single_data/BTech_Dataset_transformed/02/Normal/0150.png"
    输出示例: "./examples/BTech_Dataset_transformed/02/Normal/example.png"
    """
    if not original_path:
        return None
    
    # 1. 去掉最开头的目录 (比如 'single_data/')
    # split('/', 1) 得到 ['single_data', 'BTech_Dataset_transformed/02/Normal/0150.png']
    parts = original_path.split(os.sep)
    if len(parts) < 2:
        return None
        
    # 2. 提取中间的目录结构，并去掉最后的文件名 (0150.png)
    # 结果类似: BTech_Dataset_transformed/02/Normal
    sub_dir = os.path.dirname(os.path.join(*parts[1:]))
    
    # 3. 拼接新的路径，文件名固定为 example.png
    # 注意：这里会尝试匹配多种后缀，或者你直接写死 .png
    target_dir = os.path.join(EXAMPLE_DIR, sub_dir)
    
    # 搜索该目录下名为 example 的图片（支持多种后缀）
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG']
    for ext in extensions:
        potential_path = os.path.join(target_dir, f"example{ext}")
        if os.path.exists(potential_path):
            return potential_path
            
    print(f"[WARNING] 未找到示例图片: {target_dir}/example.*")
    return None


def build_multimodal_prompt(mode, img_path, current_img_base64):
    """
    根据模式构建对话消息列表
    """
    messages = []
    
    # 基础指导语
    instruction = "Please analyze the image first, then determine if there are any defects and output the final answer as box{Yes} or box{No}."

    # 1. 添加正例 (One-example 或 Two-examples 模式)
    if mode in ["one-example", "two-examples"]:
        pos_path = find_example_image(img_path)
        if pos_path:
            pos_b64 = encode_image(pos_path)
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pos_b64}"}},
                {"type": "text", "text": "This is a normal sample without defects. Answer: box{No}"}
            ]})
            messages.append({"role": "assistant", "content": "This is a reference for a normal object."})

    # 2. 添加负例 (仅 Two-examples 模式)
    if mode == "two-examples":
        neg_path = find_example_image(img_path)
        if neg_path:
            neg_b64 = encode_image(neg_path)
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{neg_b64}"}},
                {"type": "text", "text": "This is an anomalous sample with defects. Answer: box{Yes}"}
            ]})
            messages.append({"role": "assistant", "content": "This is a reference for a defective object."})

    # 3. 添加当前待检测图片
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_base64}"}},
            {"type": "text", "text": instruction}
        ]
    })
    
    return messages

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
            print(f"\n[DEBUG] 构建的消息: {messages}")
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
                "mode": mode,
                "dataset_name": dataset_name,
                "extract_answer": extract_ans,
                "gt": "no" if info.get("label") == "normal" else "yes"
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
    parser.add_argument("--mode", type=str, default="one-example", 
                        choices=["zero-shot", "one-example", "two-examples"],
                        help="Prompting mode")
    args = parser.parse_args()

    run_inference(args.type, args.dataset, args.model_name, args.mode)