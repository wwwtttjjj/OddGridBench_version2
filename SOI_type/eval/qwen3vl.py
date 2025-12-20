import os
import argparse
import json
import base64
import requests
from tqdm import tqdm

from configs import get_configs, max_new_tokens
from utils import *

# 可以按需改成环境变量
API_URL = "http://localhost:8081/v1/chat/completions"

def call_vllm_server(prompt, image_paths, model_path):
    """通过 vLLM REST API 调用模型"""
    messages = [{"role": "user", "content": []}]
    print(image_paths)
    # 图像转 base64
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    # 文本（使用我们限制“不要复杂推理”的 prompt）
    messages[0]["content"].append({
        "type": "input_text",
        "text": prompt.strip()
    })

    payload = {
        "model": model_path,      # 必须和 vllm serve 启动的模型标识一致
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
        msg = ""
        if resp is not None:
            msg = resp.text[:300]
        else:
            msg = str(e)
        print(f"[ERROR] vLLM request failed: {msg}")
        return None
    
def run_vllm_http(args):
    """读取 JSON -> 调 HTTP -> 写结果（串行，服务端负责并发与多卡）"""
    configs_para = get_configs(args)
    # Result_root = configs_para["Result_root"]

    json_path = configs_para["json_path"]
    model_path = configs_para["model_path"]
    save_json_path = configs_para["save_path"]
    if os.path.exists(save_json_path):
        os.remove(save_json_path)

    if not json_path or not model_path:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    # print(configs_para)
    # 已有结果 -> 去重
    processed_ids = set()
    if os.path.exists(save_json_path):
        with open(save_json_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                if "id" in item:
                    processed_ids.add(item["id"])

    # 读测试集
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    print(f"[INFO] Total samples: {len(json_data)}; processed: {len(processed_ids)}")

    for data in tqdm(json_data):
        id = data.get("id")
        if id in processed_ids:
            continue

        image_names = [os.path.join(data.get("image"), str(i)+".png") for i in range(1, data.get("total_icons") + 1)]  # list of image paths
        image_paths = [os.path.join(configs_para["image_dir"], img_name) for img_name in image_names]
        prompt = build_prompt(image_paths)

        predict_answer = call_vllm_server(prompt, image_paths, model_path)
        extract_answer = extract_answer_from_response(predict_answer)
        odd_lists = []

        odd_list = data.get("odd_icons", [])
        for odd in odd_list:
            odd_lists.append(odd.get("icon_name"))

        save_item = {
            "id": id,
            "image":data.get("image"),
            "image_num":data.get("total_icons"),
            "prompt": prompt,
            "predict_answer": predict_answer,
            "extract_answer": extract_answer,
            "answer": odd_lists,
            "odd_list": odd_list,
            "odd_count": data.get("num_odds"),
        }
        write_json(save_json_path, save_item)

    print(f"[INFO] ✅ Done! Saved results to {save_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multimodal inference via vLLM HTTP API")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")  # Qwen3-VL-2B-Instruct / Qwen3-VL-6B-Instruct
    parser.add_argument(
        "--data_type",
        type=str,
        default="normal",
        help="normal or with_number"
    )
    
    args = parser.parse_args()

    run_vllm_http(args)


if __name__ == "__main__":
    main()