
import os
os.environ.setdefault("VLLM_USE_V1", "0")
import json
import base64
import torch
import re
import glob
from tqdm import tqdm
from vllm import LLM, SamplingParams
from types import SimpleNamespace
from configs import MODEL_PATH, BASE_DATA_DIR, SAVE_DIR, max_new_tokens, encode_image, extract_answer, build_multimodal_prompt

# 配置路径
EXAMPLE_DIR = "./examples"

_VLLM_MODEL = None

def is_qwen35_model(model_name):
    return "qwen3.5" in os.path.basename(model_name).lower()


def is_gemma4_model(model_name):
    return os.path.basename(model_name).lower().startswith("gemma-4")


def needs_transformers_fallback(model_name):
    return is_qwen35_model(model_name) or is_gemma4_model(model_name)


def is_internvl_model(model_name):
    return "internvl" in os.path.basename(model_name).lower()


class TransformersChatModel:
    def __init__(self, model_path):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def chat(self, messages, sampling_params, **kwargs):
        template_kwargs = kwargs.pop("chat_template_kwargs", {})
        kwargs.update(template_kwargs)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=sampling_params.max_tokens,
            )
        text = self.processor.decode(
            generated[0][input_len:],
            skip_special_tokens=True,
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]


def get_vllm_model(model_name):
    global _VLLM_MODEL
    if _VLLM_MODEL is None:
        tp = torch.cuda.device_count()
        model_path = os.path.join(MODEL_PATH, model_name)
        if needs_transformers_fallback(model_name):
            print(f"[INFO] {model_name} is not supported by vLLM 0.8.x; using Transformers fallback.")
            _VLLM_MODEL = TransformersChatModel(model_path)
            return _VLLM_MODEL

        llm_kwargs = {
            "model": model_path,
            "max_model_len": 4096,
            "trust_remote_code": True,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": 0.8,
        }
        if is_internvl_model(model_name):
            llm_kwargs["mm_processor_kwargs"] = {"max_dynamic_patch": 1}

        _VLLM_MODEL = LLM(**llm_kwargs)
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
            chat_kwargs = {}
            if is_qwen35_model(model_name):
                chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

            outputs = llm.chat(
                messages=messages, 
                sampling_params=SamplingParams(temperature=0, max_tokens=max_new_tokens),
                **chat_kwargs,
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