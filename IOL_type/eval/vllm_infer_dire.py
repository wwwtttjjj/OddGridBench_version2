import os
os.environ.setdefault("VLLM_USE_V1", "0")
import argparse
import json
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torch
from configs import get_configs, max_new_tokens
from utils import *

# ===== 新增：vLLM =====
from vllm import LLM, SamplingParams
from types import SimpleNamespace

# ===============================
# vLLM 初始化（全局，只初始化一次）
# ===============================
_VLLM_MODEL = None

def is_qwen35_model(model_path):
    return "qwen3.5" in os.path.basename(model_path).lower()


def is_gemma4_model(model_path):
    return os.path.basename(model_path).lower().startswith("gemma-4")


def needs_transformers_fallback(model_path):
    return is_qwen35_model(model_path) or is_gemma4_model(model_path)


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


def get_vllm_model(model_path):
    global _VLLM_MODEL
    tp = torch.cuda.device_count()
    if _VLLM_MODEL is None:
        if needs_transformers_fallback(model_path):
            print(f"[INFO] {os.path.basename(model_path)} is not supported by vLLM 0.8.x; using Transformers fallback.")
            _VLLM_MODEL = TransformersChatModel(model_path)
            return _VLLM_MODEL

        _VLLM_MODEL = LLM(
            model=model_path,
            max_model_len=12000,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.8,
        )
    return _VLLM_MODEL


def call_vllm_server(prompt, image_paths, model_path):
    llm = get_vllm_model(model_path)

    content = []

    for img_path in image_paths:
        with Image.open(img_path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        })

    content.append({
        "type": "text",
        "text": prompt.strip()
    })

    messages = [{
        "role": "user",
        "content": content
    }]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    try:
        chat_kwargs = {}
        if is_qwen35_model(model_path):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            **chat_kwargs,
        )
        return outputs[0].outputs[0].text
    except Exception as e:
        print(f"[ERROR] vLLM inference failed: {e}")
        return None


def run_vllm_http(args):
    """读取 JSON -> vLLM 推理 -> 写结果（无 HTTP）"""
    configs_para = get_configs(args)

    json_path = configs_para["json_path"]
    model_path = configs_para["model_path"]
    save_json_path = configs_para["save_path"]

    if not json_path or not model_path:
        raise ValueError(f"Unknown model_name: {args.model_name}")

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

        image_names = [data.get("image")]
        image_paths = [os.path.join(configs_para["image_dir"], img_name) for img_name in image_names]
        
        if args.data_type in ["GOODADS", "RAD", "MPDD"]:
            prompt = build_prompt_different_angle(data)
        elif args.data_type in ["icon", "mnist", "hanzi"]:
            prompt = build_prompt_same_angle_synthesis(data)
        else:
            prompt = build_prompt_same_angle_real(data)

        predict_answer = call_vllm_server(prompt, image_paths, model_path)
        extract_answer = extract_answer_from_response(predict_answer)

        rows_cols = []
        odd_list = data.get("odd_list", [])
        for odd in odd_list:
            row = odd.get("row")
            col = odd.get("col")
            rows_cols.append((row, col))

        save_item = {
            "id": id,
            "image":data.get("image"),
            "class": data.get("class", ""),
            "prompt": prompt,
            "predict_answer": predict_answer,
            "extract_answer": extract_answer,
            "answer": rows_cols if rows_cols != [] else data.get("odd_rows_cols", []),
            "odd_list": odd_list,
            "odd_count": data.get("odd_count"),
            "grid_size": str(data.get("grid_size")),
        }
        write_json(save_json_path, save_item)

    print(f"[INFO] ✅ Done! Saved results to {save_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multimodal inference via vLLM Python API")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--image_type",
        type=str,
        default="normal",
        help="normal or with_number"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="RAD",
        help="icon, mnist, hanzi,VisA, BTech, MVTEC, ELPV, GOODADS, RAD, MPDD, MVTEC_loco"
    )

    args = parser.parse_args()
    run_vllm_http(args)


if __name__ == "__main__":
    main()
