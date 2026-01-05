import os
import argparse
import json
import base64
import requests
from tqdm import tqdm

# =========================
# vLLM HTTP API
# =========================
API_URL = "http://localhost:8081/v1/chat/completions"
MAX_NEW_TOKENS = 2048

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG"}


# =========================
# vLLM 调用（保持不变）
# =========================
def call_vllm_server(prompt, image_paths, model_path):
    messages = [{"role": "user", "content": []}]

    # image -> base64
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        })

    # text
    messages[0]["content"].append({
        "type": "input_text",
        "text": prompt.strip()
    })

    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
    }

    resp = None
    try:
        resp = requests.post(API_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        msg = resp.text[:300] if resp is not None else str(e)
        print(f"[ERROR] vLLM request failed: {msg}")
        return None


# =========================
# 递归扫描图片
# =========================
def collect_images(image_root):
    image_paths = []
    for root, _, files in os.walk(image_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                image_paths.append(os.path.join(root, fn))
    image_paths.sort()
    return image_paths


# =========================
# Detection Prompt
# =========================
def build_detection_prompt():
    return """
    你是一个目标检测模型。检测图像中的主要物体（我主要是为了去掉背景），返回这个物体的，目标边界框，像素坐标为 [x1, y1, x2, y2]，仅返回 JSON 数组。
    [
    {"label": "目标名称", "box": [x1, y1, x2, y2]}
    ]
    """


# =========================
# 解析 box（宽松）
# =========================
def parse_boxes(response):
    if response is None:
        return []

    try:
        return json.loads(response)
    except Exception:
        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            return json.loads(response[start:end])
        except Exception:
            print("[WARN] Failed to parse boxes")
            return []


# =========================
# 主流程
# =========================
def run(args):
    image_root = args.image_dir
    model_path = args.model_path
    save_json_path = args.save_json

    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

    image_paths = collect_images(image_root)
    print(f"[INFO] Found {len(image_paths)} images under {image_root}")

    # 支持断点续跑
    results = []
    processed = set()

    if os.path.exists(save_json_path):
        with open(save_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        for item in results:
            processed.add(item["image"])

    prompt = build_detection_prompt()

    for img_path in tqdm(image_paths):
        rel_path = os.path.relpath(img_path, image_root)
        if rel_path in processed:
            continue

        predict = call_vllm_server(
            prompt=prompt,
            image_paths=[img_path],
            model_path=model_path
        )

        boxes = parse_boxes(predict)

        results.append({
            "image": rel_path,
            "abs_path": img_path,
            "predict_raw": predict,
            "boxes": boxes
        })

        # 实时写盘
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] ✅ Done! Results saved to {save_json_path}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Run object detection via vLLM HTTP API on an image directory"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/nfsdata4/wengtengjin/oddgrid_task/models/Qwen3-VL-32B-Instruct",
        help="Model name/path used when starting vLLM serve"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="pipe_fryum",
        help="Root directory of images (recursive)"
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="./detect_results.json",
        help="Output json path"
    )

    args = parser.parse_args()
    image_list = ["capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor"]
    
    for image_name in image_list:
        args.image_dir = "manual_images/" + image_name
        args.save_json = "./" + args.image_dir + "/" + image_name + ".json"
        run(args)


if __name__ == "__main__":
    main()