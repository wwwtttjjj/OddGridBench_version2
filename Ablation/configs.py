import os
import os
import json
import base64
from io import BytesIO
from PIL import Image
import re
import glob
from tqdm import tqdm
# models_dir = "../models/"
max_new_tokens = 2048

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models")

BASE_DATA_DIR = "./single_data"
SAVE_DIR = "./results"  #

EXAMPLE_DIR = "./examples"  # 正样本和负样本


def extract_answer(predict_answer):
    if not predict_answer:
        return None
    
    # 修改点：将 \{ 改为 \{1,2}，将 \} 改为 \}1,2}
    # 这样可以同时匹配 box{No} 和 box{{No}}
    match = re.search(r'boxed\{{1,2}(Yes|No)\}{1,2}', predict_answer, re.IGNORECASE)
    
    if match:
        # .capitalize() 会把 "yes" 变成 "Yes", "no" 变成 "No"
        return match.group(1).capitalize()
    
    return None

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def find_example_image(original_path, target_type="Normal"):
    """
    将原始路径映射到对应的示例路径。
    :param original_path: 当前待测图路径，如 "single_data/BTech/02/Normal/0150.png"
    :param target_type: 想要获取的示例类型，可选 "Normal" 或 "Anomaly" (需对应你文件夹名)
    :return: 对应的 example.png 物理路径
    """
    if not original_path:
        return None
    
    # 1. 分割路径
    parts = original_path.split(os.sep)
    if len(parts) < 2:
        return None
        
    # 2. 核心逻辑：强制替换类别目录
    # 假设你的路径结构中倒数第二级是类别名（Normal/Anomaly）
    # 我们遍历路径部分，把原本的类别名替换为我们想要的 target_type
    new_parts = []
    for p in parts[1:]: # 跳过最开头的 'single_data'
        # 如果这一级目录是 Normal 或 Anomaly (忽略大小写)，则替换它
        if p.lower() in ["normal", "anomaly", "abnormal"]:
            new_parts.append(target_type)
        else:
            new_parts.append(p)
    
    # 去掉最后的文件名 (0150.png)，获取目录部分
    sub_dir = os.path.dirname(os.path.join(*new_parts))
    target_dir = os.path.join(EXAMPLE_DIR, sub_dir)
    
    # 3. 寻找文件
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG']
    for ext in extensions:
        potential_path = os.path.join(target_dir, f"example{ext}")
        if os.path.exists(potential_path):
            return potential_path
            
    # print(f"[WARNING] 未找到 {target_type} 示例: {target_dir}/example.*")
    return None


def build_multimodal_prompt(mode, img_path, current_img_base64):
    """构建多模态对话消息列表"""
    messages = []
    
    # 1. 正常参考
    if mode in ["one-example", "two-examples"]:
        pos_path = find_example_image(img_path, target_type="Normal")
        pos_b64 = encode_image(pos_path)
        if pos_b64:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pos_b64}"}},
                {"type": "text", "text": "This is a [Standard Normal Sample] without any defects. It serves as your quality baseline."}
            ]})
            messages.append({"role": "assistant", "content": "Understood. I have analyzed the normal sample and will use it as a reference."})

    # 2. 异常参考
    if mode == "two-examples":
        neg_path = find_example_image(img_path, target_type="Anomaly")
        neg_b64 = encode_image(neg_path)
        if neg_b64:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{neg_b64}"}},
                {"type": "text", "text": "This is an [Anomalous Sample] that contains defects. Please note these irregular features."}
            ]})
            messages.append({"role": "assistant", "content": "Understood. I have identified the defective features for comparison."})

    output_relu = f"""Strictly adhere to the following output rules\n
    1.You may perform observation and comparative analysis before answering.
    2. The FINAL ANSWER must be contained within exactly ONE \\boxed{{}} block.\n"""
    # 3. 最终指令
    if mode == "zero-shot":
        instruction = f"""{output_relu}
        3. Determine if there are any defects, and finally output the answer as boxed{{Yes}} or boxed{{No}}."""
    elif mode == "one-example":
        instruction = f"""
        {output_relu}
        3.Compared to the [Standard Normal Sample] provided earlier, does this image show any deviations or defects? 
        4. Based on your analysis, output the final answer as boxed{{Yes}} or boxed{{No}}."""
    else: # two-examples
        instruction = f"""
        {output_relu} 
        3. By comparing it with both the [Standard Normal Sample] and the [Anomalous Sample] above, determine if this image is defective. 
        4. Based on your comparison, output the final answer as boxed{{Yes}} or boxed{{No}}."""

    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_base64}"}},
            {"type": "text", "text": instruction}
        ]
    })
    return messages