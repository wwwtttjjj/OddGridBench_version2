import os
import json
import re
from tqdm import tqdm

# ================= 配置区 =================
# 设定你保存结果的根目录，会递归查找所有 json
RESULTS_DIRS = [
    "./results_one-example",
]
# =========================================

def extract_answer_robust(predict_answer):
    """
    增强版提取函数：
    支持 box{Yes}, box{{Yes}}, box{ Yes }, box{{ No }} 等各种情况
    """
    if not predict_answer or not isinstance(predict_answer, str):
        return None
    
    # 匹配 box 后面跟着一个或多个大括号，忽略空格，匹配 Yes 或 No
    match = re.search(r'box\{+\s*(Yes|No)\s*\}+', predict_answer, re.IGNORECASE)
    
    if match:
        return match.group(1).capitalize()
    return None

def fix_json_files():
    for base_dir in RESULTS_DIRS:
        if not os.path.exists(base_dir):
            print(f"[SKIP] 目录不存在: {base_dir}")
            continue
        
        print(f"[PROCESS] 正在处理目录: {base_dir}")
        
        # 遍历目录下所有 json 文件
        for filename in os.listdir(base_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(base_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not isinstance(data, list):
                        continue
                    
                    changed = False
                    for item in data:
                        # 获取模型原始输出
                        predict_raw = item.get("predict", "")
                        # 重新提取
                        new_ans = extract_answer_robust(predict_raw)
                        
                        # 如果提取结果不一样，则更新
                        if item.get("extract_answer") != new_ans:
                            item["extract_answer"] = new_ans
                            changed = True
                    
                    # 只有当内容发生变化时才写回文件，保护硬盘
                    if changed:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        print(f"  [FIXED] {filename}")
                    else:
                        print(f"  [OK] {filename} (无需修改)")
                        
                except Exception as e:
                    print(f"  [ERROR] 处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    fix_json_files()
    print("\n所有结果重新提取完成！")