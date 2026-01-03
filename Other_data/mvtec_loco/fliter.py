import os
import shutil
from pathlib import Path

def classify_bottle_images(top_dir: str):
    # 1. 定义目标目录（在bottle下新建normal和abnormal）
    top_path = Path(top_dir).absolute()
    normal_dir = top_path / "Normal"
    abnormal_dir = top_path / "Abnormal"
    
    # 创建目标目录（已存在则跳过）
    normal_dir.mkdir(exist_ok=True)
    abnormal_dir.mkdir(exist_ok=True)
    print(f"已创建目标目录：{normal_dir}\n{abnormal_dir}")

    # 2. 遍历bottle下所有文件（包括子目录）
    for root, _, files in os.walk(top_path):
        # 判断当前目录的上级是否包含"good"
        # 规则：只要上级目录有"good"，则当前目录下的png属于normal
        is_normal = "good" in Path(root).parts  # parts是路径的各层级列表
        if "Normal" in Path(root).parts or "Abnormal" in Path(root).parts:
            continue
        # 处理当前目录下的png文件
        for file in files:
            if file.lower().endswith(".png"):  # 兼容大小写后缀
                src_path = Path(root) / file
                # 确定目标目录
                dst_dir = normal_dir if is_normal else abnormal_dir
                dst_path = dst_dir / file

                # 避免文件名重复：重复则加后缀（如 xxx_1.png）
                counter = 1
                while dst_path.exists():
                    name, ext = os.path.splitext(file)
                    dst_path = dst_dir / f"{name}_{counter}{ext}"
                    counter += 1

                # 复制文件（若需移动则用shutil.move）
                shutil.copy2(src_path, dst_path)  # copy2保留文件元信息
                print(f"已分类：{src_path} → {dst_path}")
        # for file in files:
        #     if file.lower().endswith(".png"):  # 兼容大小写后缀
        #         if "mask" in file.lower():
        #             os.remove(Path(root) / file)

if __name__ == "__main__":
    # 替换为你的bottle目录绝对路径（比如 "/xxx/bottle"）
    dir_list = [
        "./pushpins/",
    ]
    for BOTTLE_DIR in dir_list:
        classify_bottle_images(BOTTLE_DIR)