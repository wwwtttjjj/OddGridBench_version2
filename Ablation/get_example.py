import os
import shutil
import random

def copy_random_images(src_root, dst_root):
    # 遍历 single_data 下的所有子目录
    for root, dirs, files in os.walk(src_root):
        # 过滤出常见的图片格式
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        if img_files:
            # 1. 随机选择一张图片
            random_img = random.choice(img_files)
            
            # 2. 获取原文件的后缀名 (例如 '.png' 或 '.jpg')
            file_extension = os.path.splitext(random_img)[1]
            
            # 3. 计算对应的目标路径
            relative_path = os.path.relpath(root, src_root)
            target_dir = os.path.join(dst_root, relative_path)
            
            # 4. 确保目标目录存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 5. 构造新的文件名：example.后缀
            new_name = f"example{file_extension}"
            
            # 6. 执行复制
            src_path = os.path.join(root, random_img)
            dst_path = os.path.join(target_dir, new_name)
            
            shutil.copy2(src_path, dst_path)
            print(f"已处理: {root} -> {new_name}")

# 设置路径
source_folder = 'single_data'
destination_folder = 'examples'

copy_random_images(source_folder, destination_folder)
print("\n所有子目录已完成随机抽取并重命名为 example.*")