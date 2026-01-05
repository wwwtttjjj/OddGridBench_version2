import json
import os
import shutil
import uuid
from pathlib import Path

def process_json_and_images(src_json_path: str, output_root_dir: str):
    # 新增：清空输出目录（若存在则删除后重建）
    output_root = Path(output_root_dir).absolute()
    if output_root.exists():
        print(f"清空已有输出目录：{output_root}")
        shutil.rmtree(output_root)  # 递归删除目录及所有内容
    # 重新创建空的输出目录结构
    output_root.mkdir(exist_ok=True)
    images_output_dir = output_root / "images"
    images_output_dir.mkdir(exist_ok=True)
    print(f"已创建全新输出目录：{output_root}")
    print(f"已创建全新图片输出目录：{images_output_dir}")

    # 2. 读取原始JSON文件
    with open(src_json_path, 'r', encoding='utf-8') as f:
        raw_data_list = json.load(f)
    print(f"成功读取原始JSON，共 {len(raw_data_list)} 条数据")

    # 3. 处理每条数据，构建新JSON列表
    new_json_list = []
    for raw_item in raw_data_list:
        # 3.1 提取原始字段
        raw_image_name = raw_item.get("image", "")
        raw_abs_path = raw_item.get("abs_path", "")
        raw_boxes = raw_item.get("boxes", {})

        # 3.2 计算odd_count和odd_rows_cols（修复后逻辑）
        odd_rows_cols = []
        grid_position = []
        if isinstance(raw_boxes, dict):
            grid_position = raw_boxes.get("grid_position", [])
        elif isinstance(raw_boxes, list):
            grid_position = []
        else:
            grid_position = []
        
        if isinstance(grid_position, list):
            if len(grid_position) > 0 and isinstance(grid_position[0], list):
                odd_rows_cols = grid_position.copy()
            elif len(grid_position) == 2 and all(isinstance(x, int) for x in grid_position):
                odd_rows_cols = [grid_position.copy()]
        
        odd_count = len(odd_rows_cols)

        # 3.3 构建新图片名称
        new_image_name = raw_image_name
        if raw_abs_path:
            path_parts = Path(raw_abs_path).parts
            if len(path_parts) >= 2:
                parent_dir_name = path_parts[-2]
                new_image_name = f"{parent_dir_name}_{raw_image_name}"

        # 3.4 复制图片到目标目录
        src_image_path = Path(raw_abs_path).absolute()
        dst_image_path = images_output_dir / new_image_name
        if src_image_path.exists() and src_image_path.is_file():
            counter = 1
            temp_dst_path = dst_image_path
            while temp_dst_path.exists():
                name, ext = os.path.splitext(new_image_name)
                temp_dst_path = images_output_dir / f"{name}_{counter}{ext}"
                counter += 1
            shutil.copy2(src_image_path, temp_dst_path)
            new_image_name = temp_dst_path.name
            # print(f"图片已复制：{src_image_path} → {temp_dst_path}")
        else:
            print(f"警告：原始图片不存在，跳过复制：{src_image_path}")

        # 3.5 构建新JSON条目
        new_item = {
            "id": str(uuid.uuid4()),
            "grid_size": [3, 5],
            "odd_count": odd_count,
            "odd_rows_cols": odd_rows_cols,
            "gap": None,
            "margin": None,
            "image": new_image_name,
            "image_size": [1700, 1000],
            "source_dataset": "mvtec_loco_pushpins",
        }
        new_json_list.append(new_item)

    # 4. 保存新JSON文件
    new_json_path = output_root / "iol_test_data.json"
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_list, f, ensure_ascii=False, indent=2)
    print(f"新JSON文件已保存：{new_json_path}")
    print(f"处理完成！共生成 {len(new_json_list)} 条新数据")

if __name__ == "__main__":
    # -------------------------- 配置参数（请根据你的实际情况修改） --------------------------
    # 原始JSON文件的路径（绝对路径 / 相对路径均可）
    SOURCE_JSON_PATH = "an.json"  # 替换为你的原始json文件路径
    # 输出根目录（脚本会在此目录下创建images子目录和新json文件）
    OUTPUT_ROOT_DIR = "iol_test_data"  # 可自定义目录名
    # ----------------------------------------------------------------------------------------

    # 执行处理函数
    process_json_and_images(SOURCE_JSON_PATH, OUTPUT_ROOT_DIR)