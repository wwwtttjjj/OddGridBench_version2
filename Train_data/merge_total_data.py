import json
import shutil
from pathlib import Path


def prepare_output_dir(output_dir, overwrite=True):
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists")
        print(f"[清理] 删除已有目录: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_json(json_path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path, data):
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_items_keep_original_path_for_one_dataset(
    data,
    images_dir,
):
    """
    不复制图片，直接把 image 字段改成原始图片路径。

    原始 item:
        {"image": "xxx.png", ...}

    修改后:
        {"image": "GOODADS/iol_test_data/images/xxx.png", ...}
    """
    new_data = []

    for item in data:
        old_image_name = item["image"]

        new_item = dict(item)

        # 使用相对路径，便于数据集整体移动
        new_item["image"] = str( "../../Train_data" / images_dir / old_image_name)

        # 如果你想保存绝对路径，用下面这一行替换上面那一行：
        # new_item["image"] = str((images_dir / old_image_name).resolve())

        new_data.append(new_item)

    return new_data


def merge_test_data(
    input_dir_list,
    data_type,
    output_dir,
    overwrite=True,
):
    """
    通用合并函数。

    data_type:
        - "iol"
        - "soi"

    功能：
        1. 读取多个数据集目录下的 iol_test_data.json / soi_test_data.json
        2. 不复制图片
        3. 直接把 image 字段改成原始图片路径
        4. 合并保存到新的 JSON 文件
    """
    assert data_type in ["iol", "soi"], "data_type must be 'iol' or 'soi'"

    input_subdir_name = f"{data_type}_test_data"
    json_file_name = f"{data_type}_test_data.json"

    print(f"\n========== 开始合并 {input_subdir_name} ==========")

    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    output_json_path = output_dir / json_file_name

    total_data = []

    for dir_index, input_dir in enumerate(input_dir_list, start=1):
        input_dir = Path(input_dir)

        data_dir = input_dir / input_subdir_name
        images_dir = data_dir / "images"
        json_path = data_dir / json_file_name

        if not data_dir.exists():
            print(f"[跳过] 未找到目录: {data_dir}")
            continue

        if not json_path.exists():
            print(f"[跳过] 未找到 JSON: {json_path}")
            continue

        if not images_dir.exists():
            print(f"[跳过] 未找到图片目录: {images_dir}")
            continue

        print(f"\n[{dir_index}/{len(input_dir_list)}] 处理 {data_type.upper()}: {data_dir}")

        data = load_json(json_path)
        print(f"[读取] json 数量: {len(data)}")

        new_data = collect_items_keep_original_path_for_one_dataset(
            data=data,
            images_dir=images_dir,
        )

        total_data.extend(new_data)

        print(f"[收集] {input_dir.name} {data_type.upper()}: {len(new_data)} 条数据")

    save_json(output_json_path, total_data)

    print(f"\n========== {data_type.upper()} 合并完成 ==========")
    print(f"[输出目录] {output_dir}")
    print(f"[输出 JSON] {output_json_path}")
    print(f"[JSON 数量] {len(total_data)}")


def main():
    # =========================
    # 输入配置
    # =========================
    input_dir_list = [
        r"GOODADS",
        r"MVTEC_LOCO",
        r"RAD",
        r"MPDD",
        r"BTech_Dataset_transformed",
        r"VisA",
        r"mvtec",
        r"mvtec_ad2",
    ]

    # =========================
    # 输出配置
    # =========================
    iol_output_dir = r"total_data_iol"
    soi_output_dir = r"total_data_soi"

    # =========================
    # 运行配置
    # =========================
    overwrite = True

    # =========================
    # 开始合并
    # =========================
    merge_test_data(
        input_dir_list=input_dir_list,
        data_type="iol",
        output_dir=iol_output_dir,
        overwrite=overwrite,
    )

    merge_test_data(
        input_dir_list=input_dir_list,
        data_type="soi",
        output_dir=soi_output_dir,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()