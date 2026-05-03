import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def prepare_output_dir(output_dir, overwrite=True):
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists")
        print(f"[清理] 删除已有目录: {output_dir}")
        shutil.rmtree(output_dir)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    return output_dir


def load_json(json_path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path, data):
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_unique_name(used_names, image_name, prefix):
    """
    给图片生成不冲突的新名字。
    注意：这个函数仍然在主线程中调用，避免多线程竞争 used_names。
    """
    if image_name not in used_names:
        return image_name

    image_path = Path(image_name)
    parent = image_path.parent
    stem = image_path.stem
    suffix = image_path.suffix

    index = 1
    while True:
        if str(parent) == ".":
            new_name = f"{prefix}_{stem}_{index}{suffix}"
        else:
            new_name = str(parent / f"{prefix}_{stem}_{index}{suffix}")

        if new_name not in used_names:
            return new_name

        index += 1


def copy_image_task(src_path, dst_path):
    """
    支持复制文件或目录。
    SOI 里有些 image 字段可能对应一个目录。
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        shutil.copy2(src_path, dst_path)

    elif src_path.is_dir():
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)

    else:
        raise FileNotFoundError(f"Source path not found: {src_path}")

    return str(dst_path)


def collect_copy_tasks_for_one_dataset(
    data,
    images_dir,
    output_images_dir,
    input_prefix,
    used_image_names,
):
    """
    先在主线程中生成所有复制任务和新的 JSON item。
    这样可以保证：
    1. used_image_names 不会被多线程同时修改；
    2. total_data 的顺序和原始 data 保持一致；
    3. 图片复制本身再交给线程池并发执行。
    """
    new_data = []
    copy_tasks = []

    for item in data:
        old_image_name = item["image"]

        new_image_name = make_unique_name(
            used_image_names,
            old_image_name,
            input_prefix,
        )

        used_image_names.add(new_image_name)

        new_item = dict(item)
        new_item["image"] = new_image_name
        new_data.append(new_item)

        src_image_path = images_dir / old_image_name
        dst_image_path = output_images_dir / new_image_name

        copy_tasks.append((src_image_path, dst_image_path))

    return new_data, copy_tasks


def run_copy_tasks(copy_tasks, max_workers=8, progress_prefix=""):
    """
    多线程复制图片。
    """
    if not copy_tasks:
        return

    finished = 0
    total = len(copy_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(copy_image_task, src_path, dst_path): (src_path, dst_path)
            for src_path, dst_path in copy_tasks
        }

        for future in as_completed(future_to_task):
            src_path, dst_path = future_to_task[future]

            try:
                future.result()
            except Exception as e:
                print(f"[错误] 复制失败: {src_path} -> {dst_path}")
                print(f"[错误信息] {e}")
                raise

            finished += 1

            if finished % 100 == 0 or finished == total:
                print(f"[进度] {progress_prefix}: 图片复制 {finished}/{total}")


def merge_test_data(
    input_dir_list,
    data_type,
    output_dir,
    overwrite=True,
    max_workers=8,
):
    """
    通用合并函数。

    data_type:
        - "iol"
        - "soi"
    """
    assert data_type in ["iol", "soi"], "data_type must be 'iol' or 'soi'"

    input_subdir_name = f"{data_type}_test_data"
    json_file_name = f"{data_type}_test_data.json"

    print(f"\n========== 开始合并 {input_subdir_name} ==========")

    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    output_images_dir = output_dir / "images"
    output_json_path = output_dir / json_file_name

    total_data = []
    used_image_names = set()
    all_copy_tasks = []

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

        new_data, copy_tasks = collect_copy_tasks_for_one_dataset(
            data=data,
            images_dir=images_dir,
            output_images_dir=output_images_dir,
            input_prefix=input_dir.name,
            used_image_names=used_image_names,
        )

        total_data.extend(new_data)
        all_copy_tasks.extend(copy_tasks)

        print(f"[收集] {input_dir.name} {data_type.upper()}: {len(copy_tasks)} 张图片")

    print(f"\n[复制] 开始多线程复制 {data_type.upper()} 图片，总数: {len(all_copy_tasks)}")
    run_copy_tasks(
        copy_tasks=all_copy_tasks,
        max_workers=max_workers,
        progress_prefix=data_type.upper(),
    )

    save_json(output_json_path, total_data)

    print(f"\n========== {data_type.upper()} 合并完成 ==========")
    print(f"[输出目录] {output_dir}")
    print(f"[JSON 数量] {len(total_data)}")
    print(f"[图片数量] {len(used_image_names)}")


def main():
    # =========================
    # 输入配置
    # =========================
    input_dir_list = [
        r"GOODADS",
        r"MVTEC_LOCO",
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
    max_workers = 8

    # =========================
    # 开始合并
    # =========================
    merge_test_data(
        input_dir_list=input_dir_list,
        data_type="iol",
        output_dir=iol_output_dir,
        overwrite=overwrite,
        max_workers=max_workers,
    )

    merge_test_data(
        input_dir_list=input_dir_list,
        data_type="soi",
        output_dir=soi_output_dir,
        overwrite=overwrite,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    main()