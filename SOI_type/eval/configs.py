import os
models_dir = "/data/wengtengjin/models/"
max_new_tokens = 2048

def get_configs(args):
    # 目录定义

    image_dir = "../create_data/test_data/image"
    json_path = "../create_data/test_data.json"
    # root_dir = image_dir

    # 输出路径
    Result_root = "output/"

    if args.data_type == "with_number":
        image_dir = image_dir.replace("image", "image_number")
        Result_root = "output_number/"
        
    if not os.path.exists(Result_root):
        os.mkdir(Result_root)
    return {
        "image_type": "oddgridbench",
        "image_dir": image_dir,
        "json_path": json_path,
        # "root_dir": root_dir,
        "Result_root": Result_root,
        "models_dir": models_dir,
        "model_path": os.path.join(models_dir, args.model_name),
        "save_path": os.path.join(Result_root, f"{args.model_name}.json"),
    }