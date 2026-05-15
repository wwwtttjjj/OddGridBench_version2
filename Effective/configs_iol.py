import os
# models_dir = "../models/"
max_new_tokens = 2048

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../"))
models_dir = os.path.join(ROOT_DIR, "models")

def get_configs(args):
    if args.data_type == "icon":
        image_dir = "../create_data/test_data/image"
        json_path = "../create_data/test_data.json"
    elif args.data_type == "hanzi":
        image_dir = "../Test_data/hanzi/iol_test_data/images"
        json_path = "../Test_data/hanzi/iol_test_data/iol_test_data.json"
    elif args.data_type == "mnist":
        image_dir = "../Test_data/mnist/iol_test_data/images"
        json_path = "../Test_data/mnist/iol_test_data/iol_test_data.json"
    elif args.data_type == "VisA":
        image_dir = "../Test_data/VisA/iol_test_data/images"
        json_path = "../Test_data/VisA/iol_test_data/iol_test_data.json"
    elif args.data_type == "BTech":
        image_dir = "../Test_data/BTech_Dataset_transformed/iol_test_data/images"
        json_path = "../Test_data/BTech_Dataset_transformed/iol_test_data/iol_test_data.json"
    elif args.data_type == "MVTEC_loco":
        image_dir = "../Test_data/mvtec_loco/iol_test_data/images"
        json_path = "../Test_data/mvtec_loco/iol_test_data/iol_test_data.json"
    elif args.data_type == "MVTEC":
        image_dir = "../Test_data/mvtec/iol_test_data/images"
        json_path = "../Test_data/mvtec/iol_test_data/iol_test_data.json"
    elif args.data_type == "ELPV":
        image_dir = "../Test_data/ELPV/iol_test_data/images"
        json_path = "../Test_data/ELPV/iol_test_data/iol_test_data.json"
    elif args.data_type == "GOODADS":
        image_dir = "../Test_data/GOODADS/iol_test_data/images"
        json_path = "../Test_data/GOODADS/iol_test_data/iol_test_data.json"
    elif args.data_type == "RAD":
        image_dir = "../Test_data/RAD/iol_test_data/images"
        json_path = "../Test_data/RAD/iol_test_data/iol_test_data.json"
    elif args.data_type == "MPDD":
        image_dir = "../Test_data/MPDD/iol_test_data/images"
        json_path = "../Test_data/MPDD/iol_test_data/iol_test_data.json"

    # 输出路径
    Result_root = "iol_output/"

    if args.image_type == "with_number":
        image_dir = image_dir.replace("image", "image_number")
        Result_root = "output_number/"
        
    if not os.path.exists(Result_root):
        os.mkdir(Result_root)
    return {
        "image_type": args.image_type,
        "data_type": args.data_type,
        "image_dir": image_dir,
        "json_path": json_path,
        # "root_dir": root_dir,
        "Result_root": Result_root,
        "models_dir": models_dir,
        "model_path": os.path.join(models_dir, args.model_name),
        "save_path": os.path.join(Result_root, f"{args.data_type}_{args.model_name}.json"),
    }