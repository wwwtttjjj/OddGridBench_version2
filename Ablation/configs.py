import os
# models_dir = "../models/"
max_new_tokens = 2048

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models")

# MODEL_PATH = r"/nfsdata4/wengtengjin/oddgrid_task/models/Qwen3-VL-8B-Instruct"
BASE_DATA_DIR = "./single_data"
SAVE_DIR = "./results"  #

EXAMPLE_DIR = "./examples"  # 正样本和负样本