# # # bash train_configs/qwen3_vl_dapo_loop.sh
# # # bash train_configs/qwen3_vl_dapo.sh
# # bash train_configs/qwen3_vl_dapo.sh
# # bash train_configs/qwen3_vl_gspo_loop.sh
# # bash train_configs/qwen3_vl_grpo_loop.sh
# bash train_configs/qwen3_vl_dapo_32B_lora.sh
python3 - <<'PY'
import verl, pathlib
print("verl file:", verl.__file__)
root = pathlib.Path(verl.__file__).parent

for p in root.rglob("*.py"):
    txt = p.read_text(errors="ignore")
    if "class ModelConfig" in txt:
        print("ModelConfig:", p)
        print("contains lora:", "lora" in txt)
PY