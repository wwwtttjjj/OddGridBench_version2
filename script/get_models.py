# #模型下载
# from modelscope import snapshot_download
# # model_dir = snapshot_download('Qwen/Qwen3-VL-8B-Instruct')
# # model_dir = snapshot_download('Qwen/Qwen3-VL-4B-Instruct')
# model_dir = snapshot_download('OpenGVLab/InternVL3_5-8B')
# model_dir = snapshot_download('OpenGVLab/InternVL3_5-4B')
# model_dir = snapshot_download('OpenGVLab/InternVL3_5-2B')
# model_dir = snapshot_download('OpenGVLab/InternVL3_5-38B')
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    local_dir="/nfsdata4/wengtengjin/oddgrid_task/models/LLaVA-OneVision-1.5-8B-Instruct",
    # use_auth_token=True,  # 如果需要授权
)