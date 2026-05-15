source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate



python model_merger.py --local_dir=/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Train_code/RL_code/checkpoints/Anomaly_detection/Qwen3_vl_4B_TOTAL_grpo/global_step_100/actor/

mv /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Train_code/RL_code/checkpoints/Anomaly_detection/Qwen3_vl_4B_TOTAL_grpo/global_step_100/actor/huggingface_model /nfsdata4/wengtengjin/oddgrid_task/models/