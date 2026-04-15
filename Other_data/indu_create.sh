# cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/BTech_Dataset_transformed
# bash run.sh
# cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/VisA
# bash run.sh
# cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/mvtec
# bash run.sh

python Indu_IOL_main.py
python Indu_SOI_main.py

# nohup bash indu_create.sh > create.log 2>&1 &