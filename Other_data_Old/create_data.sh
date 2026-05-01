# cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/BTech_Dataset_transformed
# bash run.sh
# cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/VisA
# bash run.sh
# cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/mvtec
# bash run.sh

samples=200
# python IOL_main.py --png_root=./hanzi/hanzi_png --samples=$samples
# python IOL_main.py --png_root=./mnist/mnist_pairs --samples=$samples
python SOI_main.py --png_root=./hanzi/hanzi_png --samples=$samples
python SOI_main.py --png_root=./mnist/mnist_pairs --samples=$samples