#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,5,6
source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate




bash mygrpo/qwen3_vl_2b_oddgrid_cispo.sh
