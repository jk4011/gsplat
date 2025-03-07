#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
version=$2
object_name=1
index_from=1
index_to=1

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/Diva360_data/3dgs_data/${object_name}_${index_to}/ \
    --result_dir ./results/${object_name}_finetune \
    --ckpt ./results/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt \
    --data_factor 1 \
    --no_colmap \
    --single_finetune \
    --port 8081 \
    --scale_reg 0.1 \
    --object_name ${object_name} \
    --wandb \
    --wandb_group ${version} \
    --disable_viewer \
    --wandb_sweep \


set +x; duration=SECONDS; RED='\033[0;31m'; Yellow='\033[1;33m'; Green='\033[0;32m'; NC='\033[0m'; echo -e "RED$((duration / 3600))hNC Yellow$((duration / 60 % 60))mNC Green$((duration % 60))sNC elapsed."
