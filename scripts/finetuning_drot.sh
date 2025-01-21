#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

object_name=penguin
origin_idx=0217
fine_tuning_idx=0239

GPU=$1

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/Diva360_data/3dgs_data/${object_name}_${fine_tuning_idx}/ \
    --result_dir ./results/${object_name}_finetune \
    --ckpt ./results/${object_name}_${origin_idx}/ckpts/ckpt_2999_rank0.pt \
    --data_factor 1 \
    --no_colmap \
    --single_finetune \
    --port 8081 \
    --scale_reg 0.1 \
    --finetuning_drot \


set +x; duration=SECONDS; RED='\033[0;31m'; Yellow='\033[1;33m'; Green='\033[0;32m'; NC='\033[0m'; echo -e "RED$((duration / 3600))hNC Yellow$((duration / 60 % 60))mNC Green$((duration % 60))sNC elapsed."
    