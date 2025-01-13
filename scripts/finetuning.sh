#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

object_name=penguin
origin_idx=0217
fine_tuning_idx=0239

CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/Diva360_data/3dgs_data/${object_name}_${fine_tuning_idx}/ \
    --result_dir ./results/${object_name}_finetune \
    --ckpt ./results/${object_name}_${origin_idx}/ckpts/ckpt_2999_rank0.pt \
    --data_factor 1 \
    --no_colmap \
    --single_finetune \
    --port 8081 \
    --scale_reg 0.1 \


    