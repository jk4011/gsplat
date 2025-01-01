#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

CUDA_VISIBLE_DEVICES=1 python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/Diva360_data/3dgs_data/penguin/ \
    --result_dir ./results/penguin \
    --data_factor 1 \
    --no_colmap \
    --port 8081 \
    --scale_reg 0.1 \
