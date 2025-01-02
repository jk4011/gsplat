#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

object_name=penguin
frame_index=0217

CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/Diva360_data/3dgs_data/${object_name}_${frame_index}/ \
    --result_dir ./results/${object_name}_${frame_index} \
    --data_factor 1 \
    --no_colmap \
    --port 8081 \
    --scale_reg 0.1 \
