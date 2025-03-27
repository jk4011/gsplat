#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
frame_index=$3
# 0239 0217

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/GESI/gsplat/data/diva360_processed/${object_name}_${frame_index}/ \
    --result_dir ./results/diva360/${object_name}_${frame_index} \
    --data_factor 1 \
    --data_name diva360 \
    --port 8081 \
    --scale_reg 0.1 \
    --disable_viewer \
    # --random_bkgd \
    