#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
index_from=$3
index_to=$4
cam_idx=$5
version=$6

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/GESI/gsplat/data/diva360_processed/${object_name}_${index_to}/ \
    --result_dir ./results/diva360/${object_name}_finetune \
    --ckpt ./results/diva360/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt \
    --data_factor 1 \
    --data_name diva360 \
    --single_finetune \
    --port 8081 \
    --scale_reg 0.1 \
    --object_name ${object_name}_[$index_from,$index_to] \
    --cam_idx $cam_idx \
    --wandb \
    --wandb_group ${version} \
    --disable_viewer \
    --naive_group \
    --without_group_refine \


