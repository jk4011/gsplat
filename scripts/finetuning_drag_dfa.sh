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
    --data_dir /data2/wlsgur4011/GESI/gsplat/data/DFA_processed/${object_name}/${index_to} \
    --result_dir ./results/dfa/${object_name}_finetune \
    --ckpt ./results/dfa/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt \
    --data_factor 1 \
    --data_name DFA \
    --single_finetune \
    --cam_idx $cam_idx \
    --port 8081 \
    --scale_reg 0.1 \
    --object_name ${object_name}_[$index_from,$index_to] \
    --wandb \
    --wandb_group ${version} \
    --disable_viewer \


