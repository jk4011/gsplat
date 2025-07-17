#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
index_from=$3
index_to=$4
cam_idx=$5

ckpt=./results/dfa/${object_name}_finetune/ckpt_finetune.pt


if [ ! -f $ckpt ]; then
    bash scripts/train_dfa.sh $GPU $object_name $index_from
fi

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/GESI/gsplat/data/DFA_processed/${object_name}/${index_to} \
    --result_dir ./results/dfa/${object_name}_finetune \
    --ckpt $ckpt \
    --data_factor 1 \
    --data_name DFA \
    --cam_idx $cam_idx \
    --port 8081 \
    --scale_reg 0.1 \
    --object_name $object_name \
    --render_traj_all \
    --disable_viewer

