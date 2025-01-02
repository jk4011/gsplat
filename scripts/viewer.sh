#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

CUDA_VISIBLE_DEVICES=1 python examples/simple_viewer.py \
    --ckpt=/data2/wlsgur4011/GESI/gsplat/results/penguin/ckpts/ckpt_29999.pt \
    --port 8083
