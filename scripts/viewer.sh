#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

CUDA_VISIBLE_DEVICES=1 python examples/simple_viewer.py \
    --ckpt=/data2/wlsgur4011/GESI/gsplat/results/penguin_0239/ckpts/ckpt_best_psnr.pt \
    --port 8081
