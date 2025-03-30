#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
# object_name=wall_e
# index_from=0222
# index_to=0286

object_name=penguin
index_from=0217
index_to=0239

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/GESI/gsplat/data/diva360_processed/${object_name}_${index_to}/ \
    --result_dir ./results/${object_name}_finetune \
    --ckpt ./results/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt \
    --data_factor 1 \
    --no_colmap \
    --single_finetune \
    --port 8081 \
    --scale_reg 0.1 \
    --wandb \
    --wandb_name ${object_name} \
    --wandb_group debug \
    # --disable_viewer \

