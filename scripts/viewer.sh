#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

CUDA_VISIBLE_DEVICES=1 python examples/simple_viewer.py \
    --ckpt=/data2/wlsgur4011/GESI/gsplat/results/diva360/dog_0177/ckpts/ckpt_best_psnr.pt \
    --port 8082

# /data2/wlsgur4011/GESI/gsplat/results/dfa/beagle_dog_s1_24fps_100/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/beagle_dog_s1_24fps_280/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/beagle_dog_s1_400/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/beagle_dog_s1_50/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/bear_run_0/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/bear_walk_110/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/cat_run_30/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/cat_walk_final_10/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/cat_walkprogressive_noz_220/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/cat_walksniff_30/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/duck_eat_grass_50/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/duck_eat_grass_60/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/duck_swim_110/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/duck_walk_0/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/duck_walk_200/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/fox_attitude_60/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/fox_walk_10/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/lion_Run_0/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/lion_Run_10/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/lion_Walk_10/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Damage_0/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Damage_10/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Howling_0/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Howling_10/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Run_20/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Walk_20/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Walk_30/ckpts/ckpt_best_psnr.pt
# /data2/wlsgur4011/GESI/gsplat/results/dfa/wolf_Walk_70/ckpts/ckpt_best_psnr.pt