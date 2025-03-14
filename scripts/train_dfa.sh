# bash scripts/train_dfa.sh 0 beagle_dog 0
#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2  # beagle_dog bear cat duck fox lion whiteTiger wolf
frame_index=$3

CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
    --data_dir /data2/wlsgur4011/GESI/gsplat/data/DFA_processed/${object_name}/${frame_index} \
    --result_dir ./results/${object_name}_${frame_index} \
    --data_factor 1 \
    --data_name DFA \
    --port 8081 \
    --scale_reg 0.1 \
    # --disable_viewer \
    # --random_bkgd \
    