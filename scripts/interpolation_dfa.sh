#!/bin/bash
SECONDS=0
set -e        # exit when error
# set -o xtrace # print command

GPU=$1
object_name="$2"
rerun="$3"

if [ -z "$rerun" ]; then
    rerun=false
fi

declare -A idx_dict
idx_dict["fox(attitude)"]="90 110 125 150 170"
idx_dict["whiteTiger(run)"]="5 10 15 20 25"  # 5 10 15 20 25
idx_dict["beagle_dog(s1_24fps)"]="300 320 345 355 360"
idx_dict["wolf(Howling)"]="5 30 75 145 170"
idx_dict["duck(eat_grass)"]="0 10 20 55 60"
idx_dict["panda(acting)"]="95 100 115 215 245"

declare -A cam_dict
cam_dict["fox(attitude)"]="32"
cam_dict["whiteTiger(run)"]="32"
cam_dict["whiteTiger(roaringwalk)"]="32"
cam_dict["beagle_dog(s1_24fps)"]="32"
cam_dict["wolf(Howling)"]="32"
cam_dict["cat(walkprogressive_noz)"]="32"
cam_dict["duck(eat_grass)"]="32"
cam_dict["panda(acting)"]="32"

cam_idx=${cam_dict[$object_name]}
idx_list=(${idx_dict[$object_name]})

idx_from=${idx_list[0]}
idx_to=${idx_list[1]}

ckpt=./results/dfa/${object_name}_${idx_from}/ckpts/ckpt_best_psnr.pt

idx_start=${idx_list[0]}

if [ ! -f "$ckpt" ]; then
    bash scripts/train_dfa.sh $GPU $object_name $idx_from
fi

for idx_to in ${idx_list[@]:1}; do
    result_dir=./results/dfa_finetune/${object_name}_${idx_start}_${idx_from}_${idx_to}

    if [ ! -f $result_dir/ckpt_finetune.pt ] || $rerun; then
        CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
            --data_dir /data2/wlsgur4011/GESI/gsplat/data/DFA_processed/${object_name}/${idx_to}/ \
            --result_dir $result_dir \
            --ckpt $ckpt \
            --data_factor 1 \
            --data_name DFA \
            --single_finetune \
            --port 8081 \
            --scale_reg 0.1 \
            --object_name ${object_name} \
            --disable_viewer \
            --cam_idx $cam_idx \
            --wandb \
            --wandb_group interpolation
        fi
    
    ckpt=$result_dir/ckpt_finetune.pt
    idx_from=$idx_to
done

echo make motion video

idx_from=${idx_list[0]}
idx_to=${idx_list[1]}
for idx_to in ${idx_list[@]:1}; do
    result_dir=./results/dfa_finetune/${object_name}_${idx_start}_${idx_from}_${idx_to}

    CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
        --data_dir /data2/wlsgur4011/GESI/gsplat/data/DFA_processed/${object_name}/${idx_to}/ \
        --result_dir $result_dir \
        --ckpt $ckpt \
        --data_factor 1 \
        --data_name DFA \
        --single_finetune \
        --port 8081 \
        --scale_reg 0.1 \
        --object_name ${object_name} \
        --disable_viewer \
        --cam_idx $cam_idx \
        --motion_video \
        --video_name ${object_name}_${idx_start}_${idx_from}_${idx_to} \
        --simple_video

    ckpt=$result_dir/ckpt_finetune.pt
    idx_from=$idx_to
done


idx_from=${idx_list[0]}
idx_to=${idx_list[1]}
for i in {0..3}; do
    for idx_to in ${idx_list[@]:1}; do
        video_path=/data2/wlsgur4011/GESI/output_video/DFA_simple/${object_name}_${idx_start}_${idx_from}_${idx_to}_${i}.mp4
        video_paths="$video_paths ${video_path}"
        idx_from=$idx_to
    done

    for idx in ${idx_list[@]:0}; do
        image_path=/data2/wlsgur4011/GESI/gsplat/data/DFA_processed/${object_name}/${idx}/images/img_00${cam_idx}_rgba.png
        image_paths="$image_paths ${image_path}"
    done

    mkdir -p /data2/wlsgur4011/GESI/output_video_interpolated/dfa/ | true
    python ../gesi/video_integration.py \
        --image_paths $image_paths \
        --video_paths $video_paths \
        --output_path /data2/wlsgur4011/GESI/output_video_interpolated/dfa/${object_name}.mp4
done