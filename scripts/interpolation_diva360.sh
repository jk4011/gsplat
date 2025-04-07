#!/bin/bash
SECONDS=0
set -e        # exit when error
# set -o xtrace # print command

GPU=$1
object_name=$2

declare -A idx_dict
idx_dict["wall_e"]="0222 0286 0303 0377 0387"
idx_dict["penguin"]="0047 0067 0160 0180 0654"
idx_dict["dog"]="0177 0179 0181 0183 0261"
idx_dict["trex"]="0002 0071 0125 0201 0321"
idx_dict["wolf"]="0350 0359 0727 0735 0747"
idx_dict["world_globe"]="0000 0035 0072 0108 0144"

cam_idx=0
idx_list=(${idx_dict[$object_name]})

for idx in ${idx_list[@]:0}; do
    bash scripts/preprocess_diva360.sh $object_name $idx | true
done

idx_from=${idx_list[0]}
idx_to=${idx_list[1]}

ckpt=./results/diva360/${object_name}_${idx_from}/ckpts/ckpt_best_psnr.pt
idx_start=${idx_list[0]}

if [ ! -f $ckpt ]; then
    bash scripts/train_diva360.sh $GPU $object_name $idx_from
fi

for idx_to in ${idx_list[@]:1}; do
    result_dir=./results/diva360_finetune/${object_name}_${idx_start}_${idx_from}_${idx_to}
    if [ ! -f $result_dir/ckpt_finetune.pt ]; then
        CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
            --data_dir /data2/wlsgur4011/GESI/gsplat/data/diva360_processed/${object_name}_${idx_to}/ \
            --result_dir $result_dir \
            --ckpt $ckpt \
            --data_factor 1 \
            --data_name diva360 \
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
    result_dir=./results/diva360_finetune/${object_name}_${idx_start}_${idx_from}_${idx_to}

    CUDA_VISIBLE_DEVICES=$GPU python examples/simple_trainer.py default \
        --data_dir /data2/wlsgur4011/GESI/gsplat/data/diva360_processed/${object_name}_${idx_to}/ \
        --result_dir $result_dir \
        --ckpt $ckpt \
        --data_factor 1 \
        --data_name diva360 \
        --single_finetune \
        --port 8081 \
        --scale_reg 0.1 \
        --object_name ${object_name} \
        --disable_viewer \
        --cam_idx $cam_idx \
        --motion_video \
        --video_name ${object_name}_${idx_start}_${idx_from}_${idx_to} \
        --simple_video \

    ckpt=$result_dir/ckpt_finetune.pt
    idx_from=$idx_to
done


idx_from=${idx_list[0]}
idx_to=${idx_list[1]}
for idx_to in ${idx_list[@]:1}; do
    video_path=/data2/wlsgur4011/GESI/output_video/diva360_simple/${object_name}_${idx_start}_${idx_from}_${idx_to}_0.mp4
    video_paths="$video_paths ${video_path}"
    idx_from=$idx_to
done

for idx in ${idx_list[@]:0}; do
    image_path=/data2/wlsgur4011/GESI/gsplat/data/diva360_processed/${object_name}_${idx}/images/cam0${cam_idx}.png
    image_paths="$image_paths ${image_path}"
done

mkdir -p /data2/wlsgur4011/GESI/output_video_interpolated/diva360/ | true
python ../gesi/video_integration.py \
    --image_paths $image_paths \
    --video_paths $video_paths \
    --output_path /data2/wlsgur4011/GESI/output_video_interpolated/diva360/${object_name}.mp4
