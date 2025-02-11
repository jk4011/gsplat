#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

object_name=$1
frame_index=$2

data_folder=/data2/wlsgur4011/Diva360_data
origin_folder=${data_folder}/processed_data/${object_name}/frames_1
processed_folder=${data_folder}/3dgs_data/${object_name}_${frame_index}


# link images
mkdir -p ${processed_folder}/images | true

cam_list=`ls ${origin_folder}`

for cam_folder in ${cam_list}; do
    filename=${origin_folder}/${cam_folder}/0000${frame_index}.png
    ln -s ${filename} ${processed_folder}/images/${cam_folder}.png
done

# create camera meta json
train_json_path=${data_folder}/processed_data/${object_name}/transforms_train.json
val_json_path=${data_folder}/processed_data/${object_name}/transforms_val.json
merged_json_path=${processed_folder}/cameras.json

jq -s '{
  frames: (.[0].frames + .[1].frames),
  aabb_scale: .[0].aabb_scale
}' ${train_json_path} ${val_json_path} > ${merged_json_path}
sed -i 's|undist/||g; s|/0000[0-9]\{4\}||g' ${merged_json_path}