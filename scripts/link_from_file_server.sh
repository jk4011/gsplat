#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

# mkdir data if not exist
data_dir=/data2/wlsgur4011/GESI/gsplat/data
mkdir -p $data_dir

ln -s /data/wlsgur4011/GESI/DFA_processed $data_dir
ln -s /data/rvi/dataset/Diva360_data $data_dir
ln -s /data/wlsgur4011/DFA $data_dir
ln -s /data/wlsgur4011/GESI/diva360_processed $data_dir
ln -s /data/wlsgur4011/GESI/results /data2/wlsgur4011/GESI/gsplat/results

ln -s /data/wlsgur4011/DFA/.cache /tmp/
