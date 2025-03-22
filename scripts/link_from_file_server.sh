#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

# mkdir data if not exist
mkdir -p /data2/wlsgur4011/GESI/gsplat/data

ln -s /data/wlsgur4011/GESI/DFA_processed /data2/wlsgur4011/GESI/gsplat/data
ln -s /data/wlsgur4011/GESI/diva360_processed /data2/wlsgur4011/GESI/gsplat/data
ln -s /data/wlsgur4011/GESI/results /data2/wlsgur4011/GESI/gsplat/results
ln -s /data/wlsgur4011/DFA/.cache /tmp/
