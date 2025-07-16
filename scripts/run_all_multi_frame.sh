#!/bin/bash

# bash scripts/run_all_multi_frame.sh finetuning_drag_diva360.sh multi_OURS
# bash scripts/run_all_multi_frame.sh finetuning_drag_diva360.sh multi_GESI
# bash scripts/run_all_multi_frame.sh finetuning_drag_diva360.sh multi_

SECONDS=0
set -e        # exit when error
set -o xtrace # print command

script_name=$1
wandb_group_name=$2

# script_name=make_motion_diva360.sh
# script_name=finetuning_drag_diva360.sh

# wandb_group_name=tmp

bash scripts/${script_name} 0   dog             0177     0279   00      ${wandb_group_name} &&
bash scripts/${script_name} 0   penguin         0217     0239   00      ${wandb_group_name} &&
bash scripts/${script_name} 0   wall_e          0222     0286   00      ${wandb_group_name} &&
bash scripts/${script_name} 0   wolf            0357     1953   00      ${wandb_group_name} &&
bash scripts/${script_name} 0   k1_hand_stand   0412     0426   01      ${wandb_group_name} &


bash scripts/${script_name} 1   dog             0177     0379   00      ${wandb_group_name} &&
bash scripts/${script_name} 1   penguin         0217     0339   00      ${wandb_group_name} &&
bash scripts/${script_name} 1   wall_e          0222     0386   00      ${wandb_group_name} &&
bash scripts/${script_name} 1   wolf            0357     1053   00      ${wandb_group_name} &&
bash scripts/${script_name} 1   k1_hand_stand   0412     0526   01      ${wandb_group_name} &


bash scripts/${script_name} 2   penguin         0217     0439   00      ${wandb_group_name} &&
bash scripts/${script_name} 2   dog             0177     0479   00      ${wandb_group_name} &&
bash scripts/${script_name} 2   wall_e          0222     0486   00      ${wandb_group_name} &&
bash scripts/${script_name} 2   wolf            0357     1553   00      ${wandb_group_name} &&
bash scripts/${script_name} 2   k1_hand_stand   0412     0626   01      ${wandb_group_name} &


bash scripts/${script_name} 3   dog             0177     0579   00      ${wandb_group_name} &&
bash scripts/${script_name} 3   penguin         0217     0539   00      ${wandb_group_name} &&
bash scripts/${script_name} 3   wall_e          0222     0586   00      ${wandb_group_name} &&
bash scripts/${script_name} 3   wolf            0357     1253   00      ${wandb_group_name} &&
bash scripts/${script_name} 3   k1_hand_stand   0412     0726   01      ${wandb_group_name} &


bash scripts/${script_name} 4   dog             0177     0679   00      ${wandb_group_name} &&
bash scripts/${script_name} 4   penguin         0217     0639   00      ${wandb_group_name} &&
bash scripts/${script_name} 4   wall_e          0222     0686   00      ${wandb_group_name} &&
bash scripts/${script_name} 4   wolf            0357     0753   00      ${wandb_group_name} &&
bash scripts/${script_name} 4   k1_hand_stand   0412     0826   01      ${wandb_group_name} &

