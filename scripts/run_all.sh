cd /data2/wlsgur4011/GESI/gsplat &&  conda activate gsplat



bash scripts/finetuning_drag.sh 1 bunny 0000 1000 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 1 dog 0177 0279 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 2 k1_double_punch 0000 0555 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 2 k1_hand_stand 0000 0300 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 3 k1_push_up 0370 0398 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 3 penguin 0217 0239 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 4 trex 0100 0300 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 4 wall_e 0222 0285  v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 5 wolf 0000 2393 v1.3.2_n_anchor_increase &
bash scripts/finetuning_drag.sh 5 truck 0078 0171 v1.3.2_n_anchor_increase &


bash scripts/wandb_sweep.sh 0 v1.5_hpo &
bash scripts/wandb_sweep.sh 0 v1.5_hpo &
bash scripts/wandb_sweep.sh 1 v1.5_hpo &
bash scripts/wandb_sweep.sh 1 v1.5_hpo &
bash scripts/wandb_sweep.sh 2 v1.5_hpo &
bash scripts/wandb_sweep.sh 2 v1.5_hpo &
bash scripts/wandb_sweep.sh 3 v1.5_hpo &
bash scripts/wandb_sweep.sh 3 v1.5_hpo &
bash scripts/wandb_sweep.sh 4 v1.5_hpo &
bash scripts/wandb_sweep.sh 4 v1.5_hpo &
bash scripts/wandb_sweep.sh 5 v1.5_hpo &
bash scripts/wandb_sweep.sh 5 v1.5_hpo &
# bash scripts/finetuning_drag.sh 1 blue_car 0141 0214
# bash scripts/finetuning_drag.sh 1 clock 0000 1500

# bash scripts/finetuning_drag.sh 2 horse 0120 0375
# bash scripts/finetuning_drag.sh 2 hour_glass 0000 1755

# bash scripts/finetuning_drag.sh 3 music_box 0095 0129
# bash scripts/finetuning_drag.sh 3 plasma_ball 0000 0100

# bash scripts/finetuning_drag.sh 4 plasma_ball_clip 0059 0185
# bash scripts/finetuning_drag.sh 4 red_car 0042 0250
# bash scripts/finetuning_drag.sh 4 stirling 0000 0612
# bash scripts/finetuning_drag.sh 4 tornado 0000 0456 


# bash scripts/finetuning_drag.sh 0 world_globe 0387 0483
