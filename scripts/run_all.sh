cd /data2/wlsgur4011/GESI/gsplat &&  conda activate gsplat



bash scripts/finetuning_drag_diva360.sh 4 penguin 0217 0239 tmp
bash scripts/finetuning_drag_diva360.sh 1 bunny 0000 1000 tmp &
bash scripts/finetuning_drag_diva360.sh 1 dog 0177 0279 tmp &
bash scripts/finetuning_drag_diva360.sh 2 k1_double_punch 0000 0555 tmp &
bash scripts/finetuning_drag_diva360.sh 2 k1_hand_stand 0000 0300 tmp &
bash scripts/finetuning_drag_diva360.sh 3 k1_push_up 0370 0398 tmp &
bash scripts/finetuning_drag_diva360.sh 3 penguin 0217 0239 tmp &
bash scripts/finetuning_drag_diva360.sh 4 trex 0100 0300 tmp &
bash scripts/finetuning_drag_diva360.sh 4 wall_e 0222 0286  tmp &
bash scripts/finetuning_drag_diva360.sh 5 wolf 0000 2393 tmp &
bash scripts/finetuning_drag_diva360.sh 5 truck 0078 0171 tmp &


bash scripts/make_motion.sh 3 penguin 0217 0239 tmp
bash scripts/make_motion.sh 1 bunny 0000 1000 tmp &
bash scripts/make_motion.sh 1 dog 0177 0279 tmp &
bash scripts/make_motion.sh 2 k1_double_punch 0000 0555 tmp &
bash scripts/make_motion.sh 2 k1_hand_stand 0000 0300 tmp &
bash scripts/make_motion.sh 3 k1_push_up 0370 0398 tmp &
bash scripts/make_motion.sh 3 penguin 0217 0239 tmp &
bash scripts/make_motion.sh 4 trex 0100 0300 tmp &
bash scripts/make_motion.sh 4 wall_e 0222 0286  tmp &
bash scripts/make_motion.sh 5 wolf 0000 2393 tmp &
bash scripts/make_motion.sh 5 truck 0078 0171 tmp &

bash scripts/wandb_sweep_diva360.sh 0 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 0 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 2 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 2 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 3 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 3 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 4 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 4 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 5 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 5 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 6 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 6 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 7 v1.9_vis_mask &
bash scripts/wandb_sweep_diva360.sh 7 v1.9_vis_mask &

# bash scripts/finetuning_drag_diva360.sh 1 blue_car 0142 0214
# bash scripts/finetuning_drag_diva360.sh 1 clock 0000 1500

# bash scripts/finetuning_drag_diva360.sh 2 horse 0120 0375
# bash scripts/finetuning_drag_diva360.sh 2 hour_glass 0000 1755

# bash scripts/finetuning_drag_diva360.sh 3 music_box 0095 0129
# bash scripts/finetuning_drag_diva360.sh 3 plasma_ball 0000 0100

# bash scripts/finetuning_drag_diva360.sh 4 plasma_ball_clip 0059 0185
# bash scripts/finetuning_drag_diva360.sh 4 red_car 0042 0250
# bash scripts/finetuning_drag_diva360.sh 4 stirling 0000 0612
# bash scripts/finetuning_drag_diva360.sh 4 tornado 0000 0456 


# bash scripts/finetuning_drag_diva360.sh 0 world_globe 0387 0483
