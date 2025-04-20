cd /data2/wlsgur4011/GESI/gsplat &&  conda activate gsplat

bash scripts/finetuning_drag_diva360.sh 0 penguin 0217 0239 0 tmp &
bash scripts/finetuning_drag_diva360.sh 1 bunny 0000 1000 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 2 dog 0177 0279 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 3 k1_double_punch 0000 0555 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 4 k1_hand_stand 0000 0300 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 5 k1_push_up 0370 0398 0 v1.19_group_refinement &
wait

bash scripts/finetuning_drag_diva360.sh 0 horse 0120 0375 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 1 trex 0100 0300 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 2 wall_e 0222 0286  0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 3 wolf 0000 2393 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 4 truck 0078 0171 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 5 clock 0000 1500 0 v1.19_group_refinement &
wait

bash scripts/finetuning_drag_diva360.sh 0 music_box 0100 0125 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 1 world_globe 0020 0074 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 2 blue_car 0142 0214 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 3 red_car 0042 0250 0 v1.19_group_refinement &
bash scripts/finetuning_drag_diva360.sh 4 stirling 0000 0045 0 v1.19_group_refinement &
