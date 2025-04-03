cd /data2/wlsgur4011/GESI/gsplat &&  conda activate gsplat

bash scripts/finetuning_drag_diva360.sh 4 penguin 0217 0239 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 4 bunny 0000 1000 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 4 dog 0177 0279 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 k1_double_punch 0000 0555 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 k1_hand_stand 0000 0300 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 k1_push_up 0370 0398 v1.16_re &
wait

bash scripts/finetuning_drag_diva360.sh 4 horse 0120 0375 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 4 trex 0100 0300 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 4 wall_e 0222 0286  v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 wolf 0000 2393 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 truck 0078 0171 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 clock 0000 1500 v1.16_re &
wait

bash scripts/finetuning_drag_diva360.sh 4 music_box 0100 0125 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 4 world_globe 0020 0074 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 4 blue_car 0142 0214 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 red_car 0042 0250 v1.16_re &
bash scripts/finetuning_drag_diva360.sh 5 stirling 0000 0045 v1.16_re &

trap "exit" INT