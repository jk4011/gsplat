# run normal
# scripts_name = ["finetuning_drag.sh", "run_wo_group_diva360.sh", "run_naive_group_diva360.sh.sh"]

scripts_name_list=("finetuning_drag_diva360.sh" "run_naive_group_diva360.sh" "run_wo_group_diva360.sh")
wandb_group_name_list=("v1.12_point_arap" "v1.12_naive_group" "v1.12_wo_group")

# loop over three scripts
for i in {0..2}
do
    scripts_name=${scripts_name_list[$i]}
    wandb_group_name=${wandb_group_name_list[$i]}
    bash scripts/$scripts_name 0 wolf 0000 2393 $wandb_group_name &&
    bash scripts/$scripts_name 0 horse 0120 0375 $wandb_group_name &&
    bash scripts/$scripts_name 0 k1_push_up 0370 0398 $wandb_group_name &

    bash scripts/$scripts_name 2 k1_hand_stand 0000 0300 $wandb_group_name &&
    bash scripts/$scripts_name 2 music_box 0100 0125 $wandb_group_name &&
    bash scripts/$scripts_name 2 trex 0100 0300 $wandb_group_name &

    bash scripts/$scripts_name 3 k1_double_punch 0000 0555 $wandb_group_name &&
    bash scripts/$scripts_name 3 world_globe 0020 0074 $wandb_group_name &

    bash scripts/$scripts_name 4 blue_car 0142 0214 $wandb_group_name &&
    bash scripts/$scripts_name 4 truck 0078 0171 $wandb_group_name &

    bash scripts/$scripts_name 5 clock 0000 1500 $wandb_group_name &&
    bash scripts/$scripts_name 5 penguin 0217 0239 $wandb_group_name &

    bash scripts/$scripts_name 6 wall_e 0222 0286 $wandb_group_name &&
    bash scripts/$scripts_name 6 bunny 0000 1000 $wandb_group_name &

    bash scripts/$scripts_name 7 red_car 0042 0250 $wandb_group_name &&
    bash scripts/$scripts_name 7 dog 0177 0279 $wandb_group_name &&
    bash scripts/$scripts_name 7 stirling 0000 0045 $wandb_group_name &

    wait
    # if keyboard interupt, stop all process
    trap "exit" INT

done

