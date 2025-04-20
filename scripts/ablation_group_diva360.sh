# run normal
# scripts_name = ["finetuning_drag.sh", "run_wo_group_diva360.sh", "run_naive_group_diva360.sh.sh"]

scripts_name_list=("finetuning_drag_diva360.sh" "run_naive_group_diva360.sh" "run_wo_group_diva360.sh" "run_wo_group_refine_diva360.sh")
wandb_group_name_list=("v2.4_data_modified" "v2.4_naive_group" "v2.4_wo_group" "v2.4_wo_group_refine")

# loop over three scripts
for i in {0..3}
do
    scripts_name=${scripts_name_list[$i]}
    wandb_group_name=${wandb_group_name_list[$i]}
    echo $scripts_name
    echo $wandb_group_name

    bash scripts/$scripts_name 1 hour_glass 0100 0200 00 $wandb_group_name &
    bash scripts/$scripts_name 1 wolf 0357 1953 00 $wandb_group_name &
    bash scripts/$scripts_name 1 trex 0135 0250 00 $wandb_group_name &

    bash scripts/$scripts_name 2 bunny 0000 1000 00 $wandb_group_name &
    bash scripts/$scripts_name 2 world_globe 0020 0074 00 $wandb_group_name &
    bash scripts/$scripts_name 2 blue_car 0142 0214 00 $wandb_group_name &

    bash scripts/$scripts_name 3 red_car 0042 0250 00 $wandb_group_name &
    bash scripts/$scripts_name 3 penguin 0217 0239 00 $wandb_group_name &
    bash scripts/$scripts_name 3 dog 0177 0279 00 $wandb_group_name &

    bash scripts/$scripts_name 4 horse 0120 0375 00 $wandb_group_name &
    bash scripts/$scripts_name 4 wall_e 0222 0286 00 $wandb_group_name &
    bash scripts/$scripts_name 4 stirling 0000 0045 00 $wandb_group_name &

    bash scripts/$scripts_name 5 tornado 0000 0456 00 $wandb_group_name &
    bash scripts/$scripts_name 5 k1_hand_stand 0412 0426 01 $wandb_group_name &
    bash scripts/$scripts_name 5 k1_double_punch 0270 0282 01 $wandb_group_name &

    bash scripts/$scripts_name 6 k1_push_up 0541 0557 01 $wandb_group_name &
    bash scripts/$scripts_name 6 music_box 0100 0125 00 $wandb_group_name &

    bash scripts/$scripts_name 7 truck 0078 0171 00 $wandb_group_name &
    bash scripts/$scripts_name 7 clock 0000 1500 00 $wandb_group_name &


    wait
    # if keyboard interupt, stop all process
    trap "exit" INT

done

