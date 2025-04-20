# run normal
# scripts_name = ["finetuning_drag.sh", "run_wo_group_dfa.sh", "run_naive_group_dfa.sh.sh"]

scripts_name_list=("finetuning_drag_dfa.sh" "run_naive_group_dfa.sh" "run_wo_group_dfa.sh", "run_wo_group_refine_dfa.sh")
wandb_group_name_list=("v1.11_re" "v1.11_naive_group" "v1.11_wo_group", "v1.11_wo_group_refine")

# loop over three scripts
for i in {0..3}
do
    scripts_name=${scripts_name_list[$i]}
    wandb_group_name=${wandb_group_name_list[$i]}
    # 요거 dfa인데 왜 object name이 diva360이지?
    bash scripts/$scripts_name 4 horse 0120 0375 $wandb_group_name &
    bash scripts/$scripts_name 4 wolf 0000 2393 $wandb_group_name &
    bash scripts/$scripts_name 4 k1_push_up 0370 0398 $wandb_group_name &
    bash scripts/$scripts_name 4 k1_hand_stand 0000 0300 $wandb_group_name &
    bash scripts/$scripts_name 4 music_box 0100 0125 $wandb_group_name &
    bash scripts/$scripts_name 4 trex 0100 0300 $wandb_group_name &
    bash scripts/$scripts_name 4 k1_double_punch 0000 0555 $wandb_group_name &
    bash scripts/$scripts_name 4 world_globe 0020 0074 $wandb_group_name &
    bash scripts/$scripts_name 4 blue_car 0142 0214 $wandb_group_name &

    bash scripts/$scripts_name 5 truck 0078 0171 $wandb_group_name &
    bash scripts/$scripts_name 5 clock 0000 1500 $wandb_group_name &
    bash scripts/$scripts_name 5 penguin 0217 0239 $wandb_group_name &
    bash scripts/$scripts_name 5 wall_e 0222 0286 $wandb_group_name &
    bash scripts/$scripts_name 5 bunny 0000 1000 $wandb_group_name &
    bash scripts/$scripts_name 5 red_car 0042 0250 $wandb_group_name &
    bash scripts/$scripts_name 5 dog 0177 0279 $wandb_group_name &
    bash scripts/$scripts_name 5 stirling 0000 0045 $wandb_group_name &

    wait
    
    trap "exit" INT
done


