# run normal
# scripts_name = ["finetuning_drag.sh", "run_wo_group_dfa.sh", "run_naive_group_dfa.sh.sh"]

scripts_name_list=("finetuning_drag_dfa.sh" "run_naive_group_dfa.sh" "run_wo_group_dfa.sh", "run_wo_group_refine_dfa.sh")
wandb_group_name_list=("v8.0_re" "v8.0_naive_group" "v8.0_wo_group", "v8.0_wo_group_refine")

# loop over three scripts
for i in {0..0}
do
    script_name=${scripts_name_list[$i]}
    # exit
    wandb_group_name=${wandb_group_name_list[$i]}
    # 요거 dfa인데 왜 object name이 diva360이지?
    

    #                           GPU object                      idx_from idx_to cam_idx wandb_group_name
    # bash scripts/${script_name} 1   "beagle_dog(s1)"            520      525    16      ${wandb_group_name} &
    # bash scripts/${script_name} 1   "beagle_dog(s1_24fps)"      190      215    32      ${wandb_group_name} &
    # bash scripts/${script_name} 0   "cat(walk_final)"           10       20     32      ${wandb_group_name} &
    # bash scripts/${script_name} 0   "wolf(Run)"                 20       25     16      ${wandb_group_name} &
    # bash scripts/${script_name} 4   "duck(swim)"                145      160    16      ${wandb_group_name} &
    # bash scripts/${script_name} 4   "whiteTiger(roaringwalk)"   15       25     32      ${wandb_group_name} &
    # bash scripts/${script_name} 5   "fox(walk)"                 70       75     24      ${wandb_group_name} &
    # bash scripts/${script_name} 5   "panda(walk)"               15       25     32      ${wandb_group_name} &

    # wait

    # bash scripts/${script_name} 1   "wolf(Howling)"             10       60     24      ${wandb_group_name} &
    # bash scripts/${script_name} 1   "bear(walk)"                110      140    16      ${wandb_group_name} &
    # bash scripts/${script_name} 0   "wolf(Damage)"              0        110    32      ${wandb_group_name} &
    # bash scripts/${script_name} 0   "cat(walksniff)"            70       150    32      ${wandb_group_name} &
    # bash scripts/${script_name} 4   "fox(attitude)"             95       145    24      ${wandb_group_name} &
    bash scripts/${script_name} 4   "lion(Walk)"                30       35     32      ${wandb_group_name} &
    # bash scripts/${script_name} 0   "panda(run)"                5        10     32      ${wandb_group_name} &
    # bash scripts/${script_name} 5   "lion(Run)"                 50       55     24      ${wandb_group_name} &

    wait
    # bash scripts/${script_name} 1   "cat(run)"                  25       30     32      ${wandb_group_name} &
    # bash scripts/${script_name} 1   "bear(run)"                 0        2      16      ${wandb_group_name} &
    # bash scripts/${script_name} 1   "fox(run)"                  25       30     32      ${wandb_group_name} &

    # bash scripts/${script_name} 0   "cat(walkprogressive_noz)"  25       30     32      ${wandb_group_name} &
    # bash scripts/${script_name} 0   "duck(eat_grass)"           5        15     32      ${wandb_group_name} &

    # bash scripts/${script_name} 4   "panda(acting)"             95       100    32      ${wandb_group_name} &
    # bash scripts/${script_name} 4   "wolf(Walk)"                85       95     16      ${wandb_group_name} &

    # bash scripts/${script_name} 5   "duck(walk)"                200      230    16      ${wandb_group_name} &
    # bash scripts/${script_name} 5   "whiteTiger(run)"           25       70     32      ${wandb_group_name} &

    # wait
    
    trap "exit" INT
done


