#!/bin/bash
# bash scripts/run_all_dfa.sh finetuning_drag_dfa.sh orient
# bash scripts/run_all_dfa.sh make_motion_dfa.sh tmp
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

script_name=$1
wandb_group_name=$2

#                           GPU object                      idx_from idx_to cam_idx wandb_group_name
bash scripts/${script_name} 4   "beagle_dog(s1)"            520      525    16      ${wandb_group_name} &&
bash scripts/${script_name} 4   "beagle_dog(s1_24fps)"      190      215    32      ${wandb_group_name} &&
bash scripts/${script_name} 4   "wolf(Howling)"             10       60     24      ${wandb_group_name} &&
bash scripts/${script_name} 4   "bear(walk)"                110      140    16      ${wandb_group_name} &&
bash scripts/${script_name} 4   "cat(run)"                  25       30     32      ${wandb_group_name} &

bash scripts/${script_name} 0   "cat(walk_final)"           10       20     32      ${wandb_group_name} &&
bash scripts/${script_name} 0   "wolf(Run)"                 20       25     16      ${wandb_group_name} &&
bash scripts/${script_name} 0   "cat(walkprogressive_noz)"  25       30     32      ${wandb_group_name} &&
bash scripts/${script_name} 0   "duck(eat_grass)"           5        15     32      ${wandb_group_name} &

bash scripts/${script_name} 0   "duck(swim)"                145      160    16      ${wandb_group_name} &&
bash scripts/${script_name} 0   "whiteTiger(roaringwalk)"   15       25     32      ${wandb_group_name} &&
bash scripts/${script_name} 0   "fox(attitude)"             95       145    24      ${wandb_group_name} &&
bash scripts/${script_name} 0   "wolf(Walk)"                85       95     16      ${wandb_group_name} &

bash scripts/${script_name} 1   "fox(walk)"                 70       75     24      ${wandb_group_name} &&
bash scripts/${script_name} 1   "panda(walk)"               15       25     32      ${wandb_group_name} &&
bash scripts/${script_name} 1   "lion(Walk)"                30       35     32      ${wandb_group_name} &&
bash scripts/${script_name} 1   "panda(acting)"             95       100    32      ${wandb_group_name} &

bash scripts/${script_name} 2   "panda(run)"                5        10     32      ${wandb_group_name} &&
bash scripts/${script_name} 2   "lion(Run)"                 50       55     24      ${wandb_group_name} &&
bash scripts/${script_name} 2   "duck(walk)"                200      230    16      ${wandb_group_name} &&
bash scripts/${script_name} 2   "whiteTiger(run)"           25       70     32      ${wandb_group_name} &

bash scripts/${script_name} 3   "wolf(Damage)"              0        110    32      ${wandb_group_name} &&
bash scripts/${script_name} 3   "cat(walksniff)"            70       150    32      ${wandb_group_name} &&
bash scripts/${script_name} 3   "bear(run)"                 0        2      16      ${wandb_group_name} &&
bash scripts/${script_name} 3   "fox(run)"                  25       30     32      ${wandb_group_name} &

wait