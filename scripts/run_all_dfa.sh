#!/bin/bash
# bash scripts/run_all_dfa.sh finetuning_drag_dfa.sh orient
# bash scripts/run_all_dfa.sh make_motion_dfa.sh tmp
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

script_name=$1
wandb_group_name=$2

#                           GPU object                      idx_from idx_to cam_idx wandb_group_name
# bash scripts/${script_name} 1   "beagle_dog(s1_24fps)"      190      215    32      ${wandb_group_name} &&
# bash scripts/${script_name} 0   "beagle_dog(s1)"            520      525    16      ${wandb_group_name} &&
# bash scripts/${script_name} 0   "wolf(Howling)"             10       60     24      ${wandb_group_name} &

# bash scripts/${script_name} 1   "bear(walk)"                110      140    16      ${wandb_group_name} &&
# bash scripts/${script_name} 1   "cat(run)"                  25       30     32      ${wandb_group_name} &
# bash scripts/${script_name} 1   "bear(run)"                 0        2      16      ${wandb_group_name} &&
bash scripts/${script_name} 0   "fox(run)"                  25       30     32      ${wandb_group_name} &

bash scripts/${script_name} 1   "cat(walk_final)"           10       20     32      ${wandb_group_name} &&
# bash scripts/${script_name} 2   "wolf(Run)"                 20       25     16      ${wandb_group_name} &&
bash scripts/${script_name} 2   "wolf(Damage)"              0        110    32      ${wandb_group_name} &&
# bash scripts/${script_name} 2   "cat(walksniff)"            70       150    32      ${wandb_group_name} &&
# bash scripts/${script_name} 2   "cat(walkprogressive_noz)"  25       30     32      ${wandb_group_name} &&

# bash scripts/${script_name} 4   "duck(swim)"                145      160    16      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "whiteTiger(roaringwalk)"   15       25     32      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "fox(attitude)"             95       145    24      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "lion(Walk)"                30       35     32      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "panda(acting)"             95       100    32      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "wolf(Walk)"                85       95     16      ${wandb_group_name} &

# bash scripts/${script_name} 5   "fox(walk)"                 70       75     24      ${wandb_group_name} &&
# bash scripts/${script_name} 5   "panda(walk)"               15       25     32      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "duck(eat_grass)"           5        15     32      ${wandb_group_name} &&
bash scripts/${script_name} 3   "panda(run)"                5        10     32      ${wandb_group_name} &&
# bash scripts/${script_name} 4   "lion(Run)"                 50       55     24      ${wandb_group_name} &

bash scripts/${script_name} 4   "duck(walk)"                200      230    16      ${wandb_group_name} &&
bash scripts/${script_name} 5   "whiteTiger(run)"           25       70     32      ${wandb_group_name} &



wait