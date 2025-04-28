
#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

script_name=$1
wandb_group_name=$2

#                           GPU object                      idx_from idx_to cam_idx wandb_group_name
bash scripts/${script_name} 0   "beagle_dog(s1)"            520      525    16      ${wandb_group_name} &&
bash scripts/${script_name} 0   "beagle_dog(s1_24fps)"      250      260    32      ${wandb_group_name} &&
bash scripts/${script_name} 0   "wolf(Howling)"             10       60     24      ${wandb_group_name} &&
bash scripts/${script_name} 0   "bear(walk)"                110      140    16      ${wandb_group_name} &&
bash scripts/${script_name} 0   "cat(run)"                  25       30     32      ${wandb_group_name} &

bash scripts/${script_name} 1   "cat(walk_final)"           10       20     32      ${wandb_group_name} &&
bash scripts/${script_name} 1   "wolf(Run)"                 20       25     16      ${wandb_group_name} &&
bash scripts/${script_name} 1   "cat(walkprogressive_noz)"  25       30     32      ${wandb_group_name} &&
bash scripts/${script_name} 1   "duck(eat_grass)"           0        10     24      ${wandb_group_name} &

bash scripts/${script_name} 2   "duck(swim)"                145      160    16      ${wandb_group_name} &&
bash scripts/${script_name} 2   "whiteTiger(roaringwalk)"   15       25     32      ${wandb_group_name} &&
bash scripts/${script_name} 2   "fox(attitude)"             95       145    24      ${wandb_group_name} &&
bash scripts/${script_name} 2   "wolf(Walk)"                10       20     32      ${wandb_group_name} &

bash scripts/${script_name} 3   "fox(walk)"                 70       75     24      ${wandb_group_name} &&
bash scripts/${script_name} 3   "panda(walk)"               15       25     32      ${wandb_group_name} &&
bash scripts/${script_name} 3   "lion(Walk)"                30       35     32      ${wandb_group_name} &&
bash scripts/${script_name} 3   "panda(acting)"             95       100    32      ${wandb_group_name} &

bash scripts/${script_name} 4   "panda(run)"                5        10      32      ${wandb_group_name} &&
bash scripts/${script_name} 4   "lion(Run)"                 70       75     24      ${wandb_group_name} &&
bash scripts/${script_name} 4   "duck(walk)"                200      230    16      ${wandb_group_name} &&
bash scripts/${script_name} 4   "whiteTiger(run)"           70       80     32      ${wandb_group_name} &

bash scripts/${script_name} 5   "wolf(Damage)"              0        110    32      ${wandb_group_name} &&
bash scripts/${script_name} 5   "cat(walksniff)"            30       110    32      ${wandb_group_name} &&
bash scripts/${script_name} 5   "bear(run)"                 0        2      16      ${wandb_group_name} &&
bash scripts/${script_name} 5   "fox(run)"                  25       30     32      ${wandb_group_name} &

wait