#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command


# finetune drag



bash scripts/finetuning_drag_dfa.sh 0 duck\(eat_grass\) 5 15 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 wolf\(Howling\) 60 170 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 wolf\(Run\) 20 25 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 whiteTiger\(roaringwalk\) 15 25 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 cat\(walksniff\) 60 75 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 beagle_dog\(s1\) 520 525 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 duck\(walk\) 120 135 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 cat\(walkprogressive_noz\) 165 210 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 0 wolf\(Run\) 35 40 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 cat\(walkprogressive_noz\) 25 30 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 fox\(run\) 25 30 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 wolf\(Howling\) 0 90 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 wolf\(Walk\) 70 80 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 beagle_dog\(s1_24fps\) 250 260 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 wolf\(Howling\) 10 60 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 beagle_dog\(s1_24fps\) 190 195 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 1 lion\(Run\) 50 55 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 bear\(run\) 5 10 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 wolf\(Damage\) 60 70 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 bear\(walk\) 125 200 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 fox\(attitude\) 65 70 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 wolf\(Damage\) 0 110 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 fox\(walk\) 70 75 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 lion\(Walk\) 30 35 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 wolf\(Damage\) 10 90 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 2 bear\(walk\) 110 140 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 beagle_dog\(s1\) 50 110 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 wolf\(Howling\) 5 65 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 duck\(swim\) 160 190 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 duck\(walk\) 0 50 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 duck\(eat_grass\) 165 295 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 duck\(eat_grass\) 0 10 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 beagle_dog\(s1_24fps\) 80 85 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 3 lion\(Run\) 30 35 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 fox\(attitude\) 90 145 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 duck\(walk\) 200 230 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 beagle_dog\(s1\) 170 175 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 duck\(swim\) 200 215 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 duck\(swim\) 205 225 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 bear\(walk\) 140 145 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 cat\(run\) 25 30 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 whiteTiger\(run\) 70 80 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 4 wolf\(Run\) 35 40 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 wolf\(Walk\) 85 95 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 wolf\(Walk\) 70 80 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 cat\(walk_final\) 10 20 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 wolf\(Run\) 30 35 24 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 duck\(eat_grass\) 50 90 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 lion\(Run\) 50 55 32 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 bear\(run\) 5 10 16 v2.0_sweep_result &
bash scripts/finetuning_drag_dfa.sh 5 wolf\(Run\) 20 25 16 v2.0_sweep_result &