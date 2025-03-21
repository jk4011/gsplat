#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command


# finetune drag



# train gaussian  
bash scripts/train_dfa.sh 4 cat\(walksniff\) 30 &
bash scripts/train_dfa.sh 4 bear\(walk\) 110 &
bash scripts/train_dfa.sh 4 beagle_dog\(s1_24fps\) 280 &
bash scripts/train_dfa.sh 4 duck\(walk\) 0 &
bash scripts/train_dfa.sh 4 fox\(walk\) 10 &
bash scripts/train_dfa.sh 4 duck\(swim\) 110 &
bash scripts/train_dfa.sh 4 duck\(walk\) 200 &
bash scripts/train_dfa.sh 4 duck\(eat_grass\) 50 &
bash scripts/train_dfa.sh 4 wolf\(Damage\) 0 &
bash scripts/train_dfa.sh 4 lion\(Walk\) 10 &
bash scripts/train_dfa.sh 4 cat\(run\) 30 &
bash scripts/train_dfa.sh 4 beagle_dog\(s1\) 400 &
bash scripts/train_dfa.sh 4 bear\(run\) 0 &
bash scripts/train_dfa.sh 4 wolf\(Howling\) 10 &
bash scripts/train_dfa.sh 5 lion\(Run\) 10 &
bash scripts/train_dfa.sh 5 wolf\(Walk\) 70 &
bash scripts/train_dfa.sh 5 fox\(attitude\) 60 &
bash scripts/train_dfa.sh 5 wolf\(Walk\) 20 &
bash scripts/train_dfa.sh 5 cat\(walk_final\) 10 &
bash scripts/train_dfa.sh 5 wolf\(Damage\) 10 &
bash scripts/train_dfa.sh 5 wolf\(Run\) 20 &
bash scripts/train_dfa.sh 5 beagle_dog\(s1_24fps\) 100 &
bash scripts/train_dfa.sh 5 beagle_dog\(s1\) 50 &
bash scripts/train_dfa.sh 5 wolf\(Howling\) 0 &
bash scripts/train_dfa.sh 5 wolf\(Walk\) 30 &
bash scripts/train_dfa.sh 5 cat\(walkprogressive_noz\) 220 &
bash scripts/train_dfa.sh 5 duck\(eat_grass\) 60 &
bash scripts/train_dfa.sh 5 lion\(Run\) 0 &