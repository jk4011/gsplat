#!/bin/bash

script_name=finetuning_drag_diva360.sh
wandb_group_name=tmp

# bash scripts/run_all_multi_frame.sh finetuning_drag_diva360.sh multi_OURS
# bash scripts/run_all_multi_frame.sh finetuning_gesi_diva360.sh multi_GESI
# bash scripts/run_all_multi_frame.sh finetuning_3dgs_diva360.sh multi_3DGS

SECONDS=0
set -e        # exit when error
set -o xtrace # print command

script_name=$1
wandb_group_name=$2

bash scripts/${script_name} 0   dog             0177     0327   00      ${wandb_group_name} &
bash scripts/${script_name} 1   dog             0177     0057   00      ${wandb_group_name} &
bash scripts/${script_name} 2   dog             0177     0012   00      ${wandb_group_name} &
bash scripts/${script_name} 3   dog             0177     0379   00      ${wandb_group_name} &
bash scripts/${script_name} 4   dog             0177     0140   00      ${wandb_group_name} &

wait
bash scripts/${script_name} 0   penguin         0217     0654   00      ${wandb_group_name} &
bash scripts/${script_name} 1   penguin         0217     0114   00      ${wandb_group_name} &
bash scripts/${script_name} 2   penguin         0217     0025   00      ${wandb_group_name} &
bash scripts/${script_name} 3   penguin         0217     0281   00      ${wandb_group_name} &
bash scripts/${script_name} 4   penguin         0217     0250   00      ${wandb_group_name} &


wait

bash scripts/${script_name} 0   wall_e          0222     0327   00      ${wandb_group_name} &
bash scripts/${script_name} 1   wall_e          0222     0057   00      ${wandb_group_name} &
bash scripts/${script_name} 2   wall_e          0222     0012   00      ${wandb_group_name} &
bash scripts/${script_name} 3   wall_e          0222     0379   00      ${wandb_group_name} &
bash scripts/${script_name} 4   wall_e          0222     0140   00      ${wandb_group_name} &


wait

bash scripts/${script_name} 0   wolf            0357     2619   00      ${wandb_group_name} &
bash scripts/${script_name} 1   wolf            0357     0456   00      ${wandb_group_name} &
bash scripts/${script_name} 2   wolf            0357     0102   00      ${wandb_group_name} &
bash scripts/${script_name} 3   wolf            0357     3037   00      ${wandb_group_name} &
bash scripts/${script_name} 4   wolf            0357     1126   00      ${wandb_group_name} &

wait

bash scripts/${script_name} 0   k1_hand_stand   0412     0654   00      ${wandb_group_name} &
bash scripts/${script_name} 1   k1_hand_stand   0412     0114   00      ${wandb_group_name} &
bash scripts/${script_name} 2   k1_hand_stand   0412     0025   00      ${wandb_group_name} &
bash scripts/${script_name} 3   k1_hand_stand   0412     0759   00      ${wandb_group_name} &
bash scripts/${script_name} 4   k1_hand_stand   0412     0281   00      ${wandb_group_name} &

wait

bash scripts/${script_name} 0   stirling        0000     0114   00      ${wandb_group_name} &
bash scripts/${script_name} 1   stirling        0000     0025   00      ${wandb_group_name} &
bash scripts/${script_name} 2   stirling        0000     0281   00      ${wandb_group_name} &
bash scripts/${script_name} 3   stirling        0000     0250   00      ${wandb_group_name} &
bash scripts/${script_name} 4   stirling        0000     0228   00      ${wandb_group_name} &

# Sampled indices for stirling: [0114, 0025, 0281, 0250, 0228]
wait

bash scripts/${script_name} 0   world_globe     0020     0654   00      ${wandb_group_name} &
bash scripts/${script_name} 1   world_globe     0020     0114   00      ${wandb_group_name} &
bash scripts/${script_name} 2   world_globe     0020     0025   00      ${wandb_group_name} &
bash scripts/${script_name} 3   world_globe     0020     0759   00      ${wandb_group_name} &
bash scripts/${script_name} 4   world_globe     0020     0281   00      ${wandb_group_name} &

# Sampled indices for world_globe: [0654, 0114, 0025, 0759, 0281]
wait

bash scripts/${script_name} 0   k1_push_up      0541     0114   00      ${wandb_group_name} &
bash scripts/${script_name} 1   k1_push_up      0541     0025   00      ${wandb_group_name} &
bash scripts/${script_name} 2   k1_push_up      0541     0281   00      ${wandb_group_name} &
bash scripts/${script_name} 3   k1_push_up      0541     0250   00      ${wandb_group_name} &
bash scripts/${script_name} 4   k1_push_up      0541     0228   00      ${wandb_group_name} &

# Sampled indices for k1_push_up: [0114, 0025, 0281, 0250, 0228]
wait

bash scripts/${script_name} 0   blue_car        0142     0163   00      ${wandb_group_name} &
bash scripts/${script_name} 1   blue_car        0142     0028   00      ${wandb_group_name} &
bash scripts/${script_name} 2   blue_car        0142     0006   00      ${wandb_group_name} &
bash scripts/${script_name} 3   blue_car        0142     0189   00      ${wandb_group_name} &
bash scripts/${script_name} 4   blue_car        0142     0070   00      ${wandb_group_name} &

# Sampled indices for blue_car: [0163, 0028, 0006, 0189, 0070]
wait

bash scripts/${script_name} 0   music_box       0100     5238   00      ${wandb_group_name} &
bash scripts/${script_name} 1   music_box       0100     0912   00      ${wandb_group_name} &
bash scripts/${script_name} 2   music_box       0100     0204   00      ${wandb_group_name} &
bash scripts/${script_name} 3   music_box       0100     2253   00      ${wandb_group_name} &
bash scripts/${script_name} 4   music_box       0100     2006   00      ${wandb_group_name} &

# Sampled indices for music_box: [5238, 0912, 0204, 2253, 2006]
wait

bash scripts/${script_name} 0   k1_double_punch 0270     0114   00      ${wandb_group_name} &
bash scripts/${script_name} 1   k1_double_punch 0270     0025   00      ${wandb_group_name} &
bash scripts/${script_name} 2   k1_double_punch 0270     0281   00      ${wandb_group_name} &
bash scripts/${script_name} 3   k1_double_punch 0270     0250   00      ${wandb_group_name} &
bash scripts/${script_name} 4   k1_double_punch 0270     0228   00      ${wandb_group_name} &

# Sampled indices for k1_double_punch: [0114, 0025, 0281, 0250, 0228]
wait

bash scripts/${script_name} 0   red_car         0042     0057   00      ${wandb_group_name} &
bash scripts/${script_name} 1   red_car         0042     0012   00      ${wandb_group_name} &
bash scripts/${script_name} 2   red_car         0042     0140   00      ${wandb_group_name} &
bash scripts/${script_name} 3   red_car         0042     0125   00      ${wandb_group_name} &
bash scripts/${script_name} 4   red_car         0042     0114   00      ${wandb_group_name} &

# Sampled indices for red_car: [0057, 0012, 0140, 0125, 0114]
wait

bash scripts/${script_name} 0   trex            0135     0327   00      ${wandb_group_name} &
bash scripts/${script_name} 1   trex            0135     0057   00      ${wandb_group_name} &
bash scripts/${script_name} 2   trex            0135     0012   00      ${wandb_group_name} &
bash scripts/${script_name} 3   trex            0135     0379   00      ${wandb_group_name} &
bash scripts/${script_name} 4   trex            0135     0140   00      ${wandb_group_name} &

# Sampled indices for trex: [0327, 0057, 0012, 0379, 0140]
wait

bash scripts/${script_name} 0   bunny           0000     1309   00      ${wandb_group_name} &
bash scripts/${script_name} 1   bunny           0000     0228   00      ${wandb_group_name} &
bash scripts/${script_name} 2   bunny           0000     0051   00      ${wandb_group_name} &
bash scripts/${script_name} 3   bunny           0000     1518   00      ${wandb_group_name} &
bash scripts/${script_name} 4   bunny           0000     0563   00      ${wandb_group_name} &

# Sampled indices for bunny: [1309, 0228, 0051, 1518, 0563]
wait

bash scripts/${script_name} 0   tornado         0000     0327   00      ${wandb_group_name} &
bash scripts/${script_name} 1   tornado         0000     0057   00      ${wandb_group_name} &
bash scripts/${script_name} 2   tornado         0000     0012   00      ${wandb_group_name} &
bash scripts/${script_name} 3   tornado         0000     0379   00      ${wandb_group_name} &
bash scripts/${script_name} 4   tornado         0000     0140   00      ${wandb_group_name} &

# Sampled indices for tornado: [0327, 0057, 0012, 0379, 0140]
wait

bash scripts/${script_name} 0   truck           0078     0163   00      ${wandb_group_name} &
bash scripts/${script_name} 1   truck           0078     0028   00      ${wandb_group_name} &
bash scripts/${script_name} 2   truck           0078     0006   00      ${wandb_group_name} &
bash scripts/${script_name} 3   truck           0078     0189   00      ${wandb_group_name} &
bash scripts/${script_name} 4   truck           0078     0070   00      ${wandb_group_name} &

# Sampled indices for truck: [0163, 0028, 0006, 0189, 0070]
wait

bash scripts/${script_name} 0   clock           0000     1309   00      ${wandb_group_name} &
bash scripts/${script_name} 1   clock           0000     0228   00      ${wandb_group_name} &
bash scripts/${script_name} 2   clock           0000     0051   00      ${wandb_group_name} &
bash scripts/${script_name} 3   clock           0000     1518   00      ${wandb_group_name} &
bash scripts/${script_name} 4   clock           0000     0563   00      ${wandb_group_name} &

# Sampled indices for clock: [1309, 0228, 0051, 1518, 0563]
wait

bash scripts/${script_name} 0   horse           0120     5238   00      ${wandb_group_name} &
bash scripts/${script_name} 1   horse           0120     0912   00      ${wandb_group_name} &
bash scripts/${script_name} 2   horse           0120     0204   00      ${wandb_group_name} &
bash scripts/${script_name} 3   horse           0120     2253   00      ${wandb_group_name} &
bash scripts/${script_name} 4   horse           0120     2006   00      ${wandb_group_name} &

# Sampled indices for horse: [5238, 0912, 0204, 2253, 2006]
wait

bash scripts/${script_name} 0   hour_glass      0100     1309   00      ${wandb_group_name} &
bash scripts/${script_name} 1   hour_glass      0100     0228   00      ${wandb_group_name} &
bash scripts/${script_name} 2   hour_glass      0100     0051   00      ${wandb_group_name} &
bash scripts/${script_name} 3   hour_glass      0100     1518   00      ${wandb_group_name} &
bash scripts/${script_name} 4   hour_glass      0100     0563   00      ${wandb_group_name} &

# Sampled indices for hour_glass: [1309, 0228, 0051, 1518, 0563]
wait



