#!/bin/bash
version=tmp

# bash scripts/run_all_dfa.sh finetuning_gesi_dfa.sh ${version}_deform_only
# 3DGS
bash scripts/run_all.sh finetuning_3dgs_diva360.sh tmp_3DGS

# # ours
# bash scripts/run_all.sh finetuning_drag_diva360.sh ${version}_OURS


# GESI
# version=tmp
# sed -i 's/diva360_gesi_deform_only/diva360_gesi/' /data2/wlsgur4011/GESI/gsplat/scripts/finetuning_gesi_diva360.sh
# sed -i 's/deform_only = True/deform_only = False/' /data2/wlsgur4011/GESI/gsplat/examples/gesi_trainer.py
# bash scripts/run_all_dfa.sh finetuning_gesi_dfa.sh ${version}_GESI
# bash scripts/run_all.sh finetuning_gesi_diva360.sh ${version}_GESI

# GESI (\mu, q only)
sed -i 's/diva360_gesi/diva360_gesi_deform_only/' /data2/wlsgur4011/GESI/gsplat/scripts/finetuning_gesi_diva360.sh
sed -i 's/deform_only = False/deform_only = True/' /data2/wlsgur4011/GESI/gsplat/examples/gesi_trainer.py

# bash scripts/run_all.sh finetuning_gesi_diva360.sh ${version}_deform_only

bash scripts/run_all_dfa.sh finetuning_3dgs_dfa.sh ${version}_3DGS
# bash scripts/run_all_dfa.sh finetuning_drag_dfa.sh ${version}_OURS
