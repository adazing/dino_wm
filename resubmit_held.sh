#!/bin/bash
# The three arms parked because scontrol hold is not permitted on this cluster.
# Run this after the evalgrid sweep reports.
set -e
CK=$SCRATCH/dagger/p_bc3000/rounds/round_00/train/checkpoints/model_best.pt
S="--account=torch_pr_37_lpinto --partition=l40s_lpinto,l40s_public --qos=gpu48"
E="--export=ALL,DINO_WM_ROOT=$HOME/dino_wm_shift"
B="$HOME/dino_wm_shift/tools/torch_dagger.sbatch dagger_pusht"

sbatch $S $E --job-name=p_L_nowm     $B \
  ++init_checkpoint=$CK ++wm_finetune=false ++eval_env_override.shape=L

sbatch $S $E --job-name=p_ctrl_nowm  $B \
  ++init_checkpoint=$CK ++wm_finetune=false

sbatch $S $E --job-name=p_L_fullpred $B \
  ++init_checkpoint=$CK ++wm_finetune=true ++wm_frac_demo=0.0 \
  ++eval_env_override.shape=L ++wm_frac_failure=0.8 ++wm_total_windows=3000 \
  ++wm_finetune_last_layer_only=false
