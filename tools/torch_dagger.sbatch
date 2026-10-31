#!/bin/bash
# Run a dino_wm DAgger loop on NYU Torch (Slurm).
#
#   sbatch --account=torch_pr_XXX_XXXXX tools/torch_dagger.sbatch dagger_wall ++wm_finetune=false
#
# First positional arg is the config name (dagger_wall | dagger_pusht); everything after it is
# passed to dagger_loop.py as Hydra overrides.
#
# RESUMING. A wall run is ~33 h and a pusht run ~40 h, which is longer than most Slurm time
# limits. The loop is resumable: rounds that finished leave a DONE marker and are skipped, so
# re-submitting with the SAME out_dir continues where it stopped. This script picks a stable
# out_dir from the job name (not a timestamp) so a resubmit lands in the same place, and
# optionally resubmits itself when it runs out of time -- see AUTO_RESUBMIT below.
#
#SBATCH --job-name=dagger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='h200|l40s'
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --requeue
#
# NOTE no --partition: NYU's docs say not to set one on Torch except for preemptible jobs.
# NOTE --account is REQUIRED on Torch and is deliberately not hardcoded here. Get yours with
#      `my_slurm_accounts` and pass it on the sbatch command line.

set -euo pipefail

CONFIG="${1:?first arg must be a config name, e.g. dagger_wall}"
shift || true

# ---- where things live on Torch. Override by exporting these before sbatch. ----------------
# HOME has a small quota on HPC -- 62 GB of datasets will NOT fit there. Only the code lives in
# $HOME; everything bulky goes to $SCRATCH, which is large but PURGED after a period of inactivity
# (NYU: ~60 days untouched). Everything under $SCRATCH here is re-copyable from the NAS.
: "${DINO_WM_ROOT:=$HOME/dino_wm}"             # the code checkout (~929 MB)
: "${DINO_WM_CKPT:=$SCRATCH/dino_wm_assets}"   # pretrained world models (~756 MB)
: "${DINO_WM_DATA:=$SCRATCH/dino_data}"        # wall_single/ and pusht_noise/ (~62 GB)
: "${DINO_WM_RUNS:=$SCRATCH/dagger}"           # run outputs (~25 GB per run)
: "${CONDA_ENV:=dino_wm}"                      # conda env name
: "${AUTO_RESUBMIT:=0}"                        # 1 = resubmit self when the time limit is near

for d in "$DINO_WM_ROOT" "$DINO_WM_CKPT" "$DINO_WM_DATA"; do
  [ -d "$d" ] || { echo "MISSING: $d -- see tools/torch_dagger.sbatch header for the layout"; exit 1; }
done

# A STABLE out_dir, so a resubmit resumes instead of starting a fresh run.
: "${RUN_NAME:=${SLURM_JOB_NAME}}"
OUT_DIR="$DINO_WM_RUNS/$RUN_NAME"

echo "=== $(date) on $(hostname)"
echo "    config   : $CONFIG"
echo "    overrides: $*"
echo "    out_dir  : $OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- environment ---------------------------------------------------------------------------
module purge
# Torch provides conda via a module on most NYU clusters; adjust if `module avail` differs.
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$DINO_WM_ROOT"

# MuJoCo needs its shared libraries on the loader path or `import mujoco_py` dies with
# "Missing path to your environment variable".
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
# Headless rendering. The compute nodes have no display; EGL uses the GPU, osmesa is the CPU
# fallback if EGL is unavailable.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export HYDRA_FULL_ERROR=1
# models/encoder/dino.py calls torch.hub.load("facebookresearch/dinov2:b48308a"), which
# downloads the repo and weights AT RUNTIME. Compute nodes have no outbound internet, so
# that fails with HTTP 500. Pre-populate this cache from a login node.
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_home}"
export WANDB_DIR="$OUT_DIR"
# Compute nodes have no direct outbound internet. wandb 0.13.1 starts a local helper that
# gives up when it cannot reach the backend, and wandb.init() then raises -- which killed
# seven jobs on 2026-10-16, in generation, before any training happened. Offline mode writes
# everything to WANDB_DIR; push it later with:  wandb sync $OUT_DIR/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OUT_DIR" "$OUT_DIR/rounds" logs

# Snapshot progress and the clock BEFORE the run, so the resubmit epilogue can tell a crash
# apart from a genuine time-out.
START_TS=$(date +%s)
DONE_AT_START=$(find "$OUT_DIR/rounds" -name DONE 2>/dev/null | wc -l)

# Snapshot progress and the clock BEFORE the run, so the resubmit epilogue can tell a crash
# apart from a genuine time-out.
START_TS=$(date +%s)
DONE_AT_START=$(find "$OUT_DIR/rounds" -name DONE 2>/dev/null | wc -l)

# ---- run -----------------------------------------------------------------------------------
# env_start_method=spawn: EGL state does not survive a fork, which is why forked env workers
# fail to render on a fresh cluster. spawn gives each worker a clean interpreter.
srun python dagger_loop.py \
  --config-name "$CONFIG" \
  "out_dir=$OUT_DIR" \
  "++ckpt_base_path=$DINO_WM_CKPT" \
  "++override_dataset_path=$DINO_WM_DATA/$( [ "$CONFIG" = dagger_pusht ] && echo pusht_noise || echo wall_single )" \
  ++env_start_method=spawn \
  "$@" && STATUS=done || STATUS=incomplete

echo "=== $(date) loop exited: $STATUS"

# ---- optional self-resubmit ----------------------------------------------------------------
# Only resubmit if the loop did NOT finish all its rounds, so a completed run stops cleanly.
if [ "$AUTO_RESUBMIT" = "1" ] && [ "$STATUS" = "incomplete" ]; then
  DONE=$(find "$OUT_DIR/rounds" -name DONE 2>/dev/null | wc -l)
  WANT="${DAGGER_ROUNDS:-10}"
  ELAPSED=$(( $(date +%s) - START_TS ))
  # A CRASH IS NOT A TIME-OUT. Resubmitting a job that died in seconds spins forever: every
  # new attempt sees the same zero finished rounds, fails the same way, and submits another.
  if [ "$DONE" -le "$DONE_AT_START" ] && [ "$ELAPSED" -lt 1200 ]; then
    echo "=== exited after ${ELAPSED}s with no new finished round: a crash, not a time-out."
    echo "=== NOT resubmitting. Read the error above before trying again."
    exit 1
  fi
  if [ "$DONE" -lt "$WANT" ]; then
    echo "=== $DONE/$WANT rounds done, resubmitting to continue"
    # Carry partition and constraint across; dropping --partition sent continuations to the
    # default partition list and straight back onto the lab cap.
    sbatch --account="${SLURM_JOB_ACCOUNT}" --job-name="$SLURM_JOB_NAME" ${SLURM_JOB_PARTITION:+--partition="$SLURM_JOB_PARTITION"} ${SLURM_JOB_CONSTRAINT:+--constraint="$SLURM_JOB_CONSTRAINT"} "$0" "$CONFIG" "$@"
  else
    echo "=== all $WANT rounds done, not resubmitting"
  fi
fi
