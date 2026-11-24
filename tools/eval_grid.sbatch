#!/bin/bash
#SBATCH --job-name=evalgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
RUN_DIR=$(cut -f1 <<<"$LINE")
SHIFT=$(cut -f2 <<<"$LINE")

module purge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dino_wm}"
cd "$HOME/dino_wm_shift"

export DINO_WM_RUNS="${DINO_WM_RUNS:-$SCRATCH/dagger}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_home}"
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"

echo "=== $(date) on $(hostname)"
echo "=== arm:   $RUN_DIR"
echo "=== shift: $SHIFT"

python tools/eval_checkpoints.py +run_dir="$RUN_DIR" ++wandb_logging=false $SHIFT
