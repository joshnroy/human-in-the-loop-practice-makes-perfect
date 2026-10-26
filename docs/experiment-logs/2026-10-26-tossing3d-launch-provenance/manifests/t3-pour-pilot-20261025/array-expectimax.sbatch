#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-2
#SBATCH --job-name=t3-pour-pilot-20261025-expectimax
#SBATCH --output=/scratch/gpfs/TSILVER/jr2860/slurm-logs/%x-%A_%a.out
set -euo pipefail
CODE=/scratch/gpfs/TSILVER/jr2860/code/0fa7c204209ad2f32102cd5b40b37cd785398c6c
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV=/scratch/gpfs/TSILVER/jr2860/envs/hitl-pmp
export FD_EXEC_PATH=/scratch/gpfs/TSILVER/jr2860/downward
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export HITL_PMP_TMPDIR=/scratch/gpfs/TSILVER/jr2860/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$HITL_PMP_TMPDIR"
export TMP="$HITL_PMP_TMPDIR" TMPDIR="$HITL_PMP_TMPDIR" TEMP="$HITL_PMP_TMPDIR"
export XDG_RUNTIME_DIR="$HITL_PMP_TMPDIR/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
exec "$CODE/scripts/with_env.sh" python -m scripts.tossing3d_launch worker \
  --sha 0fa7c204209ad2f32102cd5b40b37cd785398c6c --manifest "$1" --task "$SLURM_ARRAY_TASK_ID"
