#!/bin/bash
#SBATCH --account=tsilver
#SBATCH --partition=cpu
#SBATCH --constraint=genoa
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/TSILVER/jr2860/experiments/fragile-tossing-mat1-heavy-seed0-20261008/expectimax-h100-d10/logs/%x-%A_%a.out
set -euo pipefail
STEP_RUN_ROOT=/scratch/gpfs/TSILVER/jr2860/experiments/fragile-tossing-mat1-heavy-seed0-20261008/expectimax-h100-d10
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV=/scratch/gpfs/TSILVER/jr2860/envs/hitl-pmp
export FD_EXEC_PATH=/scratch/gpfs/TSILVER/jr2860/downward
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/xdg-$SLURM_JOB_ID-0"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$SLURM_JOB_ID-0"
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_RUNTIME_DIR" "$MPLCONFIGDIR"
cd "$STEP_RUN_ROOT/source"
scripts/with_step_env.sh python -c 'from pathlib import Path; import kinder, kinder_models, hitl_pmp; root=Path.cwd(); modules=(kinder,kinder_models,hitl_pmp); assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules); print({m.__name__: m.__file__ for m in modules}, flush=True)'
exec scripts/with_step_env.sh python "$STEP_RUN_ROOT/launch_seed.py" "$1"
