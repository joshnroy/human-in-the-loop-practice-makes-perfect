#!/bin/bash
#SBATCH --account=tsilver
#SBATCH --partition=cpu
#SBATCH --constraint=genoa
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/TSILVER/jr2860/experiments/step-reset-kappa-20261007/logs/%x-%j.out
set -euo pipefail
STEP_RUN_ROOT=/scratch/gpfs/TSILVER/jr2860/experiments/step-reset-kappa-20261007
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV=/scratch/gpfs/TSILVER/jr2860/envs/hitl-pmp
export FD_EXEC_PATH=/scratch/gpfs/TSILVER/jr2860/downward
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/xdg-$SLURM_JOB_ID"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_RUNTIME_DIR" "$MPLCONFIGDIR"
cd "$STEP_RUN_ROOT/source"
scripts/with_step_env.sh python -c 'from pathlib import Path; import kinder, kinder_models, hitl_pmp; root=Path.cwd(); modules=(kinder,kinder_models,hitl_pmp); assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules); print({m.__name__: m.__file__ for m in modules}, flush=True)'
scripts/with_step_env.sh pytest -q tests/test_step_protocol.py tests/methods/belief_space/test_tossing3d_reset_planning.py tests/methods/belief_space/test_tossing3d_model.py
exec scripts/with_step_env.sh python "$STEP_RUN_ROOT/run_della_arm.py" "$1"
