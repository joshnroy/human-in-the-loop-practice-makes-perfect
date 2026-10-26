#!/bin/bash
# One SweepOffShelf3D run per array task, group ees:
#   sbatch array-ees.sbatch manifest-ees.tsv
# Written by `python -m scripts.sweepoffshelf3d_launch della`; never edit by hand.
# Manifest line i = <name>\t<sha>\t<arm>\t<seed>\t<output dir>\t<hitl_pmp.cli argv>
#SBATCH --account=tsilver
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=16:00:00
#SBATCH --array=1-12
#SBATCH --job-name=t3-ees-human-scored-relaunch-20261025-ees
#SBATCH --output=/scratch/gpfs/TSILVER/jr2860/slurm-logs/%x-%A_%a.out
set -eo pipefail
S=/scratch/gpfs/TSILVER/jr2860
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
IFS=$'\t' read -r NAME SHA ARM SEED OUT ARGV <<<"$LINE"
[ -n "$ARGV" ] || { echo "no manifest line ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
CODE="/scratch/gpfs/TSILVER/jr2860/code/$SHA"
[ -d "$CODE" ] || { echo "no code worktree $CODE" >&2; exit 2; }
echo "task=$SLURM_ARRAY_TASK_ID name=$NAME arm=$ARM seed=$SEED node=$(hostname) start=$(date -Is)"
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV="$S/envs/hitl-pmp"
export FD_EXEC_PATH="$S/downward"
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/xdg-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
"$CODE/scripts/with_env.sh" python -c "import zstandard" || {
  echo "preflight: zstandard does not import in $HITL_PMP_CONDA_ENV; run: pip install zstandard" >&2
  exit 3
}
"$CODE/scripts/with_env.sh" python -c \
  "from scripts.sweepoffshelf3d_launch import ShelfLaunch; ShelfLaunch.require_isolated()"
eval "ARGS=($ARGV)"
mkdir -p "$OUT"
set +e
"$CODE/scripts/with_env.sh" python "$CODE/scripts/scratch_watchdog.py" \
  --path /scratch/gpfs/TSILVER --output-dir "$OUT" \
  -- "$CODE/scripts/with_env.sh" python -m hitl_pmp.cli "${ARGS[@]}"
rc=$?
echo "end=$(date -Is) rc=$rc"
exit $rc
