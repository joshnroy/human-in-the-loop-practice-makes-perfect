#!/usr/bin/env bash
#SBATCH --account=tsilver
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --constraint=genoa
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --job-name=kitchen-v13b-baselines-costblind
#SBATCH --output=/scratch/gpfs/TSILVER/jr2860/kitchen-recycling/kitchen-recycling-20261108-v13-nostandoff/logs/%x-%A_%a.out
# v13b: EES and DP x seeds 0-4 x all 6 reset costs, rerun with the baselines blind to
# the human price (#775 + #776 cherry-picked onto kitchen-v13-launch; LAUNCH_COMMIT).
# Same env and manifest argv as v13's own rows; only src differs (4 files).
set -euo pipefail
ROOT=/scratch/gpfs/TSILVER/jr2860/kitchen-recycling/kitchen-recycling-20261108-v13-nostandoff
I=${SLURM_ARRAY_TASK_ID}
SEED=$((I / 12)); REM=$((I % 12)); CI=$((REM / 2)); ARMS=(0 3); ARM=${ARMS[$((REM % 2))]}
ROW=$((24 * SEED + 4 * CI + ARM))
cd "$ROOT/source"
export PYTHONPATH="$ROOT/runtime-overlay:$ROOT/baselines-fix/src:$ROOT/source/reference/kindergarden/src:$ROOT/source/reference/kinder-baselines/kinder-models/src:$ROOT/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export FD_EXEC_PATH="$ROOT/source/downward"
export LD_LIBRARY_PATH="$ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/env/lib/libstdc++.so.6"
export TMPDIR="$ROOT/tmp/${SLURM_ARRAY_JOB_ID}-v13b-${I}"
mkdir -p "$TMPDIR"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export XDG_RUNTIME_DIR="$TMPDIR/xdg"
mkdir -m 700 "$XDG_RUNTIME_DIR"
set +u
module load gurobi/13.0.0
set -u
export PYTHONPATH="$PYTHONPATH:/scratch/gpfs/TSILVER/jr2860/launch/dp-gurobi-preflight-20261025/python-deps"
"$ROOT/env/bin/python" -c "import hitl_pmp, sys; p=hitl_pmp.__file__; print(p); sys.exit(0 if p.startswith('$ROOT/baselines-fix/src') else 1)"
REL=$("$ROOT/env/bin/python" -c "import json; r=json.load(open('$ROOT/launch/manifest.json'))['runs'][$ROW]; assert r['arm'] in ('ees','dp') and r['seed']==$SEED, r; print(r['output'])")
OUT="$ROOT/results-v13b/$REL"
mkdir -p "$OUT"
echo "task $I -> manifest row $ROW -> $OUT"
mapfile -t ARGV < <("$ROOT/env/bin/python" -c "import json; [print(a) for a in json.load(open('$ROOT/launch/manifest.json'))['runs'][$ROW]['argv']]")
"$ROOT/env/bin/python" -m hitl_pmp.cli "${ARGV[@]}" --output-dir "$OUT" > "$OUT/worker.log" 2>&1
"$ROOT/env/bin/python" -c "import json; s=json.load(open('$OUT/status.json')); print(s['phase'], s['practice_steps']); assert s['phase']=='complete'"
