#!/usr/bin/env bash
set -euo pipefail
STEP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$STEP_REPO_ROOT/src:$STEP_REPO_ROOT/reference/kindergarden/src:$STEP_REPO_ROOT/reference/kinder-baselines/kinder-models/src${PYTHONPATH:+:$PYTHONPATH}"
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
exec "$STEP_REPO_ROOT/scripts/with_env.sh" "$@"
