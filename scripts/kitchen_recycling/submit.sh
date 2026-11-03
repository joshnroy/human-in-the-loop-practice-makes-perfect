#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?Usage: bash submit.sh ABSOLUTE_DEPLOYMENT_ROOT smoke|full}"
PHASE="${2:?Specify smoke or full}"
test "${ROOT:0:1}" = /
test -x "$ROOT/env/bin/python"
test -f "$ROOT/launch/${PHASE}.sbatch"
case "$PHASE" in
  smoke) test -f "$ROOT/launch/smoke-authorization.json" ;;
  full) test -f "$ROOT/launch/authorization.json" ;;
  *) exit 2 ;;
esac
mkdir -p "$ROOT/logs"
mkdir "$ROOT/launch/submission-${PHASE}.lock"
cd "$ROOT/source"
export PYTHONPATH="$ROOT/runtime-overlay:$ROOT/source/src:$ROOT/source/reference/kindergarden/src:$ROOT/source/reference/kinder-baselines/kinder-models/src:$ROOT/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
"$ROOT/env/bin/python" -c 'import sys; from pathlib import Path; from scripts.kitchen_recycling.worker import verify; verify(launch=Path(sys.argv[1]), full=sys.argv[2] == "full")' "$ROOT/launch" "$PHASE"
sbatch --parsable "$ROOT/launch/${PHASE}.sbatch" > "$ROOT/launch/submission-${PHASE}.receipt"
cat "$ROOT/launch/submission-${PHASE}.receipt"
