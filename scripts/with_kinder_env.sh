#!/usr/bin/env bash
#
# A thin alias for `with_env.sh` that adds the two determinism pins Tossing3D runs
# want. It is NOT a separate environment any more.
#
#     scripts/with_kinder_env.sh python -m hitl_pmp.cli --env tossing3d --method ees ...
#     scripts/with_kinder_env.sh          # no command: print the resolved environment
#
# ## Why this file shrank from ~110 lines to this
#
# It used to activate a *second* interpreter -- a `kinder-venv` virtualenv beside the
# main checkout -- and its own header explained that "the two environments cannot be
# merged" because KINDER pulls MuJoCo, PyBullet and OpenCV and `kindergarden` caps
# `requires-python` at `<3.13`.
#
# **That justification was wrong, and was measured to be wrong.** Both environments were
# already Python 3.10.20, so the `<3.13` cap excluded neither. The real constraint is a
# set of version ceilings, not an interpreter split: `pybullet_helpers 0.1.1` requires
# `numpy<2.0,>=1.23.5` and pins `scipy==1.14.0` exactly, and `moviepy` caps `pillow<12.0`.
# `hitl-pmp` runs its full gate unchanged under all three, so KINDER installs directly
# into it via the `tossing3d` extra and there is nothing left to activate.
#
# What remains here is only what is genuinely specific to a simulator *run*:
# OMP_NUM_THREADS / MKL_NUM_THREADS = 1. Those are deliberately NOT in `with_env.sh`,
# because pinning them for every command would change how non-simulator work runs. They
# match what `scripts/run_sweep.py` already sets on its children: without them a bare CLI
# run inherits the machine default (24 here) while a swept run gets 1, and multi-threaded
# float reductions reassociate -- so the same seed trains to different weights and a
# sweep and a re-run are two different experiments. Setting them here means a hand-run
# reproduction matches the sweep it is reproducing.
#
# MUJOCO_GL / PYOPENGL_PLATFORM moved *into* `with_env.sh`, because KINDER now lives in
# the default env and a plain `pytest` imports it -- see that file's EGL note for why
# getting this wrong fails silently rather than loudly.
#
# Keeping this as an alias rather than deleting it: it is referenced from docs and from
# muscle memory, the thread pins are a real and separate concern, and a script that
# quietly does the right thing beats a "no such file" for anyone who types it.

set -euo pipefail

script_source="${BASH_SOURCE[0]}"
while [ -L "$script_source" ]; do
    script_dir="$(cd -P "$(dirname "$script_source")" && pwd)"
    script_source="$(readlink "$script_source")"
    [[ "$script_source" != /* ]] && script_source="$script_dir/$script_source"
done
SCRIPT_DIR="$(cd -P "$(dirname "$script_source")" && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec "$SCRIPT_DIR/with_env.sh" "$@"
