#!/usr/bin/env bash
#
# Run a command inside a fully set-up KINDER environment -- `with_env.sh`'s twin for
# anything that touches the simulator (`--env tossing3d`).
#
#     scripts/with_kinder_env.sh python -m hitl_pmp.cli --env tossing3d --method ees ...
#     scripts/with_kinder_env.sh python -m scripts.run_sweep --env tossing3d ...
#     scripts/with_kinder_env.sh          # no command: print the resolved environment
#
# Why a SECOND wrapper rather than a flag on the first: KINDER lives in its own
# virtualenv, never in `hitl-pmp`. It pulls MuJoCo, PyBullet and OpenCV, and
# `kindergarden` caps `requires-python` at `<3.13` (see CLAUDE.md). The two
# environments cannot be merged, so the two wrappers cannot be either -- and
# `with_env.sh`'s conda activation would actively select the interpreter that has no
# `kinder` in it.
#
# Everything `with_env.sh` sets, this sets too, for the same reasons:
#   * PYTHONPATH, so a *worktree* imports its own src/ and not the main checkout's
#   * FD_EXEC_PATH, for planning-based methods (`--method ees`)
# ...plus two that only matter on this side:
#   * MUJOCO_GL / PYOPENGL_PLATFORM = egl
#   * OMP_NUM_THREADS / MKL_NUM_THREADS = 1
#
# The EGL pair is not cosmetic. `register_all_environments()` forces `osmesa` when
# DISPLAY is unset; under `osmesa` `import mujoco` raises, and `_check_deps` swallows
# *every* exception -- so all Dynamic3D environments are skipped IN SILENCE and
# `kinder.make("kinder/Tossing3D-o1-v0")` fails much later with a `NameNotFound` that
# names nothing relevant. That trap has cost an hour more than once.
#
# The thread pins match what `scripts/run_sweep.py` already sets on its children.
# Without them a bare CLI run inherits the machine default (24 here) while a swept run
# gets 1, and multi-threaded float reductions reassociate -- so the same seed trains to
# different weights and a sweep and a re-run are two different experiments. Setting it
# here means a hand-run reproduction matches the sweep it is reproducing.

set -euo pipefail

# Resolve this script's directory even when reached through a symlink, then take the
# repo root as its parent. Deliberately not `git rev-parse`: `$PWD` is wrong the moment
# a caller cd's, and the wrapper must be correct wherever it is invoked from.
script_source="${BASH_SOURCE[0]}"
while [ -L "$script_source" ]; do
    script_dir="$(cd -P "$(dirname "$script_source")" && pwd)"
    script_source="$(readlink "$script_source")"
    [[ "$script_source" != /* ]] && script_source="$script_dir/$script_source"
done
REPO_ROOT="$(cd -P "$(dirname "$script_source")/.." && pwd)"

# The main checkout, which is where sibling directories actually live. In a worktree the
# literal parent is .git/worktrees/..., where no sibling venv will ever be -- the same
# resolution with_env.sh uses for Fast Downward.
git_common_dir="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
if [ -n "$git_common_dir" ]; then
    MAIN_CHECKOUT="$(dirname "$git_common_dir")"
else
    MAIN_CHECKOUT="$REPO_ROOT"
fi

# --- the KINDER venv -------------------------------------------------------
# KINDER_VENV may be preset to point elsewhere; otherwise use CLAUDE.md's documented
# sibling location. Failing loudly beats falling back to a python without `kinder`,
# which would surface as a confusing ModuleNotFoundError deep inside a domain import.
KINDER_VENV="${KINDER_VENV:-$(dirname "$MAIN_CHECKOUT")/kinder-venv}"
KINDER_PYTHON="$KINDER_VENV/bin/python"
if [ ! -x "$KINDER_PYTHON" ]; then
    echo "with_kinder_env.sh: no KINDER venv at $KINDER_VENV" >&2
    echo "  create it with:" >&2
    echo "    python3.10 -m venv $KINDER_VENV" >&2
    echo "    $KINDER_VENV/bin/pip install -e reference/kindergarden \\" >&2
    echo "        -e reference/kinder-baselines/kinder-models" >&2
    echo "  or set KINDER_VENV to an existing one." >&2
    exit 1
fi

# --- Fast Downward ---------------------------------------------------------
# Only planning-based methods need this, so a missing checkout is not fatal here --
# those runs fail loudly on their own.
if [ -z "${FD_EXEC_PATH:-}" ]; then
    candidate="$(dirname "$MAIN_CHECKOUT")/downward"
    [ -d "$candidate" ] && FD_EXEC_PATH="$candidate"
fi
[ -n "${FD_EXEC_PATH:-}" ] && export FD_EXEC_PATH

# --- PYTHONPATH ------------------------------------------------------------
# Prepended, not overwritten, so an existing entry is preserved -- but this checkout's
# src/ wins, which is the whole point.
#
# The two `reference/` source roots are on here for the same reason `src/` is, and the
# omission was a real hole: the KINDER venv installs `kindergarden` and `kinder-models`
# **editable**, so their `.pth` files carry the *main checkout's* absolute paths. From a
# worktree, `import kinder_models` therefore resolved to the main checkout's submodule at
# whatever commit it happened to be sitting on -- not at this branch's pin -- and nothing
# errored, because both trees export the same module names. That is the same class of
# silent skew `src/` was already guarded against, one directory over.
#
# It bit for real on 2026-08-12: the main checkout was on `3524010` while this branch
# pinned `1b564a1`, and the difference is exactly the release-speed parameter the toss
# stack is built on (`grep -c release_speed`: 6 in one tree, **0** in the other). A speed
# sweep that picked up the main checkout's copy would have silently measured the
# unparameterised toss at a single speed and looked entirely normal doing it.
#
# PYTHONPATH is searched before site-packages, so this wins over the `.pth` entries. In
# the main checkout the two paths name the same trees the venv would have resolved
# anyway, so this is a no-op there rather than a behaviour change.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/reference/kindergarden/src:$REPO_ROOT/reference/kinder-baselines/kinder-models/src${PYTHONPATH:+:$PYTHONPATH}"

# --- rendering and math threads --------------------------------------------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# With no command, report what was resolved. This doubles as the sanity check an agent
# is told to run before trusting any number: `hitl_pmp` must resolve inside REPO_ROOT,
# and `kinder` must resolve inside reference/.
if [ "$#" -eq 0 ]; then
    echo "REPO_ROOT     $REPO_ROOT"
    echo "python        $KINDER_PYTHON"
    echo "PYTHONPATH    $PYTHONPATH"
    echo "FD_EXEC_PATH  ${FD_EXEC_PATH:-<unset>}"
    echo "MUJOCO_GL     $MUJOCO_GL"
    "$KINDER_PYTHON" -c 'import hitl_pmp; print("hitl_pmp      " + hitl_pmp.__file__)'
    "$KINDER_PYTHON" -c 'import kinder; print("kinder        " + kinder.__file__)'
    # `kinder_models` is reported separately rather than assumed to follow `kinder`: they
    # are two independent submodules with two independent editable pointers, and it is
    # the one whose pin moves most often.
    "$KINDER_PYTHON" -c 'import kinder_models; print("kinder_models " + kinder_models.__file__)'
    exit 0
fi

# `python` as the first argument means "the venv's python", so callers can write the
# same command shape `with_env.sh` takes rather than knowing the interpreter's path.
if [ "$1" = "python" ]; then
    shift
    exec "$KINDER_PYTHON" "$@"
fi

exec "$@"
