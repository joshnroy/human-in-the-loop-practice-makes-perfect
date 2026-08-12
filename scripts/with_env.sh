#!/usr/bin/env bash
#
# Run a command inside a fully set-up hitl-pmp environment.
#
#     scripts/with_env.sh pytest
#     scripts/with_env.sh python -m scripts.run_sweep --env lightswitch ...
#     scripts/with_env.sh            # no command: print the resolved environment
#
# Why this exists rather than a documented block of `source`/`export` lines: an
# agent sandbox (the primary audience for the `hitl-env` skill) refuses `source
# ...`, refuses `VAR=x cmd` prefixes, and gives every command a *fresh* shell, so
# an `export` in one call is gone by the next. A setup block made of those three
# forms therefore cannot be executed by the people it is written for. A wrapper
# can: it is one plain command, and everything it needs to set lives inside it.
#
# Humans in an interactive shell can keep using `conda activate` directly; this
# is additive, not a replacement.
#
# Four things get set, and the third is the one that silently corrupts results
# if missed:
#   * the `hitl-pmp` conda env (`base` has mismatched dependency versions)
#   * FD_EXEC_PATH, for planning-based methods (`--method ees`) and planning/'s tests
#   * PYTHONPATH, so a *worktree* imports its own src/ and not the main checkout's
#   * MUJOCO_GL / PYOPENGL_PLATFORM = egl, because KINDER now installs into this same
#     env rather than a separate venv (see the EGL note further down)
#
# PYTHONPATH is derived from this script's own location, never `$PWD`, so the
# wrapper is correct no matter where it is invoked from. That matters here: the
# editable install's .pth file points at the main checkout's absolute path, so a
# worktree that does not set PYTHONPATH runs its own driver against a different
# checkout's library and nothing errors -- the run just measures the wrong thing.

set -euo pipefail

# Resolve this script's directory even when reached through a symlink, then take
# the repo root as its parent. Deliberately not `git rev-parse`: the wrapper must
# still work from an archive, and `$PWD` is wrong the moment a caller cd's.
script_source="${BASH_SOURCE[0]}"
while [ -L "$script_source" ]; do
    script_dir="$(cd -P "$(dirname "$script_source")" && pwd)"
    script_source="$(readlink "$script_source")"
    [[ "$script_source" != /* ]] && script_source="$script_dir/$script_source"
done
REPO_ROOT="$(cd -P "$(dirname "$script_source")/.." && pwd)"

# --- conda -----------------------------------------------------------------
# CONDA_SH may be preset to point at a non-default install; otherwise probe the
# usual locations. Failing loudly beats falling back to `base`, which is the
# documented cause of confusing version-mismatch errors.
if [ -z "${CONDA_SH:-}" ]; then
    for candidate in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$candidate" ]; then
            CONDA_SH="$candidate"
            break
        fi
    done
fi
if [ -z "${CONDA_SH:-}" ] || [ ! -f "$CONDA_SH" ]; then
    echo "with_env.sh: cannot find conda.sh; set CONDA_SH to its path" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "${HITL_PMP_CONDA_ENV:-hitl-pmp}"

# --- Fast Downward ---------------------------------------------------------
# Only planning-based methods need this, so a missing checkout is not fatal here
# -- those tests fail loudly on their own. The default follows the sibling
# convention CLAUDE.md documents (`../downward`), resolved against the *main*
# checkout rather than this one: in a worktree the literal parent directory is
# .git/worktrees/..., where no sibling clone will ever be.
if [ -z "${FD_EXEC_PATH:-}" ]; then
    git_common_dir="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [ -n "$git_common_dir" ]; then
        main_checkout="$(dirname "$git_common_dir")"
        candidate="$(dirname "$main_checkout")/downward"
        [ -d "$candidate" ] && FD_EXEC_PATH="$candidate"
    fi
fi
[ -n "${FD_EXEC_PATH:-}" ] && export FD_EXEC_PATH

# --- PYTHONPATH ------------------------------------------------------------
# Prepended, not overwritten, so an existing entry is preserved -- but this
# checkout's src/ wins, which is the whole point.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# --- rendering backend -----------------------------------------------------
# KINDER installs into THIS env now (the `tossing3d` extra), so a plain `pytest`
# imports it, and the EGL pair has to be set here rather than in a second wrapper.
#
# It is not cosmetic and it is not optional. `register_all_environments()` forces
# `osmesa` when DISPLAY is unset; under `osmesa` `import mujoco` raises, and
# `_check_deps` swallows *every* exception -- so all Dynamic3D environments are
# skipped IN SILENCE and `kinder.make("kinder/Tossing3D-o1-v0")` fails much later
# with a `NameNotFound` that names nothing relevant. That trap has cost an hour
# more than once, and unification makes it reachable from the default gate.
#
# Harmless when KINDER is not installed: nothing reads these but MuJoCo/PyOpenGL.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# With no command, report what was resolved. This doubles as the skill's sanity
# check: `import hitl_pmp` must resolve inside REPO_ROOT.
if [ "$#" -eq 0 ]; then
    echo "REPO_ROOT     $REPO_ROOT"
    echo "conda env     ${CONDA_DEFAULT_ENV:-<none>}"
    echo "python        $(command -v python)"
    echo "PYTHONPATH    $PYTHONPATH"
    echo "FD_EXEC_PATH  ${FD_EXEC_PATH:-<unset>}"
    echo "MUJOCO_GL     $MUJOCO_GL"
    python -c 'import hitl_pmp; print("hitl_pmp      " + hitl_pmp.__file__)'
    python -c 'from importlib.util import find_spec; print("kinder        " + ("installed" if find_spec("kinder") else "<not installed -- tossing3d tests will skip>"))'
    exit 0
fi

exec "$@"
