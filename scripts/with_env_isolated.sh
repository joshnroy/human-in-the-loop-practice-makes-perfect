#!/usr/bin/env bash
#
# with_env.sh, plus this worktree's OWN reference/ checkouts on PYTHONPATH.
#
# The editable installs for `kinder` and `kinder_models` are .pth entries pointing
# at the MAIN checkout's reference/ trees, so a worktree that only sets
# PYTHONPATH=<worktree>/src still imports another checkout's simulator. PYTHONPATH
# is scanned before site-packages .pth paths, so prepending the worktree's own
# reference/ src dirs is what actually pins the experiment's dependencies.
#
#     scripts/with_env_isolated.sh python -c 'import kinder; print(kinder.__file__)'

set -euo pipefail

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -P "$script_dir/.." && pwd)"

export PYTHONPATH="$REPO_ROOT/reference/kindergarden/src:$REPO_ROOT/reference/kinder-baselines/kinder-models/src:$REPO_ROOT/reference/kinder-baselines/kinder-bilevel-planning/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$REPO_ROOT/scripts/with_env.sh" "$@"
