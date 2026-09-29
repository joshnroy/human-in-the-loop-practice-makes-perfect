#!/usr/bin/env bash
set -euo pipefail
sweep_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$sweep_repo_root/reference/kindergarden/src:$sweep_repo_root/reference/kinder-baselines/kinder-models/src:$sweep_repo_root/reference/kinder-baselines/kinder-bilevel-planning/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$sweep_repo_root/scripts/with_env.sh" "$@"
