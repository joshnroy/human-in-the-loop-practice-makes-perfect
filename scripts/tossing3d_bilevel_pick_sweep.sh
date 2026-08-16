#!/usr/bin/env bash
#
# Run tossing3d_bilevel_pick_probe.py over a seed range, N at a time.
#
#     scripts/tossing3d_bilevel_pick_sweep.sh <out-dir> <samples-per-step> <workers> <pick-mode> [first] [last]
#
# One process per seed: a KINDER rollout holds a PyBullet client and a MuJoCo model
# alive, and the refiner grounds a fresh controller per sampling attempt. The whole
# sweep runs inside one memory-capped scope so a leak hits a wall instead of the
# session's OOM policy.

set -euo pipefail

out_dir="$1"
samples_per_step="$2"
workers="$3"
pick_mode="$4"
first="${5:-100}"
last="${6:-139}"

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -P "$script_dir/.." && pwd)"

mkdir -p "$out_dir"

seq "$first" "$last" | xargs -P "$workers" -I {} \
    "$REPO_ROOT/scripts/with_env_isolated.sh" python -m scripts.tossing3d_bilevel_pick_probe \
    --seed {} \
    --samples-per-step "$samples_per_step" \
    --pick-mode "$pick_mode" \
    --output "$out_dir/seed_{}.json"
