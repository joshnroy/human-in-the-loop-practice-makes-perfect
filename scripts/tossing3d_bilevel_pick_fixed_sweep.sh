#!/usr/bin/env bash
#
# Run tossing3d_bilevel_pick_fixed_probe.py over a seed range at one fixed standoff.
#
#     scripts/tossing3d_bilevel_pick_fixed_sweep.sh <out-dir> <spt> <workers> <distance> <rot> [first] [last]

set -euo pipefail

out_dir="$1"
samples_per_step="$2"
workers="$3"
distance="$4"
rot="$5"
first="${6:-100}"
last="${7:-139}"

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -P "$script_dir/.." && pwd)"

mkdir -p "$out_dir"

seq "$first" "$last" | xargs -P "$workers" -I {} \
    "$REPO_ROOT/scripts/with_env_isolated.sh" python -m scripts.tossing3d_bilevel_pick_fixed_probe \
    --seed {} \
    --samples-per-step "$samples_per_step" \
    --distance "$distance" \
    --rot "$rot" \
    --output "$out_dir/seed_{}.json"
