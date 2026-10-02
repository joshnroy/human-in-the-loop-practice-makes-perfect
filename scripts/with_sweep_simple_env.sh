#!/usr/bin/env bash
set -euo pipefail
simple_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$simple_root/reference/kindergarden/src:$simple_root/reference/kinder-baselines/kinder-models/src:$simple_root/reference/kinder-baselines/kinder-bilevel-planning/src"
exec "$simple_root/scripts/with_env.sh" "$@"
