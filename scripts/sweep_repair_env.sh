#!/usr/bin/env bash
set -euo pipefail
repair_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repair_root/reference/kindergarden/src:$repair_root/reference/kinder-baselines/kinder-models/src:$repair_root/reference/kinder-baselines/kinder-bilevel-planning/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$repair_root/scripts/with_env.sh" "$@"
