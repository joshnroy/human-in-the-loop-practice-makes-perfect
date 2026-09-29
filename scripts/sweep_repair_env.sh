#!/usr/bin/env bash
set -euo pipefail
repair_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repair_root/reference/kindergarden:$repair_root/reference/kinder-baselines/kinder-models:$repair_root/reference/kinder-baselines/kinder-bilevel-planning${PYTHONPATH:+:$PYTHONPATH}"
exec "$repair_root/scripts/with_env.sh" "$@"
