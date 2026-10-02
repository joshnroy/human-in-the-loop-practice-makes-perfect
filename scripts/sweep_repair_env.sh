#!/usr/bin/env bash
set -euo pipefail
repair_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repair_root/scripts/with_sweep_env.sh" "$@"
