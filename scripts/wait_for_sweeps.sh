#!/usr/bin/env bash
# Block until every named sweep directory holds 40 seed_*.json files.
set -euo pipefail
while true; do
    done_all=1
    for dir in "$@"; do
        count=$(find "$dir" -name 'seed_*.json' 2>/dev/null | wc -l)
        if [ "$count" -lt 40 ]; then done_all=0; fi
    done
    [ "$done_all" -eq 1 ] && break
    sleep 20
done
echo "all sweeps complete"
