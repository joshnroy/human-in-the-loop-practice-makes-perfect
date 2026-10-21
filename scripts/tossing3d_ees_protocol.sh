#!/usr/bin/env bash
#
# The standard Tossing3D EES protocol: scripts/run_sweep.py with this domain's own
# hard-won args baked in, instead of the generic CLI defaults.
#
#     scripts/with_env.sh scripts/tossing3d_ees_protocol.sh \
#         --practice-reset-policy scheduled --results-root results/my-run
#
# Why this exists: the global CLI defaults (--num-cycles 10
# --max-steps-per-interaction 150) are untuned generic-runner values, not this
# domain's. This domain's own protocol -- --num-cycles 100
# --max-steps-per-interaction 20 -- was established across two real experiments that
# both converged on the same numbers:
#   * docs/experiment-logs/2026-08-08-tossing3d-100cycle-plateau.md, which found EES
#     plateaus around cycle 12 of 100 -- fewer cycles than that would have hidden it.
#   * docs/experiment-logs/2026-08-08-tossing3d-reset-free-remeasured.md, the first
#     *correct* reset-free measurement, after #179 fixed --practice-reset-policy never
#     being a no-op on this domain.
# That protocol was then lost and silently replaced by the CLI defaults in an ad hoc
# scheduled-vs-never sweep on 2026-08-19 -- a 10-cycle run is a materially smaller,
# differently-shaped experiment than either log above, not a cheaper version of the
# same one. This wrapper exists so the numbers live in the repo, not in a doc someone
# has to remember to go re-read.
#
# --num-seeds defaults to 10, the paper's own seed count and what both logs above
# used. Pass a smaller number explicitly for a cheap pilot, knowing it will not carry
# either log's statistical power.
set -euo pipefail

NUM_CYCLES=100
MAX_STEPS_PER_INTERACTION=20
NUM_TEST_TASKS=10
NUM_SEEDS=10
RESULTS_ROOT=""
PRACTICE_RESET_POLICY=""
EXTRA_SHARED_ARGS=""
DRY_RUN=0

usage() {
    echo "usage: $(basename "$0") --practice-reset-policy <policy> --results-root <dir>" \
        "[--num-seeds N] [--extra-shared-args \"...\"] [--dry-run]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --practice-reset-policy) PRACTICE_RESET_POLICY="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --extra-shared-args) EXTRA_SHARED_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$PRACTICE_RESET_POLICY" || -z "$RESULTS_ROOT" ]]; then
    usage
    exit 1
fi

SHARED_ARGS="--num-test-tasks $NUM_TEST_TASKS --practice-reset-policy $PRACTICE_RESET_POLICY"
if [[ -n "$EXTRA_SHARED_ARGS" ]]; then
    SHARED_ARGS="$SHARED_ARGS $EXTRA_SHARED_ARGS"
fi

CMD=(python -m scripts.run_sweep
    --env tossing3d
    --methods ees
    --num-seeds "$NUM_SEEDS"
    --results-root "$RESULTS_ROOT"
    --shared-args "$SHARED_ARGS"
    --method-args "ees=--num-cycles $NUM_CYCLES --max-steps-per-interaction $MAX_STEPS_PER_INTERACTION"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
