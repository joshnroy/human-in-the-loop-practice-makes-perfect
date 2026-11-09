#!/usr/bin/env bash
# Real-robot Tossing3D run (`--env tossing3d_real --method pomdp`), Curtis determinized
# A* by default. Every knob is an environment variable whose default is the 2026-11-08
# trial 3 value; README.md beside this file gives the settings of each launch that day.
#
#   BUDGET=100000 launch.sh real                       # the robot (asks one y first)
#   BUDGET=100000 launch.sh real --resume <outdir>     # resume after a stop
#   BUDGET=3000 OUT=/some/dir launch.sh fake           # dry run, no robot
#
# Protocol (Josh, 2026-11-08): practice in cycles of up to 20 actions; an evaluation of
# NUM_TEST_TASKS tasks after every MEASURE_EVERY_CYCLES-th cycle. A policy snapshot is
# written after every learning update so any cycle can be evaluated later. BUDGET caps
# practice steps (a robot skill uses its measured seconds x 10); there is no fixed cycle
# count in code: Josh stops by answering n at the restore prompt after an evaluation.
# Costs: wall clock RATE per second, human resets timed and weighted HUMAN_WEIGHT x, so a
# timed reset costs HUMAN_WEIGHT * (RESET_COST + RATE * seconds).
set -euo pipefail

ROBOT=${1:-real}
shift || true

# Where run folders and cost files go, and the checkout whose code runs.
HERE=${HERE:-/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/final-20261105}
WORKTREE=${WORKTREE:-/home/josh/Documents/repos/research/hitl-pmp-results/agent-wt/launch-20261108d}

ARM=${ARM:-curtis}
RESET_COST=${RESET_COST:-5}
HUMAN_WEIGHT=${HUMAN_WEIGHT:-10}
SEED=${SEED:-2}
RATE=${RATE:-0.1}
# 3e-4 rescaled by the robot skills' wall-clock / tick price (~2.5x), so a practice
# cycle needs the same expected gain as in the sim (0.62% vs 0.63%).
LAMBDA=${LAMBDA:-1.2e-4}
NUM_TEST_TASKS=${NUM_TEST_TASKS:-20}
MEASURE_EVERY_CYCLES=${MEASURE_EVERY_CYCLES:-2}
# yes: evaluate the untrained policy before practice (step 0); no: skip it.
INITIAL_MEASUREMENT=${INITIAL_MEASUREMENT:-no}
RUN_TAG=${RUN_TAG:-F50-20261108d}
OUT=${OUT:-$HERE/run-$ROBOT-$ARM-w$HUMAN_WEIGHT-seed$SEED-$RUN_TAG}

# The robot's skill server, and the file holding its RPC authkey (never the key here).
ROBOT_HOST=${ROBOT_HOST:-192.168.0.76}
ROBOT_PORT=${ROBOT_PORT:-50002}
ROBOT_AUTHKEY_FILE=${ROBOT_AUTHKEY_FILE:-/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/.rpc_authkey}
NUC_LOGS_PATH=${NUC_LOGS_PATH:-prpl-tidybot-test/logs}
# The webcam recorder (webcam/server.py), host:port.
WEBCAM=${WEBCAM:-127.0.0.1:8766}

COSTS=$HERE/costs/wallclock-w$HUMAN_WEIGHT.json
mkdir -p "$HERE/costs"
cat > "$COSTS" <<JSON
{
  "robot_step": {"value": 0.01},
  "human_skills": {
    "reset_cube_far": {"cost": {"value": $RESET_COST}, "duration": {"value": 1}},
    "reset_cube_and_bin_near": {"cost": {"value": $RESET_COST}, "duration": {"value": 1}}
  },
  "human_weight": $HUMAN_WEIGHT,
  "objective_lambda": $LAMBDA
}
JSON
case "$ARM" in
  expectimax) PLANNER=(--pomdp-solver expectimax --pomdp-search-depth 6) ;;
  curtis) PLANNER=(--pomdp-solver determinized_astar --pomdp-determinized-variant curtis
    --pomdp-num-rollouts all --pomdp-draw-scope node --pomdp-max-search-iterations 100
    --pomdp-observation-probability-weight 0.01) ;;
  *) echo "ARM must be expectimax or curtis" >&2; exit 2 ;;
esac
case "$INITIAL_MEASUREMENT" in
  yes) MEASURE=() ;;
  no) MEASURE=(--no-initial-measurement) ;;
  *) echo "INITIAL_MEASUREMENT must be yes or no" >&2; exit 2 ;;
esac
ROBOT_ARGS=(--robot "$ROBOT")
if [ "$ROBOT" = real ]; then
  read -r -p "Start the robot run ($ARM, wall-clock costs, human x$HUMAN_WEIGHT, lambda $LAMBDA, seed $SEED; it drives home first)? [y/n] " answer
  [ "$answer" = y ] || { echo "not started"; exit 1; }
  ROBOT_ARGS+=(--robot-no-confirm-home --robot-host "$ROBOT_HOST" --robot-port "$ROBOT_PORT"
    --robot-authkey-file "$ROBOT_AUTHKEY_FILE"
    --nuc-logs-path "$NUC_LOGS_PATH")
fi
# Webcam (webcam/server.py): record this run into $OUT/webcam/ until exit.
webcam() { curl -s -m 3 -o /dev/null -X POST "http://$WEBCAM/record/$1" \
  --data-urlencode "dir=$OUT" --data-urlencode "pid=$$" || echo "webcam: $1 failed" >&2; }
webcam start
trap 'webcam stop' EXIT
cd "$WORKTREE"
scripts/with_env.sh python -m hitl_pmp.cli \
  --env tossing3d_real --method pomdp --preset tossing3d-canonical \
  --practice-cost-config "$COSTS" "${PLANNER[@]}" \
  --practice-step-budget "${BUDGET:?set BUDGET (practice steps; a skill uses its seconds x 10)}" \
  --num-test-tasks "$NUM_TEST_TASKS" --measure-every-cycles "$MEASURE_EVERY_CYCLES" "${MEASURE[@]}" \
  --wall-clock-cost-per-second "$RATE" --pomdp-linear-cost-lambda "$LAMBDA" \
  --seed "$SEED" --output-dir "$OUT" "${ROBOT_ARGS[@]}" "$@" 2>&1 | tee -a "$OUT.out"
