#!/usr/bin/env bash
# Paper seed 3 (2026-11-10): identical to launch-paper-mlp-seed2.sh except SEED and the
# output folder's date. Curtis planner, MLP sampler, fixed 1.40 m stand, box on the rug at
# October's spot (0.222, 1.346), hitl launch/real-20261109e @ 0fd3d8b4b, robot code ce25bae
# on the October environment (skill server restarted 09:01 Nov 10 with the same command, the one seed 2 used), 10-task
# evaluations before practice and after every cycle, y/s before each evaluation.
# Seed 0 went 1/9 -> 8/9 and seed 1 2/10 -> 9/10 after one cycle; the code is deliberately unchanged so the seeds
# compare (a never-thrown task is still scored a miss by the run: exclude it in GROUND-TRUTH.md).
#
#   BUDGET=100000 ./launch-paper-mlp-seed3.sh real                    # the robot (asks one y first)
#   BUDGET=100000 ./launch-paper-mlp-seed3.sh real --resume <outdir>  # resume after a stop
#   BUDGET=1500 OUT=/some/dir ./launch-paper-mlp-seed3.sh fake        # dry run, no robot
set -euo pipefail
HERE=/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/final-20261105
ROBOT=${1:-real}
shift || true
ARM=curtis
RESET_COST=5  # fixed part of every timed human reset, x human weight: a reset costs 10*(5 + 0.1*seconds) = 50 + seconds
HUMAN_WEIGHT=10
SEED=3  # paper seed 3 (2026-11-10), same setup as seeds 0, 1 and 2: Curtis planner, MLP sampler, fixed 1.40 m, box on the rug at October's spot (0.222, 1.346), robot code ce25bae on the October environment, y/s before each evaluation
RATE=0.1
# 3e-4 rescaled by the robot skills' wall-clock / tick price (~2.5x), so a practice
# cycle needs the same expected gain as in the sim (0.62% vs 0.63%).
LAMBDA=1.2e-4
OUT=${OUT:-$HERE/run-$ROBOT-$ARM-w$HUMAN_WEIGHT-seed$SEED-paper-mlp-140-rug-octenv-20261110}
COSTS=$HERE/costs/wallclock-w$HUMAN_WEIGHT-paper-mlp-seed$SEED.json  # own file, same content as seed 8's: a dry run must not rewrite the live run's
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
ROBOT_ARGS=(--robot "$ROBOT")
if [ "$ROBOT" = real ]; then
  read -r -p "Start the robot run ($ARM, wall-clock costs, human x$HUMAN_WEIGHT, lambda $LAMBDA, seed $SEED; it drives home first)? [y/n] " answer
  [ "$answer" = y ] || { echo "not started"; exit 1; }
  ROBOT_ARGS+=(--robot-no-confirm-home --robot-host 192.168.0.76 --robot-port 50002
    --robot-authkey-file /home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/.rpc_authkey
    --nuc-logs-path prpl-tidybot-ce25bae-octenv/logs)
fi
# Webcam (tidybot-real/webcam/server.py): record this run into $OUT/webcam/ until exit.
webcam() { curl -s -m 3 -o /dev/null -X POST "http://127.0.0.1:8766/record/$1" \
  --data-urlencode "dir=$OUT" --data-urlencode "pid=$$" || echo "webcam: $1 failed" >&2; }
if [ "$ROBOT" = real ]; then  # never from a dry run: it would steal a live run's recording
  webcam start
  trap 'webcam stop' EXIT
fi
cd /home/josh/Documents/repos/research/hitl-pmp-results/agent-wt/launch-20261110a
scripts/with_env.sh python -m hitl_pmp.cli \
  --env tossing3d_real --method pomdp --preset tossing3d-canonical \
  --practice-cost-config "$COSTS" "${PLANNER[@]}" \
  --practice-step-budget "${BUDGET:?set BUDGET (practice steps; a skill uses its seconds x 10)}" --num-test-tasks 10 --measure-every-cycles 1 --evaluation-control-steps 2000 \
  --wall-clock-cost-per-second "$RATE" --pomdp-linear-cost-lambda "$LAMBDA" \
  --seed "$SEED" --output-dir "$OUT" "${ROBOT_ARGS[@]}" "$@" 2>&1 | tee -a "$OUT.out"
