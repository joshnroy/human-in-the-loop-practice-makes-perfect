#!/usr/bin/env bash
# FALLBACK, 2026-11-09: the BLUE CUBE into the same cardboard box from the same 1.40 m
# stand, seed 9. Josh's fallback if trial 9 (the bottle, seed 8, launch-turned140-seed8.sh)
# shows no learning. Everything but the object, the seed and OUT is seed 8's launcher
# unchanged (same hitl launch branch launch-20261109d @ d10ba3786, same planner, costs,
# lambda, 10-task evaluations, 2000-step evaluation budget, one throw per task, the old
# range 100-140 deg/s x 500-850 ms), so the two runs compare directly.
#   - The hitl side has nothing object-specific: the object is chosen on the robot side
#     only. Restart the skill server WITHOUT +object=bottle and without the two
#     bottle-only target_source overrides; keep every bin.* and dump_bin override:
#       python scripts/skill_server.py mode=real bin.length=0.39 bin.width=0.29 \
#         bin.height=0.21 bin.wall=0.004 bin.base_length=0.39 bin.base_width=0.29 \
#         bin.hue_range=null bin.min_saturation=140 "bin.search_bounds=[-0.6,0.8,-1.4,1.6]" \
#         dump_bin.held_max=0.985
#     (robot fork a1e9650; the cube is the default object there, found by
#     prpl_tidybot.cube_detector, which has never run live: run cube_check.py first.)
#   - The stand stays FIXED at 1.40 m (launch branch d) so this compares with seed 8.
#     October's cube run 10 (2026-10-21, 1/5 -> 5/5 into this box) learned the standoff
#     over 1.37-1.45 m instead; not changed here on purpose.
#   - Differs from the seed 8 launcher in one more way: the webcam is recorded only for
#     `real`. A `fake` dry run used to POST /record/start too, which re-points the webcam
#     server's single run recording at the dry run's folder (and stops it when the dry
#     run exits), i.e. it would steal a live run's webcam recording. And COSTS is its own
#     file (costs/wallclock-w10-cube-seed9.json, byte-identical to seed 8's), so a dry
#     run never rewrites the file a live run was started from.
#   - Dry run 2026-11-09 13:41 ET: OUT=../cube-fallback-dryrun-20261109/run BUDGET=1500
#     fake -> exit 0, no traceback, 2 evaluations, phase complete, git d10ba3786.
#
# Switch: stop seed 8 (n at its prompt), restart the skill server as above, run
# cube_check.py (watched), then: BUDGET=100000 ./launch-cube-140-seed9.sh real
#
# The seed 8 header follows, kept as it was:
# Real-robot run, Curtis determinized A* (2026-11-09 trial 9: the turned cardboard box (0.29 m along the throw) at a 1.40 m stand, the bottle's edge of reach, far spot (0.245, 1.27); seed 8, old range 100-140 x 500-850; step-0 evaluation then an evaluation after every practice cycle (Josh: one cycle first, ~1 h), 10 tasks each, 2000-step evaluation budget (3 pick retries) and one throw per task). hitl: launch-20261109d = launch-20261109c + the 1.40 m and one-throw commit. Robot fork a1e9650, skill server bin.length=0.39 bin.width=0.29.
#
#   BUDGET=100000 launch.sh real                       # the robot (asks one y first)
#   BUDGET=100000 launch.sh real --resume <outdir>     # resume after a stop
#   BUDGET=3000 OUT=/some/dir launch.sh fake           # dry run, no robot
#
# Protocol (Josh, 2026-11-08): practice in cycles of up to 20 actions; a 20-task
# evaluation after every 2nd cycle (after cycles 2 and 4 for the planned 4); no step-0
# evaluation: the untrained policy draws throws uniformly whatever its inputs, so the
# step-0 point is the morning run's (9/20, same robot code dc9679f). A policy snapshot is
# written after every learning update so any cycle can be evaluated later. Budget cap
# 100k practice steps (a robot skill uses its measured seconds x 10); no fixed cycle
# count in code: Josh stops by answering n at the restore prompt after an evaluation.
# Costs: wall clock 0.1/s, human resets timed and weighted 10x, lambda 1.2e-4.
# hitl code: launch branch launch/real-20261108d (main eb857ef12 + #768 #770 #771 #773 #774; #773: a
# timed human reset costs 10*(RESET_COST + 0.1*seconds)). Previously: #768 standoff is not
# a toss classifier input). Robot code: see RUN-NOTES.md in the output folder.
set -euo pipefail
HERE=/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/final-20261105
ROBOT=${1:-real}
shift || true
ARM=curtis
RESET_COST=5  # fixed part of every timed human reset, x human weight: a reset costs 10*(5 + 0.1*seconds) = 50 + seconds
HUMAN_WEIGHT=10
SEED=0  # paper seed 0 (2026-11-09 night): Curtis planner, MLP sampler, fixed 1.40 m, box on the rug at October's spot (0.222, 1.346), robot code ce25bae on the October environment, y/s before each evaluation
RATE=0.1
# 3e-4 rescaled by the robot skills' wall-clock / tick price (~2.5x), so a practice
# cycle needs the same expected gain as in the sim (0.62% vs 0.63%).
LAMBDA=1.2e-4
OUT=${OUT:-$HERE/run-$ROBOT-$ARM-w$HUMAN_WEIGHT-seed$SEED-paper-mlp-140-rug-octenv-20261109}
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
cd /home/josh/Documents/repos/research/hitl-pmp-results/agent-wt/launch-20261109e
scripts/with_env.sh python -m hitl_pmp.cli \
  --env tossing3d_real --method pomdp --preset tossing3d-canonical \
  --practice-cost-config "$COSTS" "${PLANNER[@]}" \
  --practice-step-budget "${BUDGET:?set BUDGET (practice steps; a skill uses its seconds x 10)}" --num-test-tasks 10 --measure-every-cycles 1 --evaluation-control-steps 2000 \
  --wall-clock-cost-per-second "$RATE" --pomdp-linear-cost-lambda "$LAMBDA" \
  --seed "$SEED" --output-dir "$OUT" "${ROBOT_ARGS[@]}" "$@" 2>&1 | tee -a "$OUT.out"
