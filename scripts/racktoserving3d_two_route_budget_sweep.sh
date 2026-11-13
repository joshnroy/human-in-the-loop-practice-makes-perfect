#!/usr/bin/env bash
# Two-route Serving budget sweep: 5 arms x 5 practice budgets x 5 seeds = 125 runs.
#
#   scripts/racktoserving3d_two_route_budget_sweep.sh <results-root>               # launch
#   DRY_CHECK=1 scripts/racktoserving3d_two_route_budget_sweep.sh <results-root>   # parse only
#
# Setup: `--start-layout middle_three --side-offset 0.13 --plate-tuck 0.035`, two-route
# rewards (Sweep 3, GatherAndSweep 5), `--evaluation-policy method`, Beta(1, 3) on every
# skill for every method, human cost 5 through the tick-priced cost file, 10 test tasks,
# evaluation after practice, math threads pinned to one per process. The evaluation
# before practice does not depend on the practice budget, so only the smallest-budget
# cell of each arm and seed runs it (`--no-initial-measurement` elsewhere): read the
# untrained policy's result from that cell. INITIAL_MEASUREMENT=0 (default 1) skips it in
# every cell, for a sweep whose before-practice results already exist elsewhere.
#
# Sampler features. SAMPLER_FEATURES (default parameters) is passed to every
# cell as `--sampler-features`: what the learned sampler's classifier reads for Sweep
# and GatherAndSweep (`RackServingConfig.sampler_features`). Every run of this script
# before 2026-11-13, the 125-cell sweep of 2026-11-12 included, had no such flag and
# used what is now SAMPLER_FEATURES=full_state.
#
# Sampler ties. Every arm passes `--sampler-break-ties-among-best` (Josh, 2026-11-13,
# Serving only): when more than half of a skill's candidates tie at the top score but
# not all of them, the pick is drawn among the tied-best rather than over all
# candidates. SAMPLER_TIES=all restores the rule every run before that date used.
#
# Starting learning rates (Josh, 2026-11-12): DP's three arms and ours run with a
# skill's starting learning rate set to a base rate over its number of continuous
# parameters, and to zero for a skill with none (`--dp-learning-rate-by-parameters`,
# `--pomdp-learning-rate-by-parameters`): Sweep has 1 parameter, GatherAndSweep 3,
# PickScraper and the other skills 0. This is our addition to both planners, the
# episode arm included; EES has no learning-rate prior and is unchanged.
#
# Practice sessions are SESSION_ACTIONS (default 10) actions long, robot skills and
# human resets alike, instead of the usual 150: a skill's sampler is refitted, and DP
# re-plans, only when a session ends. A refit took about 5 seconds in the
# forced-target learning-curve runs of 2026-11-12.
#
# Arms (results under <results-root>/budget-<steps>/<arm>/<method>/<seed>/):
#   ees                     EES, `--practice-groundings all`; reward-blind, no cost estimate.
#   ours                    the belief-space method (Curtis determinized planner,
#                           lambda 0.0015), its current pricing of untried skills. Its
#                           plans may only spend the practice steps that are left: the
#                           step protocol tells it the remaining step budget before every
#                           action. Ours cells run before that change are kept beside
#                           the new ones as <seed>.no-step-budget-<time>.
#   dp-episodes             the paper's DP: an episode budget, nothing priced.
#   dp-steps                DP on the run's own step budget (Josh, 2026-11-13; the DP
#                           arm of every sweep from that date): the budget is the
#                           practice step budget itself, an episode of a skill is priced
#                           at that skill's measured length (STEP_LENGTHS, fixed for the
#                           run), and DP re-solves for the steps that are left at the
#                           start of every session. Not in the default ARMS, which stay
#                           the 2026-11-12 sweep's.
#   dp-cost-fixed-prices    our cost accounting on DP's decision rule: a cost budget with
#                           the per-skill price table fixed for the whole run.
#   dp-cost-learned-prices  the same with the table only a starting value, replaced by
#                           DP's own observed cost of each skill.
#
# Budgets. Every arm is capped by the same `--practice-step-budget`: 10000, 20000,
# 40000, 80000, 160000 control steps. DP also needs a budget in its own unit:
#   cost      0.01 per control step, plus 5 for each human reset assumed: one batch
#             return per 4000 steps (one GatherAndSweep try) and 7 tool recoveries (what
#             DP's PickScraper practice took in every Beta(1, 3) run so far).
#             cost = steps * 0.01125 + 35  ->  147, 260, 485, 935, 1835.
#   episodes  steps / 1875, the mean control steps per target rollout of the setup-A
#             smoke run's episode arm (12 rollouts in 22485 steps)  ->  5, 11, 21, 43, 85.
#             This is the least principled number in the design: an episode has no
#             length, so any conversion assumes a mix of skills.
#
# Order and resuming. Cells run seed-major (every cell of seed 0 is queued before any of
# seed 1), each through scripts/run_sweep.py under run-heavy. MAX_RUNS (default 14) caps
# concurrent runs: it is the size of this script's pool. The pool can be resized while
# it runs: its process id is in <results-root>/pool.pid, and `kill -USR1 <pid>` adds one
# concurrent run, `kill -USR2 <pid>` removes one (GNU xargs). run-heavy's slot count is
# set well above the pool (HEAVY_SLOTS, default 40) so other sessions' jobs holding
# slots do not block these; run-heavy's own memory check before each run still applies,
# but several runs can pass it at once, so size the pool to the memory that is free.
# A cell whose run finished successfully is skipped; so is one whose run is still going
# (a run outlives the pool that started it, so a relaunched pool must not move a live
# run's directory from under it). Any other unfinished cell's directory is moved aside
# (`<seed>.partial-<time>`, never deleted) and the cell is run again.
# FIRST (default empty) names "<seed>:<arm>" pairs whose cells are queued before
# everything else, e.g. FIRST="0:ours"; the rest keep the seed-major order.
#
# Estimate, from the setup-A smoke runs on a loaded machine (about 30 control steps/s
# for DP, 20 for EES, 12 for ours; an evaluation of 10 tasks is roughly 30000 steps in a
# second process):
#   compute  about 540000 run-seconds: 360000 of practice and 180000 of evaluation
#            (125 after-practice evaluations and 25 before-practice ones).
#   wall     about 11 hours at 14 concurrent runs, about 8 at 18; one seed (25 cells)
#            takes at least the 4 to 5 hours of its slowest cell, ours at 160000 steps.
#   memory   1.3 to 1.6 GB per run, plus about 1.3 GB for its evaluation process while
#            an evaluation is in flight, so up to about 2.9 GB per run when runs start
#            together: about 41 GB at 14 runs, about 52 GB at 18. The agents' memory
#            ceiling on agni is 48 GB, which is why the default is 14 and not 18.
#
# DP's cost-unit budget and its price tables are this project's adaptation; the paper's
# DP counts episodes.
set -euo pipefail

ROOT=${1:?usage: [DRY_CHECK=1] [MAX_RUNS=14] $0 <results-root>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export ROOT REPO
export MAX_RUNS=${MAX_RUNS:-14}
export KINDER=${KINDER_CHECKOUT:-/home/josh/.codex/worktrees/serving-scraper-welds/human-in-the-loop-practice-makes-perfect}
export GUROBI=${GUROBI_HOME:-/home/josh/Documents/repos/research/hitl-pmp-results/gurobi-13.0.0}
export TASKS=${TEST_TASKS:-10}
export SESSION_ACTIONS=${SESSION_ACTIONS:-10}
SEEDS=${SEEDS:-"0 1 2 3 4"}
export BUDGETS=${BUDGETS:-"10000 20000 40000 80000 160000"}
ARMS=${ARMS:-"ees ours dp-episodes dp-cost-fixed-prices dp-cost-learned-prices"}
export LAYOUT="--start-layout middle_three --side-offset 0.13 --plate-tuck 0.035"
export INITIAL_MEASUREMENT=${INITIAL_MEASUREMENT:-1}
export SAMPLER_TIES=${SAMPLER_TIES:-best}
export SAMPLER_FEATURES=${SAMPLER_FEATURES:-parameters}
# DP's per-skill prices, in control steps: the mean of each skill's uninterrupted
# executions in the setup-A smoke runs of 2026-11-12 (seed 0, 40000 steps). Small
# samples: PickScraper 424 (n=9) and GatherAndSweep 3362, rounded to 3400 (n=8), from
# DP's cost-budget run; Sweep 737 (n=3), ReturnScraper 488 (n=7), PickFood 728 (n=7)
# and ReturnFoodToRack 466 (n=4) from EES's run. They are bare skill steps; DP's own
# observed price also folds in any setup it executes for a try, of which those runs
# had none.
# dp-steps' episode lengths, in control steps: its own table, not PRICES above.
export STEP_LENGTHS="Sweep=347 GatherAndSweep=3284 PickScraper=431 ReturnScraper=488 PickFood=728 ReturnFoodToRack=466"
export PRICES="PickScraper=424 GatherAndSweep=3400 ReturnScraper=488 PickFood=728 ReturnFoodToRack=466 Sweep=737"

cell_arguments() {  # <budget> <arm>: prints "<method>" then "<shared args>" then "<method args>"
  local steps=$1 arm=$2 method flags table="" flag pair
  local shared="--two-route-rewards 1 $LAYOUT --sampler-features $SAMPLER_FEATURES \
--evaluation-policy method --bulk-transfer-only 1 \
--surface-kind wire_solid --scraper-attachment grasp_weld --practice-step-budget $steps \
--measurement-interval-steps $steps --num-test-tasks $TASKS --evaluation-control-steps 10000 \
--evaluation-workers 1 --evaluation-task-rng per-task --practice-reset-policy never \
--skill-step-limit 12000 --defer-rendering --record-motion all --record-skill-competence \
--record-sampler-draws --human-interruption-cost 5 \
--practice-cost-config src/hitl_pmp/presets/racktoserving3d_tick_priced_costs.json \
--max-steps-per-interaction $SESSION_ACTIONS"
  [ "$SAMPLER_TIES" = all ] || shared="$shared --sampler-break-ties-among-best"
  # The untrained policy is measured once per arm and seed, in the smallest budget's cell.
  [ "$steps" = "${BUDGETS%% *}" ] && [ "$INITIAL_MEASUREMENT" = 1 ] || shared="$shared --no-initial-measurement"
  local dp="--ees-prior-alpha 1 --ees-prior-beta 3 --goal-pursuit-horizon 0 --dp-solver gurobi \
--dp-learning-rate-by-parameters"
  case $arm in
    ees)
      method=ees
      flags="--ees-prior-alpha 1 --ees-prior-beta 3 --practice-groundings all --goal-pursuit-horizon 0" ;;
    ours)
      method=pomdp
      # The belief-space method takes its cost weight from the cost file, not from
      # --pomdp-linear-cost-lambda (configure_practice_accounting overwrites the flag),
      # so this arm gets a copy of the cost file whose only difference is the weight
      # 0.0015. Runs before 2026-11-12 23:10 ET used the shared file's 0.0003.
      shared="${shared/racktoserving3d_tick_priced_costs.json/racktoserving3d_two_route_ours_costs.json}"
      flags="--pomdp-solver determinized_astar --pomdp-determinized-variant curtis \
--pomdp-observation-probability-weight 0.01 --pomdp-linear-cost-lambda 0.0015 \
--pomdp-competence-prior 1 3 --stop-on-no-practice-action \
--pomdp-learning-rate-by-parameters" ;;
    dp-episodes)
      method=dp
      flags="$dp --dp-budget $(( (steps + 937) / 1875 )) --dp-budget-unit episodes --dp-allocation-schedule once" ;;
    dp-steps)
      method=dp
      for pair in $STEP_LENGTHS; do table="$table --dp-skill-cost-steps-for $pair"; done
      flags="$dp --dp-budget $steps --dp-budget-unit steps --dp-allocation-schedule period$table" ;;
    dp-cost-fixed-prices | dp-cost-learned-prices)
      method=dp
      flag=--dp-skill-cost-steps-for
      [ "$arm" = dp-cost-learned-prices ] && flag=--dp-skill-cost-prior-steps-for
      for pair in $PRICES; do table="$table $flag $pair"; done
      flags="$dp --dp-budget $(( steps * 1125 / 100000 + 35 )) --dp-budget-unit cost$table" ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
  printf '%s\n%s\n%s\n' "$method" "$shared" "$flags"
}

run_cell() {  # <seed> <budget> <arm>
  local seed=$1 steps=$2 arm=$3 method shared flags
  { read -r method; read -r shared; read -r flags; } < <(cell_arguments "$steps" "$arm")
  local root="$ROOT/budget-$steps/$arm" run="$ROOT/budget-$steps/$arm/$method/$seed"
  if [ -n "${DRY_CHECK:-}" ]; then
    printf '%s\n' "--env racktoserving3d --method $method --seed $seed --output-dir $run $shared $flags"
    return 0
  fi
  if [ -f "$run/stats.json" ] && grep -q '"succeeded": true' "$run/timing.json" 2>/dev/null; then
    echo "[skip] seed $seed budget $steps $arm: finished"
    return 0
  fi
  # hitl_pmp.cli carries its run directory on its command line.
  if pgrep -f -- "--output-dir $run( |\$)" > /dev/null; then
    echo "[skip] seed $seed budget $steps $arm: a run is still writing $run"
    return 0
  fi
  [ -e "$run" ] && mv "$run" "$run.partial-$(date +%s)"
  mkdir -p "$ROOT/logs"
  cd "$REPO"
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="$GUROBI/site:$KINDER/reference/kindergarden/src:$KINDER/reference/kinder-baselines/kinder-models/src" \
    GRB_LICENSE_FILE="$GUROBI/gurobi.lic" HEAVY_SLOTS="${HEAVY_SLOTS:-40}" \
    HEAVY_MIN_FREE_GB="${HEAVY_MIN_FREE_GB:-14}" \
    run-heavy scripts/with_env.sh python -m scripts.run_sweep --env racktoserving3d \
    --skip-skill-benchmark-gate --methods "$method" --seeds "$seed" --max-workers 1 \
    --results-root "$root" --shared-args "$shared" --method-args "$method=$flags" \
    > "$ROOT/logs/seed$seed-budget$steps-$arm.log" 2>&1 \
    && echo "[ok] seed $seed budget $steps $arm" \
    || echo "[FAILED] seed $seed budget $steps $arm: see $ROOT/logs/seed$seed-budget$steps-$arm.log"
}
export -f cell_arguments run_cell

cells() {
  local seed steps arm pair
  for pair in ${FIRST:-}; do for steps in $BUDGETS; do
    echo "${pair%%:*} $steps ${pair#*:}"
  done; done
  for seed in $SEEDS; do for steps in $BUDGETS; do for arm in $ARMS; do
    case " ${FIRST:-} " in *" $seed:$arm "*) continue ;; esac
    echo "$seed $steps $arm"
  done; done; done
}

if [ -n "${DRY_CHECK:-}" ]; then
  # Parse every cell's arguments exactly as hitl_pmp.cli would, starting no simulator.
  cd "$REPO"
  cells | while read -r seed steps arm; do run_cell "$seed" "$steps" "$arm"; done \
    | scripts/with_env.sh python -c '
import shlex, sys
from hitl_pmp.cli import Cli
count = 0
for line in sys.stdin:
    Cli.parse_args(argv=shlex.split(line))
    count += 1
print(f"dry check: {count} cells parsed")
'
  exit 0
fi

mkdir -p "$ROOT"
xargs -P "$MAX_RUNS" -L 1 bash -c 'run_cell "$@"' _ < <(cells) &
echo $! > "$ROOT/pool.pid"
wait
