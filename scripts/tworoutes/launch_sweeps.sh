#!/usr/bin/env bash
# The Two Routes comparison (docs/experiment-logs/2026-11-12-tworoutes-three-methods.md):
# every cell is one scripts/run_sweep.py invocation of EES, DP and ours over seeds 0..9.
#
#   run-heavy scripts/tworoutes/launch_sweeps.sh <results-root> [sweep ...]
#
# Sweeps: cost (human cost c at budget 500), lambda (ours alone at the smaller lambda),
# budget (budget at c = 1), rate (Route B's learning rate at c = 10). Default: all four.
#
# The human reset's duration equals its cost, so one budget, in cost units, binds all
# three methods: the practice-step budget, which is also DP's --dp-budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$1
shift
SWEEPS=("${@:-cost lambda budget rate}")
WORKERS=${TWOROUTES_MAX_WORKERS:-6}
SEEDS=${TWOROUTES_NUM_SEEDS:-10}
# Cells run at once; a cell is one run_sweep of up to WORKERS runs.
PARALLEL=${TWOROUTES_PARALLEL_CELLS:-1}
spawn() { # run one cell in the background, at most PARALLEL at a time
  "$@" &
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n || true; done
}
LAMBDA=0.02
SMALL_LAMBDA=0.002

costs() { # <human cost> <lambda> -> path of the cost file
  local path="$ROOT/costs/c$1-lambda$2.json"
  mkdir -p "$(dirname "$path")"
  printf '{"robot_step": {"value": 1}, "human_skills": {"human_reset_b": {"cost": {"value": %s}, "duration": {"value": %s}}}, "human_weight": 1, "objective_lambda": %s}\n' \
    "$1" "$1" "$2" >"$path"
  echo "$path"
}

cell() { # <results dir> <budget> <cost file> <methods> [extra shared args]
  if [ -e "$1" ]; then
    echo "skipping $1: already there"
    return
  fi
  # shellcheck disable=SC2086
  scripts/with_env.sh python -m scripts.run_sweep --env tworoutes --methods $4 \
    --num-seeds "$SEEDS" --max-workers "$WORKERS" --results-root "$1" \
    --shared-args "--practice-step-budget $2 --practice-cost-config $3 --practice-reset-policy never --defer-rendering --num-test-tasks 20 --evaluation-max-pour-attempts 1 --evaluation-policy method --measurement-interval-steps 1000000 --max-steps-per-interaction 10 ${5:-}" \
    --method-args "ees=--goal-pursuit-horizon 0" \
    --method-args "dp=--goal-pursuit-horizon 0 --dp-budget $2 --dp-solver enumeration --dp-initial-competence RouteA=0.2 --dp-initial-competence RouteB=0.1" \
    --method-args "pomdp=--pomdp-solver determinized_astar --pomdp-determinized-variant curtis --pomdp-observation-probability-weight 0.01 --pomdp-inference-engine grid"
}

matched_costs() { # <human cost> -> cost file: a price only, one practice step long
  local path="$ROOT/costs/matched-c$1.json"
  mkdir -p "$(dirname "$path")"
  printf '{"robot_step": {"value": 1}, "human_skills": {"human_reset_b": {"cost": {"value": %s}, "duration": {"value": 1}}}, "human_weight": 1, "objective_lambda": %s}\n' \
    "$1" "$LAMBDA" >"$path"
  echo "$path"
}

# As Tossing and Serving price a human reset: it costs c but takes one practice step, so
# the 500-step budget binds every method alike; DP's budget is in episodes and never binds
# first; and all three methods start from one Beta(1, 3) competence prior.
matched_cell() { # <results dir> <human cost>
  if [ -e "$1" ]; then
    echo "skipping $1: already there"
    return
  fi
  scripts/with_env.sh python -m scripts.run_sweep --env tworoutes --methods ees dp pomdp \
    --num-seeds "$SEEDS" --max-workers "$WORKERS" --results-root "$1" \
    --shared-args "--practice-step-budget 500 --practice-cost-config $(matched_costs "$2") --practice-reset-policy never --defer-rendering --num-test-tasks 20 --evaluation-max-pour-attempts 1 --evaluation-policy method --measurement-interval-steps 1000000 --max-steps-per-interaction 10 --competence-prior-mean 0.25 --competence-prior-strength 4" \
    --method-args "ees=--goal-pursuit-horizon 0" \
    --method-args "dp=--goal-pursuit-horizon 0 --dp-budget 250 --dp-budget-unit episodes --dp-solver enumeration" \
    --method-args "pomdp=--pomdp-solver determinized_astar --pomdp-determinized-variant curtis --pomdp-observation-probability-weight 0.01 --pomdp-inference-engine grid"
}

for sweep in ${SWEEPS[*]}; do
  case $sweep in
    cost)
      for c in 1 3 10 20 30 100; do
        spawn cell "$ROOT/cost/c$c" 500 "$(costs "$c" $LAMBDA)" "ees dp pomdp"
      done
      ;;
    lambda)
      for c in 1 3 10 20 30 100; do
        spawn cell "$ROOT/lambda$SMALL_LAMBDA/c$c" 500 "$(costs "$c" $SMALL_LAMBDA)" "pomdp"
      done
      ;;
    budget)
      for budget in 20 40 80 150 300; do
        spawn cell "$ROOT/budget/b$budget" "$budget" "$(costs 1 $LAMBDA)" "ees dp pomdp"
      done
      ;;
    matched)
      for c in 1 3 10 20 30 100; do
        spawn matched_cell "$ROOT/matched/c$c" "$c"
      done
      ;;
    rate)
      for rate in 0.02 0.10; do
        spawn cell "$ROOT/rate/r$rate" 500 "$(costs 10 $LAMBDA)" "ees dp pomdp" "--route-b-rate $rate"
      done
      ;;
    *)
      echo "unknown sweep: $sweep" >&2
      exit 2
      ;;
  esac
done
wait
