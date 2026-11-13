#!/usr/bin/env bash
# Two-route Serving smoke runs: EES, three DP arms and the belief-space method, one
# seed each, Beta(1, 3) on every skill for every method.
#
# DP's arms, by results directory:
#   dp-episodes/           the paper's DP: an episode budget, nothing priced.
#   dp-cost-fixed-prices/  our cost accounting on DP's decision rule: a cost budget
#                          with the per-skill price table fixed for the whole run.
#   main/dp                the same, with the table only a starting value that DP
#                          replaces with its own observed cost (prices estimated online).
#
#   LAYOUT_FLAGS="<start-layout and plate flags>" scripts/racktoserving3d_two_route_smoke.sh <results-root>
#
# LAYOUT_FLAGS is required and has no default: it is the start setup under test, e.g.
#   LAYOUT_FLAGS="--start-layout middle_three --side-offset 0.10"
# plus whatever plate-position flag the chosen setup needs.
#
# Three invocations of scripts.run_sweep, because every DP arm is `--method dp`. Each
# goes through run-heavy on agni; five simulator runs at once.
set -euo pipefail

ROOT=${1:?usage: LAYOUT_FLAGS="..." $0 <results-root>}
: "${LAYOUT_FLAGS:?set LAYOUT_FLAGS to the start-layout and plate flags of the setup under test}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
KINDER=${KINDER_CHECKOUT:-/home/josh/.codex/worktrees/serving-scraper-welds/human-in-the-loop-practice-makes-perfect}
GUROBI=${GUROBI_HOME:-/home/josh/Documents/repos/research/hitl-pmp-results/gurobi-13.0.0}
STEPS=${PRACTICE_STEPS:-40000}
TASKS=${TEST_TASKS:-10}
# 0.01 per control step, as the cost file charges, plus 5 per human reset for about ten resets.
DP_COST_BUDGET=${DP_COST_BUDGET:-450}
DP_EPISODE_BUDGET=${DP_EPISODE_BUDGET:-12}

SHARED="--two-route-rewards 1 $LAYOUT_FLAGS --evaluation-policy method --bulk-transfer-only 1 \
--surface-kind wire_solid --scraper-attachment grasp_weld --practice-step-budget $STEPS \
--measurement-interval-steps $STEPS --num-test-tasks $TASKS --evaluation-control-steps 10000 \
--evaluation-workers 1 --evaluation-task-rng per-task --practice-reset-policy never \
--skill-step-limit 12000 --defer-rendering --record-motion all --record-skill-competence \
--record-sampler-draws --human-interruption-cost 5 \
--practice-cost-config src/hitl_pmp/presets/racktoserving3d_tick_priced_costs.json"
EES="--ees-prior-alpha 1 --ees-prior-beta 3 --practice-groundings all --goal-pursuit-horizon 0"
DP="--ees-prior-alpha 1 --ees-prior-beta 3 --goal-pursuit-horizon 0 --dp-solver gurobi"
# Mean control steps of each skill's uninterrupted executions in the setup-A smoke runs
# of 2026-11-12 (small samples; see scripts/racktoserving3d_two_route_budget_sweep.sh).
DP_PRICES="--dp-skill-cost-prior-steps-for PickScraper=424 \
--dp-skill-cost-prior-steps-for GatherAndSweep=3400 \
--dp-skill-cost-prior-steps-for ReturnScraper=488 \
--dp-skill-cost-prior-steps-for PickFood=728 \
--dp-skill-cost-prior-steps-for ReturnFoodToRack=466 \
--dp-skill-cost-prior-steps-for Sweep=737"
OURS="--pomdp-solver determinized_astar --pomdp-determinized-variant curtis \
--pomdp-observation-probability-weight 0.01 --pomdp-linear-cost-lambda 0.0015 \
--pomdp-competence-prior 1 3 --stop-on-no-practice-action"

cd "$REPO"
sweep() {
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="$GUROBI/site:$KINDER/reference/kindergarden/src:$KINDER/reference/kinder-baselines/kinder-models/src" \
    GRB_LICENSE_FILE="$GUROBI/gurobi.lic" HEAVY_SLOTS="${HEAVY_SLOTS:-20}" \
    HEAVY_MIN_FREE_GB="${HEAVY_MIN_FREE_GB:-14}" \
    run-heavy scripts/with_env.sh python -m scripts.run_sweep --env racktoserving3d \
    --skip-skill-benchmark-gate --num-seeds 1 --shared-args "$SHARED" "$@"
}

sweep --methods ees dp pomdp --max-workers 3 --results-root "$ROOT/main" \
  --method-args "ees=$EES" \
  --method-args "dp=$DP --dp-budget $DP_COST_BUDGET --dp-budget-unit cost $DP_PRICES" \
  --method-args "pomdp=$OURS" &
sweep --methods dp --max-workers 1 --results-root "$ROOT/dp-cost-fixed-prices" \
  --method-args "dp=$DP --dp-budget $DP_COST_BUDGET --dp-budget-unit cost ${DP_PRICES//--dp-skill-cost-prior-steps-for/--dp-skill-cost-steps-for}" &
sweep --methods dp --max-workers 1 --results-root "$ROOT/dp-episodes" \
  --method-args "dp=$DP --dp-budget $DP_EPISODE_BUDGET --dp-budget-unit episodes --dp-allocation-schedule once" &
wait
