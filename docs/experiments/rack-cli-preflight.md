# Rack CLI preflight: prepared, NOT launched

These commands are a pending smoke-test plan, not approval to run. Do not run
before the physics owner confirms transfer/recovery checks pass and announces an
available native slot. Reaching the rack via push-rear does not validate transfer.
No geometry, controller, algorithm, or Algorithm 3 change is part of this plan.

## Current compatibility

- EES is wired to `RackServingCli` in both cycle and control-step modes.
- Both POMDP solvers now route through a rack-specific `PracticeDomain` registered
  for `RackServingCli`. Unit tests construct the real methods without initializing
  native physics; this removes the missing-domain blocker, not the physics gate.
- "Curtis" here means the existing `--method pomdp --pomdp-solver determinized_astar`
  implementation. There is no `--method curtis` or `--pomdp-backend` flag.
- Solver choices: `expectimax`, `determinized_astar`. Inference engine choices:
  `particle`, `grid`. Competence model choices: `global_curve`, `local_trend`.
  The pending commands use existing Model B (`local_trend`) with grid inference
  (25 x 16) and 1024 cost particles. Parser acceptance is not a physical pass.
- Exact expectimax uses `--pomdp-search-depth 1` for this diagnostic only;
  determinized A* refuses a search depth.
  Determinized A* ignores that depth argument: its limits here are 32 queue pops
  per decision and four remaining action slots per session. No approximation or
  modification is added to either implementation.

## Rack model contract

The domain infers six robot skills (`PickScraper`, `GatherAndSweep`, `ReturnScraper`,
`PickFood`, `PlaceFoodOnTray`, `ReturnFoodToRack`) and the two offered human
skills (`human_batch_return`, `human_tool_recovery`). Only `GatherAndSweep(s, delta, d)` receives
sampler examples; the others receive outcome examples. Food is ready from the
start (no cooling, no Wait skill, since 2026-11-03). Human skills use the provider's configured interruption cost, charged
on success and failure, and are excluded from deployment.
Disabled or deployment-only providers also remove human beliefs from practice.

The analytical deployment value starts with five foods on the rack. Its horizon
is 12 skill actions; either sweep delivery or individual pick/place delivery is available.
The symbolic goal is all food on the tray, not scraper returned home.
Beta(1, 3) competence priors and a fixed-seed 4096-sample expectation mirror the
existing shelf domain as modeling assumptions, not measured rack performance.
Default search depth is three; the smoke commands explicitly request one.
No new symmetry reduction, algorithm, or search modification is introduced.

Like the shelf analytical value, deployment failures leave this abstract state
unchanged. It does not model settling, spills, contact adhesion, or automatic
clock advancement during motion. Practice still uses the actual rack operators
and learned failure effects; evaluation still uses the live backend verdict.
In particular, transient `OnTray` observations followed by roll-off are not
validated delivery. Native learner execution remains blocked pending diagnosis.

## Budgets and interpretation

Use 4000 practice control steps, 4000 evaluation control steps per test task,
2000 control steps per individual skill, and four skill actions per session.
The supplied estimates total 438 + 800 + 1300 = 2538 steps for pickup, sweep,
and return, leaving 1462 steps for additional motion/overhead. They are estimates,
not validated transfer or recovery bounds; any timeout/cutoff is diagnostic.
Initialization and planning are not charged physics steps, so wall/memory caps
remain necessary. The 2000 per-skill cap accommodates the estimated 1300-step
return, unlike the old 500-step evaluation default.

The existing wire rack, 45 mm dumpling width, and scraper geometry remain intact.
(Note, 2026-11-04: the food is now Pouring3D's frozen potsticker, whose size is read
from its shape; `--dumpling-width` no longer exists and was removed from `COMMON`.)
No policy is forced to execute pickup/sweep/return: it may choose food actions,
human help, or STOP. These tiny settings do not evaluate method quality.

**One-cycle caveat:** the step runner deliberately supersedes `--num-cycles`.
Four actions limits one session, not the whole run; sessions continue until the
4000-step budget or an existing stopping condition. Do not advertise a step-budget
command with `--num-cycles 1` as one cycle. A genuine one-cycle alternative is
provided below, without a total control-step bound; it retains per-skill and wall
caps. Its normal evaluation horizon is 13 skill actions per task, not four.

## Exact pending commands

After the physical pass and explicit go-ahead, use this setup in bash. The
function also requires `RACK_PHYSICS_PASS_CONFIRMED=yes` set only after that
go-ahead, creates a unique unit/output
name, and refuses existing output. Each invocation queues through `run-heavy`,
then runs under `agents.slice` with 6 GiB memory, zero swap and an 8-minute
maximum (TERM at 475 seconds, KILL five seconds later). A 10-minute outer
maximum (TERM at 595 seconds, KILL five seconds later) also bounds queue waiting.
Execute one command at a time and inspect its outcome before the next.
Native execution requires the normal sandbox escalation; no learner run has been requested here.
These caps are diagnostic limits, not a runtime estimate.

```bash
cd /home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/.codex-worktrees/kitchen-rack-sweep

COMMON=(
  --env racktoserving3d
  --seed 0
  --canonical-seed 0
  --num-test-tasks 1
  --surface-kind wire
  --no-scene-bg
  --skill-step-limit 2000
  --human-reset
  --human-interruption-cost 5
  --practice-reset-policy never
  --max-steps-per-interaction 4
  --goal-pursuit-horizon 0
  --planning-timeout 5
  --sampler-max-train-iters 1
  --defer-rendering
)

STEP_FLAGS=(
  --practice-step-budget 4000
  --measurement-interval-steps 4000
  --evaluation-control-steps 4000
  --no-initial-measurement
  --human-skill-steps 1
  --stop-on-no-practice-action
)

POMDP_FLAGS=(
  --method pomdp
  --pomdp-max-search-iterations 32
  --pomdp-competence-model local_trend
  --pomdp-inference-engine grid
  --pomdp-grid-competence-bins 25
  --pomdp-grid-learning-rate-bins 16
  --pomdp-num-particles 1024
  --pomdp-num-samples 4
  --pomdp-observation-probability-weight 0
  --pomdp-linear-cost-lambda 3e-6
  --no-pomdp-log-full-search-tree
  --pomdp-compress-decision-log
)

run_preflight() {
  if [[ ${RACK_PHYSICS_PASS_CONFIRMED:-} != yes ]]; then
    printf '%s\n' 'STOP: physics pass and a native slot must be explicitly confirmed.' >&2
    return 2
  fi
  local name="$1"
  shift
  local stamp="$(date -u +%Y%m%dT%H%M%S%N)-$$"
  local unit="rack-cli-${name}-${stamp}"
  local output="$PWD/artifacts/rack-cli-preflight/${name}-${stamp}"
  [[ ! -e "$output" ]] || return 2
  timeout --signal=TERM --kill-after=5s 595s \
    run-heavy \
    systemd-run --user --scope --unit="$unit" --slice=agents.slice \
      -p MemoryMax=6G -p MemorySwapMax=0 -p OOMPolicy=continue \
      timeout --signal=TERM --kill-after=5s 475s \
      scripts/with_env.sh python -m hitl_pmp.cli \
      "${COMMON[@]}" "$@" --output-dir "$output"
}
```

The commands below are **not to be run yet**. The rack domain is implemented and
unit-tested, but all three learners remain blocked on the physical pass and an
available slot. No launch-gate override is supplied; the existing shelf gate is
not a rack physics certificate.

```bash
run_preflight ees-step "${STEP_FLAGS[@]}" --method ees
run_preflight curtis-step "${STEP_FLAGS[@]}" "${POMDP_FLAGS[@]}" --pomdp-solver determinized_astar
run_preflight expectimax-step "${STEP_FLAGS[@]}" "${POMDP_FLAGS[@]}" --pomdp-solver expectimax \
  --pomdp-search-depth 1
```

For a literal one-cycle smoke **instead of** the corresponding step-budget run:

```bash
run_preflight ees-cycle --num-cycles 1 --method ees
run_preflight curtis-cycle --num-cycles 1 "${POMDP_FLAGS[@]}" --pomdp-solver determinized_astar
run_preflight expectimax-cycle --num-cycles 1 "${POMDP_FLAGS[@]}" --pomdp-solver expectimax \
  --pomdp-search-depth 1
```

The one-cycle versions use the ordinary runner and can evaluate before and after
practice. Their evaluation is not subject to `--evaluation-control-steps`, which
only the step runner reads. Do not mix the two protocols' measurements. The one
sampler training iteration, four theta samples, shallow expectimax and 32-pop A*
are explicitly smoke-only settings, not canonical experiment configurations.

## Native observer test

This separate test initializes a scene, runs `PickScraper` to an exact three-control-step
cutoff, and verifies adoption issues no further physics step. It validates hooks,
not transfer. On 2026-10-31 it acquired an available shared slot and passed:
`1 passed in 2.08s`, in scope `rack-cutoff-20261031T214634770045645-1152119.scope`.
The slot is released. This is not a learner launch or a physical delivery pass.
Any repeat must wait for an available slot. Its execution maximum is 180 seconds,
including five seconds of termination grace.

```bash
timeout --signal=TERM --kill-after=5s 595s \
  run-heavy \
  systemd-run --user --scope \
    --unit="rack-cutoff-$(date -u +%Y%m%dT%H%M%S%N)-$$" --slice=agents.slice \
    -p MemoryMax=6G -p MemorySwapMax=0 -p OOMPolicy=continue \
    timeout --signal=TERM --kill-after=5s 175s \
    env HITL_RACK_NATIVE_CUTOFF=1 scripts/with_env.sh pytest -q \
    tests/environments/racktoserving3d/test_native_control_step_cutoff.py
```
