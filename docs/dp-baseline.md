# Deliberate Practice baseline

`--method dp --dp-budget 2000` implements **DP-adapted**, based on
[Vats et al., Deliberate Practice, §§5.1–5.3](https://arxiv.org/html/2608.13415v1).
It is not an unmodified reproduction of the paper's controller-learning experiments.

## Objective and curriculum

The default Gurobi backend jointly optimizes integer training allocations and
task-policy occupancy using bilinear predicted-success/occupancy products. It
includes every reachable symbolic success transition, including alternatives not
present in today's best plan and skills never executed before.
Failures enter an absorbing zero-reward state. Tossing3D's terminal reward is one
for task success; the objective is mean undiscounted success probability over the
seen tasks. The solver is exact within numerical tolerances for this integer-
allocation terminal-reward model, not an unchanged copy of the discounted model.

The episodic formulation bounds each occupancy by one and enforces an acyclic
deterministic task policy through rank constraints. This is sufficient here:
success probabilities and discount are at most one, rewards occur only at
terminal states and are nonnegative, and failure absorbs. A cycle cannot improve
a policy's terminal value. Initial mass sums to one, and an acyclic policy visits
each state at most once. Thus unit occupancy bounds are valid even at gamma=1;
we do not substitute gamma=1 into the paper's bound 1/(1-gamma).

`--dp-solver gurobi` requires Gurobi to certify both maximum task value and
least-cost tie breaking. `dp_allocations.jsonl` records status, objective, bound,
absolute/relative gap, runtime, allocation, completed episodes and physical cost.
A time limit or failed solve raises; it never silently switches to enumeration.
`--dp-solver enumeration` is an explicit independent small-problem reference.

The cheapest-allocation stage preserves an independently evaluated integer
incumbent with an absolute task-value allowance of `1e-8`, rather than fixing
the reward to a floating-point solver objective. The primary bound is tightened
by the task MDP's all-skills-perfect upper bound. Both stages independently
check physical spend, and returned task value must be within `1e-7` of that
bound; the actual absolute/relative gaps are recorded, not overwritten with zero.
This avoids the reproduced saturation failure where a primary solver objective
of `1.0000009725762244` was pinned despite the reward's upper bound of one.
See Gurobi's [numerical-tolerance guidance](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html).

The selected policy supplies the curriculum order: practice predecessors before
downstream skills, finish their allocated attempts and fit the shared samplers
before advancing. Setup is planned with the shared planner, including human
assistance during practice. If the next curriculum stage is unreachable, the
period stops rather than skipping to random or zero-allocation targets. A fresh
solve with no useful affordable allocation ends practice without STOP-only resets.

`--dp-max-states` is a hard symbolic graph cap. The enumeration reference also
has `--dp-max-skills` and `--dp-max-allocations` caps. Gurobi uses an explicit
`--dp-solver-time-limit` instead of silently inheriting enumeration's limits.

## Two arms, same inner learner

Neither arm uses CMA-ES. Both retain the existing EES parameter sampler, human
competence scoring and practice execution protocol. The method, not the user,
chooses each target skill.

### Held-out Tossing3D prior calibration

`scripts/tossing3d_dp_priors.py` collects shared-learner competence curves on
reserved fitting seeds 2000--2002 and independent validation seeds 2010--2012.
Neither the paired experiment seeds nor their deployment evaluations enter the
fit. Each learner receives 60 toss training rollouts with the same epsilon=0.5,
100 candidates, 10000 sampler training iterations and the shared parameter
proposal/classifier. Samplers refit every ten training tosses. Frozen deployment
clones are probed greedily on 16 independent scenes at 0, 20 and 60 training
rollouts; probes never train the sampler or advance the training learner's RNG.

The initial success probability is the pooled initial-probe rate with Jeffreys
smoothing. Its Beta prior has a fixed weak strength of two, not the pooled
calibration sample count. A nonnegative least-squares linear improvement rate
is fitted against later competence probes, anchored at that initial probability,
in units of all completed training target rollouts, including exploratory draws.
PickCube has no learned parameters and receives zero learning-rate prior.
OpenGripper is measured separately using the same parameterless controller:
each reserved seed supplies 16 fresh scenes, executes the shared PickCube,
relocates movables without changing the robot, and verifies the actual
ClosedEmpty atom and every recovery precondition before executing OpenGripper.
Fit and validation outcomes remain separate, and its initial learning rate is
zero. These probes cover recovery after a pick and movable reset, not every
possible failed-grasp configuration. The collector checks the exact simulator
pins before creating probe artifacts; shared environments are not assumed pinned.
Validation data only assess the frozen fit; they never adjust its parameters.
The held-out Brier score is compared with the old defaults. A measured zero
improvement rate is retained, never replaced with a floor to force practice.

`--dp-prior-calibration PATH` loads the resulting source-bound report for DP
robot competence and learning-rate initialization. Human priors and every EES
default remain unchanged. Every shared robot skill, including currently
unreachable recovery skills, must have a prior at method construction, before
practice or deployment planning. Human skills keep their separate priors.
Missing robot-skill priors, source drift, failed
validation, overlapping splits or experiment-seed reuse raise. Allocation logs
include the report digest and the actual competence/rate used for each skill.
The launcher requires this report and invalidates cost-budget calibration when
the skill-prior digest changes. Bare DP construction without the flag retains
legacy defaults for backwards compatibility, not as an approved full-run setup.

This is a disclosed domain-specific calibration, not an unpublished numerical
prior attributed to the paper. Fresh-scene skill probes do not validate the
experiment's reset-free practice distribution, and fitting through 60 rollouts
does not validate predicted saturation or mastery hundreds of rollouts later.
Collection-driver hashes are preserved separately from fitting-driver hashes;
all shared learner and environment sources must match across collection and fit.
Existing Pick/Toss raw curves can be refitted without changing their measured
parameters. Use `collect-open --seed SEED --output DIR` for recovery probes and
`fit --root CURVES --open-root OPEN_PROBES --output REPORT` to bind the combined
report to the current fitter source. Preserve earlier reports: changing this
report or algorithm source invalidates the old cost-calibration binding and
requires regenerating it before full-launch preparation.

| Arm | Budget unit | Allocation schedule |
| --- | --- | --- |
| DP episode-budget / paper-style | Completed target-skill rollouts | Once before the curriculum |
| DP cost-budget / modified | Robot plus human cost, B=2000 | Every practice period |

A completed target rollout counts whether it succeeds or fails. Setup robot
executions and human resets do not consume an additional episode allocation;
their physical costs are always reported. An interrupted target retains its
spending but does not count as a completed episode. Allocate-once preserves its
remaining allocation across harness periods and stops when it is consumed.
The paper does not unambiguously prescribe reallocation frequency or its
CMA-ES-generation/budget bookkeeping; "paper-style" identifies our operational
interpretation, not an exact reproduction. Both modes can be selected independently
through `--dp-budget-unit` and `--dp-allocation-schedule`.

Allocation counts are integers, matching executed trials. The published equations
do not explicitly declare integrality, so this is not a claim of equivalence to
a continuous-allocation relaxation. Finishing allocated attempts is also not a
guarantee of mastering the real skill. With Beta(10,1) and initial Delta=0.1,
the allocate-once arm can predict saturation after very few attempts. Our
minimum-cost tie break and pruning beyond predicted saturation then select only
those few attempts, leaving most of the budget unused. The paper does not specify
this tie-breaking rule, and its globally optimal allocations need not be unique.
This behavior must not be attributed to the authors' implementation. Priors and
tie handling require review before claiming a meaningful full-run comparison;
we do not silently change priors or replace the paper's <= budget by equality
just to force spending.

### Independent cost calibration

`python -m scripts.tossing3d_dp_launch prepare --phase calibration ...` writes six
3200-step pilot tasks: reserved seeds 1000--1002 for plain and always-ask cost-mode
DP. These are not paired experiment seeds, and evaluation outcomes are not used
to choose budgets. Each reset-policy group receives its own calibrated episode
budget:

`floor(2000 / (sum pilot physical cost / sum completed target rollouts))`.

The report binds calibration to exact algorithm source hashes and refuses
incomplete pilots or non-certified allocations. Calibration matches an available
budget approximately, not guaranteed actual spending by allocate-once DP.
Changing budget unit and allocation timing together means these two arms do not
isolate the causal effect of budget units alone.

After `calibrate --launch-json ... --out ...`, `prepare --phase smoke ...` writes
four seed-0 checks, and `prepare --phase full ...` writes 24 tasks (two modes,
plain/always-ask, seeds 0--5). Preparation never submits jobs. Both require a
matching completed calibration report; full experiments must use reviewed,
merged source, not the patched validation snapshot.

### Gurobi environment

Install the optional `dp` dependency (`gurobipy==13.0.0`) in a supported Python
environment. Gurobi 13.0.0 has no Python 3.14 wheel; Della's Python 3.10 is tested.
Load `gurobi/13.0.0` on each compute node and verify licensed optimization there.
The prepared scripts require `DP_GUROBI_PYTHONPATH` for the isolated binding
directory; the running EES environment is not modified. The sampler dependency
and source are shared, not replaced by a Gurobi-specific learner.

Prepared short validation arrays request 62 minutes so Della assigns production
QoS automatically. Requests of 61 minutes or less land in the two-job-per-user
test QoS, even if a different QoS is requested explicitly. This follows
[Princeton's documented scheduling guidance](https://researchcomputing.princeton.edu/support/knowledge-base/job-priority).

## Explicit adaptations

- The modified arm uses actual robot/human costs and per-period reallocation;
  the paper-style arm uses the episode/allocate-once interpretation above.
- Attempt costs estimate the whole directed attempt: setup, failed/retried setup,
  human resets, target execution, and trailing always-ask resets. Setup executions
  do not consume a second target allocation. Unfinished spending remains attached
  to its intended target and contributes to subsequent estimates.
- Robot costs start from a one-control-step prior. This is optimistic before
  observing setup costs; cost estimation is empirical, not exact forecasting.
- Samplers, competence priors, skill abstractions, and execution are shared with
  the EES comparison. The linear model uses a per-attempt smoothed improvement
  estimate; the paper does not specify our Beta prior or smoothing constant.
- Humans are scored during practice but excluded from deployment. Consequently
  their allocations are zero under the same task-value rule; they can still be
  used as setup and consume the cost budget. They are not fallback practice targets.
- Always-ask removes optional human-reset actions and bundles forced reset charges
  with the directed attempt that caused them.

## What reset reasoning this does not add

Our belief-space method predicts how practice successes and failures change the
physical state, which help request will become necessary, and its future cost.
That is **state-dependent reset lookahead**. DP-adapted instead uses empirical
reset-inclusive attempt prices. It does not search a practice-state tree or
predict an explicit future reset count. Adding that would be another method or
ablation, not a fidelity repair to DP's deployment-policy objective.

## Invalidated pre-fix runs

The first DP implementation scored cached deployment plans, allowed zero-count
fallback targets, and excluded ordinary setup prefixes from attempt prices. Its
passing tests did not establish fidelity to the paper's joint optimization.
Full Della array `16220900` was canceled; those partial results must not be used
as evidence about the repaired baseline. EES is unaffected by these DP repairs.
