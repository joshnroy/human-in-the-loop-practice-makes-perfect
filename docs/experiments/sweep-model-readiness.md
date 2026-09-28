# Sweep Model B readiness audit

Checkpoint: 2026-09-28, 18:14 America/New_York. This is an implementation and
readiness audit, not a launched experiment or evidence of a learning effect.
The proposed experiment remains blocked by model semantics and unverified search
tractability. No numeric horizon reduction or solver substitution was made.

## Goal and scope

Adapt the EXP-22c Model B method to five-cube Sweep, with learned stock
OpenDrawer, PickWiper, and Sweep parameter samplers; individually selectable robot
recovery skills; optional human assistance; continuous practice; and separate
held-out evaluation. The proposed scientific hypothesis is that assistance cost
changes the selected recovery mechanism. The code must offer alternatives rather
than force that result.

The Sweep-only implementation commits are `4d3a41ff` (adapter), `a3118285`
(contract tests), `472dc400` (exact caches), and `0eb60beb` (grid forwarding).
The last method test checkpoint passed 12/12 tests. Ruff and mypy passed for the
changed method files. The environment contract checkpoint passed 13/14 tests,
with 1/14 strict expected failure documenting future pick feasibility.

## Canonical configuration: actual run, not method defaults

Source: `hitl-pmp-results/exp22c/gpoff/results/local_trend-lam3e-6/pomdp/0/config_snapshot.json`,
experiment commit `767323315f3b7d7b1ac4d06b7bb3844a31c5525a`.
Snapshot SHA-256:
`99fea44618aa1f7fbda21dfb3936a8363d803c6c644809201429bec2a21f2cf9`.

| Argument | Actual EXP-22c value | Sweep forwarding requirement |
| --- | --- | --- |
| pomdp_competence_model | local_trend | Existing Model B |
| pomdp_competence_evidence | non_epsilon | Preserve evidence filter |
| pomdp_inference_engine | grid | Explicit override; particle-only adapter restriction fixed |
| pomdp_grid_competence_bins | 25 | Preserve |
| pomdp_grid_learning_rate_bins | 16 | Preserve |
| pomdp_competence_process_noise_std | 0.03 | Preserve |
| pomdp_learning_rate_process_noise_std | 0.005 | Preserve |
| pomdp_learning_rate_decay | 0.9 | Preserve |
| pomdp_learning_rate_max | 0.15 | Preserve |
| pomdp_learning_time_scale | 1.0 | Preserve |
| pomdp_linear_cost_lambda | 3e-6 | Explicit override; default is None |
| pomdp_solver | expectimax | Preserve exact solver |
| pomdp_search_depth | 6 | Toss value only; Sweep requires justified recovery coverage |
| pomdp_max_search_iterations | 100 | Preserve; not an expectimax cutoff |
| pomdp_num_particles | 1024 | Explicit override; cost marginal uses these particles |
| pomdp_num_samples | 100 | Preserve; exact J does not add sampling noise |
| pomdp_observation_probability_weight | 0.0 | Explicit override; default is 0.1 |
| pomdp_log_full_search_tree | False | Preserve |

There are 18 `pomdp_*` entries in this table. A prior chat summary said 19; that
count was incorrect. Competence inference uses a 25-by-16 grid, while cost
inference retains 1024 particles. The canonical method does **not** use 1024
competence particles.

Other canonical settings: exploration epsilon 0.5; goal-pursuit horizon 0,
initial cycles 1, interval 1; EES reset gate false; competence window/recency 5;
planning timeout 10 seconds; sampler training limit 10000 iterations; practice
history enabled; other reproduction flags false. The harness has 50 cycles,
20 actions per cycle, 10 test tasks, and practice reset policy `never`.

The live Sweep factory is routed from the existing CLI. The integration owner
has corrected `ees_reset_gate` to `reset_cost_gate`, sampler-draw recording, and
resolved configuration metadata. The factory regression test checks grid, 1024
cost particles, weight zero, gate off, and recorder enabled without constructing
a simulator. The latest combined adapter check passed 27/28 tests, with 1/28
expected failure for the unresolved pick-feasibility model. Manifest overrides
must remain visible in `config_snapshot.json` as well as the saved manifest.

## Corrections to earlier diagnostics

Earlier probes used particle inference and observation weight 0.1. They are
noncanonical diagnostics, not evidence about canonical grid/weight-zero behavior.
The earlier claim that a surprise penalty dominates the canonical assistance
cost sweep is withdrawn: the actual observation weight is zero.

The cost-prior diagnostic below remains applicable because both inference engines
use the same independent cost marginal. The updated adapter uses the existing
shared grid implementation; it introduces no new prior or inference algorithm.

## Exact performance changes and bounded probes

Exact memoization caches deployment values using the stock-controller beliefs,
their pending example counts, deployment specification, and failure model.
Configured accumulated cost is applied after the cache lookup. Recovery beliefs
cannot directly affect a stock-only terminal deployment policy. Weak-reference
signature caching does not retain posterior buffers, checks identity to prevent
stale hits after object-id reuse, and remains bounded. Competence moments are
cached within each deployment evaluation. No action, outcome, horizon, or solver
was removed or approximated by these changes.

The fixed diagnostic fixture has 27 robot skills plus 1 human skill, five loose
blocked cubes, an open drawer, and a held wiper. Deployment starts with all five
cubes in the pile and uses the stock policy over five actions. Robot configured
costs are 1, human cost 3, and random-exploration competence 0.25. These are
**test values, not calibration**. The fixture predates the new OpenGripper
recovery operator. Its copied symbolic source SHA-256 is
`bc2dd659bb570cbf8174fc0fc4173ea5dfc49efe73e061c2f3c6cc94d9ed0b4d`.

All probes used systemd services with MemoryMax=6G, MemorySwapMax=0, and
OOMPolicy=continue. The stated service wall limit includes startup; solver time
is reported separately. cProfile overhead is included. These are individual
operational measurements; no population speedup inference is supported.

| Diagnostic | Depth | Service limit | Solver seconds | Expanded nodes | Cached states | Completed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Before exact caches, particle/weight0.1 | 20 | 30 s | 29.132 | 311 | 3675 | 0/1 |
| After exact caches, particle/weight0.1 | 20 | 30 s | 28.899 | 3722 | 35603 | 0/1 |
| Corrected grid/weight0 | 20 | 30 s | 28.978 | 3823 | 36509 | 0/1 |
| Depth-series first case, grid/weight0 | 8 | 15 s | 14.037 | 1577 | 19014 | 0/1 |

Observed expanded-node counts, retaining each time limit:

```text
particle/.1 before, 30 s   ###                                    311
particle/.1 cached, 30 s   ##################################### 3722
grid/0 cached,     30 s   ###################################### 3823
grid/0 depth8,     15 s   ################                       1577
```

The requested depth series was 8, 10, 12, 14, 16, 18, 20, with a stop at the
first timeout. Depth 8 timed out, so depths 10 through 20 were **not attempted
in that series**. There were 0/1 completed depths before stopping. The separate
depth-20 probe above also timed out. No final root action or value was available
from these incomplete searches. This does not establish that depth 20 is
impossible, but it provides no evidence that full recovery lookahead is ready.
The experiment horizon has not been lowered to obtain a completion.

Local evidence: services `sweep-modelb-profile-before`,
`sweep-modelb-profile-after`, `sweep-modelb-profile-grid-zero`, and
`sweep-modelb-depth-8-scaling`; profile files `/tmp/sweep-before.prof`,
`/tmp/sweep-after.prof`, `/tmp/sweep-grid-zero.prof`, and
`/tmp/sweep-grid-depth-8.prof`. Probe scripts are under `/tmp/sweep-search-*.py`
and `/tmp/sweep-depth-scaling.py`; these are scratch diagnostics, not launchers.

## Proposed native-cost matrix, not launched

Keep each configured robot primitive cost at 1, matching the Toss stock-skill
cost unit; vary human cost only. Use Model B lambda 3e-6 and weight zero.
For EES, freeze the normalization at its initial Beta(10,1) competence cost,
`d = -log(10/11) = 0.0953101798`, with the reset gate off. Do not retune the
normalization as observed competences change.

| Method | Low human cost | Middle human cost | High human cost | Control |
| --- | ---: | ---: | ---: | --- |
| Model B | 1 | 10 | 20 | No human skill |
| EES | 0.09531018 | 0.95310180 | 1.90620360 | No human skill |

Five pick/place pairs require 10 transfers. Adding wiper parking, drawer closing,
and robot parking gives a nominal 13-action restoration; one extra manipulation
per cube gives 18. These are structural cost references, **not verified
executable recovery plans**. Failures or other recovery operations may exceed
those counts. Levels 1/10/20 bracket the transfer sequence without exceeding the
current per-observation cost cap. They do not guarantee the intended behavior.

At lambda 3e-6, a human cost of 20 versus robot cost of 13 differs by only
0.000021 deployment-value units. With other terms held fixed, an expected
learning or remaining-action advantage larger than that still favors assistance.
There is no canonical surprise penalty. The cost hypothesis should be tested,
not made true by changing the penalty, hiding actions, or forcing resets.

Both human and robot cost beliefs initially use a scaled Beta(1,9) marginal on
[0,20], approximately mean 2. The observation likelihood is Student-t with four
degrees of freedom and scale 0.1. The 1024-point initial representation has a
largest cost particle of 11.42756017 despite the theoretical upper bound 20.
Repeated configured cost observations update the filter; costs are not initially
known merely because the experiment supplies their configured values.

Pure cost-filter calculation, seed 0, 1024 particles (not robot trials):

| Configured observation | Mean after 1 observation | After 2 | After 5 | After 10 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.99111 | 0.99746 | 0.99859 | 0.99872 |
| 10 | 9.91830 | 9.88676 | 9.91406 | 9.95772 |
| 20 | 3.85450 | 8.52331 | 14.06548 | 15.98668 |

```text
Cost20 posterior mean (each # approximately one cost unit)
 1 observation   ####                 3.85
 2 observations  #########            8.52
 5 observations  ##############      14.07
10 observations  ################    15.99
configured      ####################20.00
```

Record configured charges and posterior cost means separately. Global scaling
`c' = a*c` preserves the linear objective only with `lambda' = lambda/a`;
preserving the full inference process additionally requires scaling the cost
prior, its particles/support, likelihood scale, and hard budget. Scaling only
configured costs with the original filter is not an equivalent change of units.

## Remaining model and authorization blockers

- Sweep currently requires only AnyCubeInPile but predicts all five InDrawer.
  This can assign nonzero all-five success probability with four cubes on the
  floor. Requiring all five InPile would also prohibit valid completion when
  four are already in the drawer and one remains in the pile. Partial-practice
  availability and a consistent all-five success model need an explicit decision.
- Conditional effects are not a standalone repair: EES currently labels success
  using unconditional add-effects only. Making Sweep's unconditional set empty
  would label every attempt successful. The success-label contract must remain
  coherent with the outcome model.
- ParkWiper cannot soundly infer that every loose cube becomes controller-pickable.
  The real planner's feasibility depends on the future robot and wiper pose.
  A root-versus-imagined feasibility distinction or an explicit uncertainty model
  needs approval and validation; no unconditional Pickable effect was inserted.
- Actual evaluation currently plans over all provider robot skills, including
  OpenResetDrawer, while the terminal objective assumes stock-only deployment.
  A proposed deployment-skills hook would default to existing skills for other
  domains and restrict Sweep evaluation and speculative scoring consistently.
- Intermediate drawer positions lie between the current Open and Closed
  classifiers and have no corresponding drawer action in the draft model.
- The existing movable-reset interface promises to preserve the robot. A human
  effect asserting RobotHome can conflict with that contract when the robot is
  away. Movable-only versus full-state same-start restoration remains unanswered.
- Recovery controllers are physically fixed, but retain the inherited generic
  Model B learning-rate priors and per-attempt clocks. This distinction is
  deliberate; no zero-rate prior was invented. Recovery controllers do not
  directly enter the stock-only terminal deployment value.

Automatic approval review rejected a combined symbolic drawer/deployment update
because it could alter experiment design without approval for those exact
semantics. That rejected command did not execute. Parallel hook work stopped
before edits; no alternate route performed the rejected changes. These concerns
were relayed to the root agent, along with a request to answer the live
integration agent. Unaffected tests, profiling, configuration corrections, and
this audit continued.
