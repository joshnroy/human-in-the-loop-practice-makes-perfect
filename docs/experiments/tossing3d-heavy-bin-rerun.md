# Tossing3D: 3.8 kg bin rerun proposal

> Provenance backfill (2026-10-26): exact source commit/tree and original launch
> argv were not retained for this local probe. Both are unknown; the dependency
> revisions below and saved scripts/outputs are the remaining evidence.

**Superseded mass:** the default is now 1 kg. See
[`tossing3d-1kg-bin-validation.md`](tossing3d-1kg-bin-validation.md) for current
physics checks. The 3.8 kg validation below is retained as historical evidence;
use 1 kg for the proposed new runs.

The simulation backend now defaults to a 3.8 kg bin and 0.1 kg cubes in both
layouts. Explicit masses in a custom task configuration take precedence. The bin
remains movable. `scripts/with_env.sh` sends TMP, TMPDIR, and TEMP to
`artifacts/tmp`, or to `HITL_PMP_TMPDIR` when supplied.

## Current validation and blockers

Compiled MuJoCo subtree masses are 3.8 kg and 0.1 kg in both layouts, and with
the fixed-bin opt-out. The saved dragging regression still moves the bin
0.137679 m, compared with 0.187681 m for the previous 0.1 kg bin. This is one
matched case, not a tip-rate or drag-rate estimate.

The pouring success tests are deliberately retained: the heavier bin currently
breaks the previously successful controller. The first saved refusal
(`exp22_A`, barrier layout) fails at the lift stage after 251 control steps; the
maximum bin origin height is only 0.0957 m. Raising motor forces or lowering the
mass to make a success assertion pass would change the experiment. The earlier
video validates the 0.1 kg prototype, not the new bin.

The targeted integration suite finished with **168 passed and 8 failed** in
312.53 seconds. All eight failures are the physical pouring success checks: four
saved refusal fixtures in both layouts fail to reach the lift pose. The remaining
layout, fixed-bin, KINDER-fidelity, and belief-model checks passed. After adding
compiled-mass assertions, the four applicable movable/fixed-bin checks passed in
6.59 seconds. Scoped Ruff, formatting, mypy, and `git diff --check` passed.

Diagnostic files are under `artifacts/tossing3d-pour-bin-3p8kg/`:
`mass-and-drag-check.json`, `lift-diagnostic.json`, and `lift-trace.json`.
No Della jobs have been submitted for this change.

## Open questions

| Question | Recommendation |
| --- | --- |
| Can this arm and rim grasp lift and invert the 3.8 kg bin along this path? | Resolve the lift failure with the intended actuator and grip model before learning experiments. Diagnose tracking, contact slip, and lever arm separately. |
| How stable is the heavier bin during ordinary picks and tosses? | Measure translation, maximum tilt, and uprightness after each attempt across seeds. The one replay still drags 13.8 cm. |
| Which solver revision should be compared? | Integrate the mass and pouring changes with the exact relaunch-2 solver revision. The archived runs used `pomdp_determinized_variant=curtis` and `pomdp_draw_scope=node`; this checkout exposes neither option. Its `determinized_astar` is a different implementation, so a matching solver name is insufficient. |
| Does pouring reduce human resets independently of the mass change? | After the main replication, pair the three plain arms with pour disabled and the same 3.8 kg movable bin. An independent pour-off switch is needed; fixing the bin would confound this comparison. |

Tipped-bin failures may still require a human reset; this is accepted behavior.

## Proposed Della runs

First run the four saved refusal fixtures in both layouts (eight physics
rollouts). Require actual lifting, sideways inversion after the base quarter-turn,
return placement, and a successful follow-up cube pick. Then run six short harness
pilots: each plain method below on seeds 0 and 2, 6,400 control ticks, evaluations
every 3,200 ticks, and early stopping disabled. These pilots are diagnostics, not
replacement benchmark seeds.

After those pass, rerun the six relaunch-2 arms below from fresh training on
paired seeds 0–5, for 36 full runs. All use Tossing3D-o1, barrier practice and
evaluation layouts, 160,000 practice control ticks, evaluations every 3,200 ticks,
10 fixed held-out tasks per seed, a 500-control-tick evaluation cap, and stopping
after three consecutive 10/10 evaluations. Use 20 action slots per practice
period, no goal pursuit, no scheduled resets, epsilon 0.5, 10,000 sampler training
iterations, canonical seed 125, and test seed offset 10,000. Keep the initial
evaluation, deferred rendering, skill timing, and compressed decision logs.

| Arm | Seeds / full runs | Important settings |
| --- | --- | --- |
| EES | 0–5 / 6 | `--method ees --no-ees-reset-gate`; Beta(10,1), competence window/recency 5/5; reproduce-predicators explore-target-only, seen-task-order, random-when-stranded, and practice-target-history enabled; sampler draws recorded. |
| EES always-ask | 0–5 / 6 | Same EES settings plus `--always-ask-reset`; price `reset_to_sampled_start` explicitly. |
| Expectimax | 0–5 / 6 | `--method pomdp --pomdp-solver expectimax`; depth 6, observation-probability weight 0, Model B/local_trend, grid 25×16, 1,024 cost particles, 100 theta samples, non_epsilon evidence. |
| Expectimax always-ask | 0–5 / 6 | Same expectimax settings plus `--always-ask-reset`; price `reset_to_sampled_start` explicitly. |
| Determinized A* (Curtis) | 0–5 / 6 | Relaunch-2 Curtis variant with node-scoped draws; 100 queue iterations per decision, observation-probability weight 0.01; same Model B inference settings as expectimax. Restore the matching implementation first. |
| Determinized A* (Curtis) always-ask | 0–5 / 6 | Same Curtis settings plus `--always-ask-reset`; restore the matching implementation and price `reset_to_sampled_start` explicitly. |

The archived relaunch-2 cost files used robot tick cost 0.01, human reset cost 5,
and objective lambda 0.0003. This checkout's canonical file uses costs 100 times
larger (1 and 500), with lambda 0.000003. The product lambda × cost is equivalent
for the linear objective, but do not accidentally combine one scale's costs with
the other scale's lambda. Reuse the archived units for the comparison and verify
the EES costs too. Copies of all six historical argument snapshots and cost files
are saved in `artifacts/tossing3d-pour-bin-3p8kg/rerun-configs/`; argument snapshots
include old output and checkout paths and are references, not executable commands.

An optional second stage adds 18 runs: the three plain methods with pouring
disabled, paired seeds 0–5, and exactly the same masses, costs, and protocol.
Keep both layouts in local physics validation; a same-side learning experiment
would be a separate study rather than a relaunch-2 replication.

On Della, run one seed per array task against a frozen code/submodule snapshot.
Point `HITL_PMP_TMPDIR` at a job-specific directory under the project's GPFS scratch
allocation before invoking `scripts/with_env.sh`; the wrapper exports all three
temporary-directory variables. Keep BLAS/OpenMP thread counts explicit, save the
resolved configuration and code identity, and use fresh output directories.
Resource requests should follow the passing pilot's measured memory and runtime.

Compare success, resets to first 10/10, resets at matched practice ticks, and both
the last scheduled and end-of-practice evaluation scores, as in relaunch 2. Also
retain pour attempts, successes, failure stages, ticks spent pouring, bin
translation/tilt, tipped-bin resets, planning timeouts, and stuck-run guard exits.
