# SweepSimple3D baseline and integration assessment

Checkpoint: 2026-09-30. This is a prototype assessment, not an experimental success-rate estimate.

| Evidence | Result | Interpretation |
|---|---|---|
| Published/native Simple bilevel benchmark | Not supplied | Neither a measured failure nor 0% success. The upstream sweep model implements SweepIntoDrawer3D. |
| Custom diagnostic forward task | 2/2 tested native seeds; 5/5 cubes within 6/10 actions | Promising physical feasibility. Controller overrides and revisions differ; not a frozen policy benchmark. |
| Complete custom robot reset | 0/2 tested native seeds | Learning experiments remain blocked. Latest seed0 counterfactual replay retains 20/20 actions and 3/5 restored cubes. |
| Native chassis collision fix regressions | 203/203 | Software validation, not physical task success. |
| Calibration / method smokes | 0/48 contexts; 0/2 methods | Still required after controller readiness. |

The highest registered native Simple variants contain 50 cubes. Left-of-island tasks have 1, 5, 10, and 50 cubes; right-of-counter and right-of-island tasks each have 50. The approved learning experiment remains the five-cube left-of-island task.

## Current recommendation

Continue controller development, but do not launch or present Simple as a ready Ours/EES comparison yet. Forward manipulation is feasible in the tested scenes. The unresolved issue is reliable, budgeted recovery to the native start distribution, particularly blade leveling/overlap during reverse contact.

The north placement stance at the same 0.70 m distance now passes physical placement and fresh pickup. A later reverse-motion failure exposed native chassis collisions with side-counter drawer handles that the upstream base planner omitted. Exact native chassis filtering now rejects those routes. Its checked fixed-reverse 0.65 m fallback physically clears the obstruction and moves the cube farther toward start; subsequent leveling still fails. The native physics and predicates were not changed.

Seed1 also exposes a placement target near the native yaw boundary: its original target settles at about 46.6°, outside the allowed ±45°. A separate central-yaw diagnostic passes the unchanged settled-state checks. That diagnostic is not yet a production default. The next reverse cube fails blade overlap, and a bounded physical tracking correction is being tested. Failed attempts and replay provenance remain in the artifacts.

## Baseline distinction

Pinned KINDER is `8f600231da8ee898da1055144665a8c5f6c246f0`; baselines is `427ad6cdfcc79efb489ebabc8fd39d77e104bb4e`. Native task registrations are in KINDER `src/kinder/__init__.py:492–542`. Baselines `tidybot3d_sweep3D.py:55–58` loads Drawer configurations. Its `experiments/conf/env/sweep3d-o5.yaml` selects SweepIntoDrawer3D.

Our global CLI exposes skill-oracle, random-skills, EES, and POMDP, not a bilevel baseline. Simple's oracle selects the first applicable task skill; it is not symbolic bilevel search. Its generic CLI budgets also conflict with Simple's approved run validation. Therefore no truthful frozen custom-bilevel invocation exists today. Adding an evaluation-only baseline requires an explicit search/refinement policy, reproducible parameter sampling, native goal scoring, fixed seeds/budgets, and source hashes. It must remain separate from the approved scientific arms.

## Remaining Ours/EES integration and readiness

1. Verify explicit Place/Pick/reverse/robot-return recovery within 20 counted actions, with no hidden resets and the settled native start contract.
2. Promote the exact verified diagnostic controller choices into production defaults, then repeat fresh native forward and recovery checks without overrides.
3. Validate the approved pickup/sweep parameter supports and all native predicate/operator/accounting contracts.
4. Run 24 fixed calibration contexts for each of the two learned skills, rotating all five targets as approved; preserve failures.
5. Pass EES and Ours method smokes, including explicit recovery planning and independent evaluation state.
6. Freeze clean source/dependencies, calibrated manifest, and hashed evidence; let the persistent readiness gate launch the approved cheap/high paired-seed 2×2, then approved expansion.

Protocol stays unchanged: Ours Model B grid inference versus EES; only determinized A* with 1,000 expansions; 50 practice cycles of 20 total actions; ten held-out evaluation tasks with ten actions; no scheduled practice resets. New floor-specific skills remain disclosed; native physics, goal/start predicates, and reusable shared primitives retain their existing semantics.

Sources: [official Simple page](https://prpl-group.com/kinder-site/environments/sweepsimple3d/index.html), [native registration](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden/blob/8f600231da8ee898da1055144665a8c5f6c246f0/src/kinder/__init__.py#L492), [upstream Drawer model](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines/blob/427ad6cdfcc79efb489ebabc8fd39d77e104bb4e/kinder-bilevel-planning/src/kinder_bilevel_planning/env_models/dynamic3d/tidybot3d_sweep3D.py#L55).
