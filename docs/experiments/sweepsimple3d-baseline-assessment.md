# SweepSimple3D baseline and integration assessment

Checkpoint: 2026-09-30. This is a prototype assessment, not an experimental success-rate estimate.

| Evidence | Result | Interpretation |
|---|---|---|
| Published/native Simple bilevel benchmark | Not supplied | Neither a measured failure nor 0% success. The upstream sweep model implements SweepIntoDrawer3D. |
| Fresh production forward readiness script | 2/2 tested native seeds; 5/5 cubes within 6/10 actions on each | Ordinary action dispatch and production controller defaults, with fixed nominal parameters and scripted cube order. Not a bilevel-policy benchmark. |
| Custom diagnostic forward task | 2/2 tested native seeds; 5/5 cubes within 6/10 actions | Promising physical feasibility. Controller overrides and revisions differ; not a frozen policy benchmark. |
| Fresh production complete robot reset | 0/2 tested native seeds | Both runs complete goal, Place, and pickup, then fail the first reverse skill after 9 total actions. Scientific launch remains blocked. |
| Integrated regression suite | 238/238 | Software validation, not physical task success. |
| Calibration / method smokes | 0/48 contexts; 0/2 methods | Still required after controller readiness. |

The highest registered native Simple variants contain 50 cubes. Left-of-island tasks have 1, 5, 10, and 50 cubes; right-of-counter and right-of-island tasks each have 50. The approved learning experiment remains the five-cube left-of-island task.

## Current recommendation

Continue controller development, but do not launch or present Simple as a ready Ours/EES comparison yet. Forward manipulation is feasible in the tested scenes. The unresolved issue is reliable, budgeted recovery to the native start distribution, particularly retaining sufficient native joint reserve during reverse contact.

The north placement stance at the same 0.70 m distance now passes physical placement and fresh pickup. A later reverse-motion failure exposed native chassis collisions with side-counter drawer handles that the upstream base planner omitted. Exact native chassis filtering now rejects those routes. Its checked fixed-reverse 0.65 m fallback physically clears the obstruction and moves the cube farther toward start; subsequent leveling still fails. The native physics and predicates were not changed.

Seed1 also exposes a placement target near the native yaw boundary: its original target settles at about 46.6°, outside the allowed ±45°. A separate central-yaw diagnostic passes the unchanged settled-state checks. The native midpoint-yaw target is now a production default and passes Place/regrasp on both fresh seeds. Earlier overlap and leveling corrections are integrated; the fresh failure is repeated early unloading at the native joint-reserve guard. Failed attempts and replay provenance remain in the artifacts.

| Problem | Correction / evidence | Remaining validation |
|---|---|---|
| Place stance intersects displaced cubes | Original-first, checked north stance; physical Place and pickup pass | Fresh full recovery |
| Native chassis hits side-counter handles omitted by base planner | Native scratch collision filtering; checked .65 m fixed-reverse fallback crosses the obstruction physically | Fresh full recovery |
| Arm converges without sufficient physical blade lowering | Tighter bounded tracking plus measured no-progress leveling restores the previously failing cube | Other cubes and fresh recovery |
| Unloaded leveling target blocked by joint limits | Checked 10–20 mm raised leveling candidates preserve the .95 tilt target and native limits | Leveling passes; complete recovery still fails |
| Successful pickup falsely fails because robot stays home | Remove unconditional base-location effects from pickup/sweep; ignore predicted base facts and retain real native observations | Fresh pickup passes; symbolic parity tests pass; method smokes pending |
| Near-boundary wiper yaw rotates outside range on floor impact | Native midpoint-yaw target integrated; same final native predicate | Fresh release passes 2/2; full recovery pending |

Latest fresh production proofs `production-native-seed0-v253` and `production-native-seed1-v254` are frozen at `324db170`, with ordinary action dispatch and no controller monkeypatches. Both reach 5/5 native goal cubes in 6/10 forward actions and complete counted Place/Pick. Both fail the first reverse skill after 9 total actions with three consecutive no-progress strokes. These are two seeds retested after earlier revisions, not four independent seeds.

The selected floor approaches leave approximately .016 rad of native joint reserve. Loaded contact exhausts the .01-rad guard after about five ticks, so the controller unloads before useful progress and then repeats nearly the same approach. Raised leveling succeeds and native bounds remain unchanged. The next correction must improve checked approach geometry or usable reserve; relaxing the guard or counting a failed reset as success is not permitted.

Counterfactual retries v255b and v256 also fail without cube progress. Ranking a higher-reserve .99-tilt pose trades the joint-limit stop for a tilt stop. Selecting another same-elbow IK branch improves some margins but does not maintain both reserves after physical contact. Neither diagnostic is promoted. The ongoing investigation compares jointly feasible pre-contact geometry and the previously successful narrow-contact reverse diagnostic; all failed ancestral actions remain counted.

Simple-only model correction `3de27093` makes both existing planning engines forget uncertain base-location facts identically after Pick/Sweep, and permits an idempotent Return whenever the hand is empty. Twelve focused tests pass. Actual native observations are unchanged. Both learners and calibration use observed add effects as skill success; the production readiness proof additionally rejects controller errors. This distinction is explicit and unchanged.

## Additional parameter and recovery diagnostics

Four single-cube production support probes use seed 0/cube 4 and ordinary Pick/Sweep dispatch. At distance 0.70 m, both heading endpoints ±15° succeed (2/2). At heading 0°, distance 0.40 m fails blade overlap and 0.55 m fails the no-progress check (0/2). These selected feasibility checks are not random calibration or full-task success estimates. The approved ranges remain unchanged. All four outcomes and result hashes are retained in `scratchpad/sweepsimple3d/forward-support-observations-v257-v261.json`.

Broad southward reverse contact rotates the tool relative to the palm by about 71 mm and 0.99 rad between the measured first-south and later-contact states. Diagnostic v259 localizes existing narrow contact to southward robot reset, at fixed 0.55 m distance with the native cube-contact ceiling, and prefers existing nearly upright approaches with 0.03 rad native joint reserve. It returns cube 0 under the native predicate at counted action 11, then fails cube 1's blade-overlap check at action 12. Final reset count is 1/5 cubes, with no wiper/robot return executed. Earlier failed actions are retained; this replay has no readiness credit and is not promoted. A fresh seed-1 transfer check runs the identical frozen candidate while the second-cube failure is diagnosed.

The fresh seed-1 narrow-contact transfer v260 fails on the first reverse cube: its trailing blade tip contacts a different cube and exhausts the loaded stroke before reaching the target. Other replay candidates reach 2/5 reset cubes but subsequently hit an upstream conservative base-footprint check. These candidates were not promoted, and the footprint check was preserved.

Replay v265 instead retains the broad 0.70 m south stance, tries a native blade-long-axis roll of +0.2 rad first, and consistently uses the existing geometry-derived cube contact ceiling as the fixed-reset height target. The prior internal 8 mm target caused unnecessary lowering even when the blade already satisfied actual overlap. Cube 0 returns under the unchanged native predicate at tick 1643/action 11; cube 1 then fails actual blade-height overlap at tick 4410/action 12. Final count is 1/5 reset cubes. The two minimal reset-only geometry corrections are integrated at `1193bcad`, with 24/24 focused tests and 238/238 established regressions passing; forward behavior and all acceptance criteria are unchanged. This is partial prototype progress, not full reset or readiness evidence.

The remaining cube-1 failure ends with the blade edge about 70.9 mm above the floor, tilt 0.873 rad, and native joint reserve 0.031 rad. Repeated raised leveling follows low-reserve contact approaches; there is no cube contact at the final overlap failure. A checked initial approach that avoids this sequence is under investigation. Fresh full physical proofs, calibration, and method smokes remain pending.

## Baseline distinction

Pinned KINDER is `8f600231da8ee898da1055144665a8c5f6c246f0`; baselines is `427ad6cdfcc79efb489ebabc8fd39d77e104bb4e`. Native task registrations are in KINDER `src/kinder/__init__.py:492–542`. Baselines `tidybot3d_sweep3D.py:55–58` loads Drawer configurations. Its `experiments/conf/env/sweep3d-o5.yaml` selects SweepIntoDrawer3D.

Our global CLI exposes skill-oracle, random-skills, EES, and POMDP, not a bilevel baseline. Simple's oracle selects the first applicable task skill; it is not symbolic bilevel search. Its generic CLI budgets also conflict with Simple's approved run validation. Therefore no truthful frozen custom-bilevel invocation exists today. Adding an evaluation-only baseline requires an explicit search/refinement policy, reproducible parameter sampling, native goal scoring, fixed seeds/budgets, and source hashes. It must remain separate from the approved scientific arms.

## Remaining Ours/EES integration and readiness

1. Verify explicit Place/Pick/reverse/robot-return recovery with counted actions, no hidden resets, and the settled native start contract. The current diagnostic conservatively caps the combined forward-and-reset sequence at 20 actions; the experiment retains physical state across 20-action practice cycles, so this diagnostic cap is not a new reset deadline.
2. Promote the exact verified diagnostic controller choices into production defaults, then repeat fresh native forward and recovery checks without overrides.
3. Validate the approved pickup/sweep parameter supports and all native predicate/operator/accounting contracts.
4. Run 24 fixed calibration contexts for each of the two learned skills, rotating all five targets as approved; preserve failures.
5. Pass EES and Ours method smokes, including explicit recovery planning and independent evaluation state.
6. Freeze clean source/dependencies, calibrated manifest, and hashed evidence; let the persistent readiness gate launch the approved cheap/high paired-seed 2×2, then approved expansion.

Protocol stays unchanged: Ours Model B grid inference versus EES; only determinized A* with 1,000 expansions; 50 practice cycles of 20 total actions; ten held-out evaluation tasks with ten actions; no scheduled practice resets. New floor-specific skills remain disclosed; native physics, goal/start predicates, and reusable shared primitives retain their existing semantics.

Sources: [official Simple page](https://prpl-group.com/kinder-site/environments/sweepsimple3d/index.html), [native registration](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden/blob/8f600231da8ee898da1055144665a8c5f6c246f0/src/kinder/__init__.py#L492), [upstream Drawer model](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines/blob/427ad6cdfcc79efb489ebabc8fd39d77e104bb4e/kinder-bilevel-planning/src/kinder_bilevel_planning/env_models/dynamic3d/tidybot3d_sweep3D.py#L55).
