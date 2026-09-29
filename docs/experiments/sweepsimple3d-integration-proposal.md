# SweepSimple3D: native audit and proposed integration

Status: Josh explicitly approved the proposal on2026-09-29 through the parent
session: new floor-wiper pickup, per-cube forward/reverse sweeps, individual wiper
and robot return, the stated feasibility supports, and10-action deployment for
both methods while practice remains20actions. Implementation is underway.
No learning experiments have been run.

## Native task and the missing implementation

Proposed variant: `SweepSimple3D-o5-sweep_the_blocks_to_the_left_side_of_the_kitchen_island`.
This keeps five cubes, but their configured size is0.01m and the wiper handle is0.30m.
The cubes and wiper start on the floor. Cubes start in x[1.25,1.75],y[0.2,0.7];
their goal is the floor region x[0.125,0.875],y[1.0,1.375]. The wiper starts in
x[1.25,1.75],y[0.7,1.2],yaw[-45,45]degrees. The native task requires all five
cubes in the goal region. No drawer operation belongs to this task.

Pinned `kinder-models` has no SweepSimple3D controller or symbolic model. Its
`dynamic3d/sweep3D` module is for SweepIntoDrawer3D. The corresponding bilevel factory
constructs the Drawer task explicitly; its operators require OnTable and DrawerOpen
and its Sweep effect is InDrawer. Its three fixed relative sweep transforms target
a drawer stroke, not this task's floor goal. The transforms are relative to observed
objects, not fixed world-height constants, but this does not make the controllers
an implementation of SweepSimple3D.

Sources under the pinned worktree:

- `reference/kindergarden/src/kinder/envs/dynamic3d/tasks/SweepSimple3D/SweepSimple3D-o5-sweep_the_blocks_to_the_left_side_of_the_kitchen_island.json`
- `reference/kindergarden/src/kinder/envs/dynamic3d/envs.py`, `_check_goals`
- `reference/kinder-baselines/kinder-bilevel-planning/src/kinder_bilevel_planning/env_models/dynamic3d/tidybot3d_sweep3D.py`
- `reference/kinder-baselines/kinder-models/src/kinder_models/dynamic3d/sweep3D/parameterized_skills.py`

## Completed native checks

The bounded audit instantiates seeds0,1,2 and10000–10009 directly from the pinned
native task, without changing physics or task configuration. It reads the exact
native goal checker and restores the original full observed state. The JSON retains
every declared-region/yaw check, position and failure, plus import provenance.

All13/13 original observations round-trip exactly within1e-7, and native goal reads
leave state unchanged13/13. Strict native region containment rejects the wiper in
13/13 starts: its resting origin settles to about−0.000108m, below the ground plane.
The audit separately records the existing1cm wiper support tolerance from the Drawer
start/reset checker; this tolerance never modifies native goal scoring. With that
shared support tolerance, all13/13 starts pass the declared region and yaw checks.
No invalid seed is resampled. This is initialization evidence, not controller readiness.

Audit command: `scripts/with_env.sh python scripts/audit_sweepsimple3d.py`, under a
6GiB systemd service with MemorySwapMax=0. Output:
`scratchpad/sweepsimple3d/native-audit.json`; image:`native-seed0.png`.

## Proposed smallest meaningful controller set

These are NEW local skills, not claimed upstream implementations. Reuse existing
native simulator, goal/region checking, collision geometry, inverse kinematics,
and motion-planning utilities without changing their behavior.

| Skill | Role | Parameters / target |
|---|---|---|
| PickFloorWiper | Learnable task skill | Feasibility-test base distance0.55–0.85m and heading±15degrees around a geometric grasp approach; preserve collision checks |
| SweepCubeToGoal | Learnable task skill, grounded once per cube | Feasibility-test base distance0.40–0.70m and heading±15degrees around a geometric sweep approach; execute a physical tool path toward the existing native goal region |
| SweepCubeToStart | Fixed recovery skill, grounded once per cube | Same physical machinery, target the cube's declared start region; no teleportation and no forced sequence |
| PlaceWiperAtStart | Fixed recovery skill | Return the wiper into its declared floor region and allowed orientation |
| OpenGripper | Fixed recovery skill | Explicitly release whatever is actually held |
| ReturnRobotToStart | Fixed recovery skill | Collision-checked return to original valid robot start |

Distance/angle bounds above are proposed feasibility ranges, NOT authorized frozen
experiment ranges. Success is observed after execution. Separate physical holding
and plannability checks prevent the known Drawer integration mistakes. A failed
pick or sweep stays a failed attempt; no human fallback or hidden reset occurs.
Do not claim an object is irrecoverable merely because the proposed controller fails.

One cube per sweep keeps individual recovery actions available to both planners;
do not hide all five returns inside a single cheap reset macro. This changes the
task skill count relative to Drawer: picking the wiper and sweeping five cubes
requires at least6task actions. The existing5-action deployment horizon therefore
cannot simply be copied. Propose a fixed10-action deployment horizon for BOTH
methods, subject to approval and feasibility before scientific runs. Practice
remains20actions per cycle, including each human intervention.

## Unchanged experiment protocol and explicit decisions

Preserve EES versus Ours ModelB; GPoff; grid25x16;1024cost particles;lambda3e-6;
determinized_astar only,1000queue pops;50practice cycles;20action cap;10held-out
evaluation tasks in an isolated simulator; no automatic practice resets.
Human reset restores the original validated initial sample. Robot recovery targets
the same declared region/orientation conditions. Retain initial strict and tolerance
checks in the published data.

Copy the approved native configured human costs exactly: Ours1,10,20; EES
0.0953101798043249,0.9531017980432489,1.9062035960864978; plus no-human arms.
Run cheap/high2x2 first with one paired valid seed, then middle/no-human and seed
expansion only after readiness. Report actual robot-reset action counts; the new
environment may not make these nominal costs bracket the robot sequence, and no
crossover is guaranteed or forced.

Decision approved by Josh on 2026-09-29: the five-cube native variant, the new explicit floor
skills and feasibility ranges above, and 10-action deployment horizon. This is a
real controller/model extension, not merely an environment-name conversion.

## Readiness and work order after approval

1. Floor-wiper pickup and physical forward sweep feasibility, preserving failures.
2. Learnable parameter variation and identical sampler support for both methods.
3. Individual reverse-sweep/wiper/robot recovery, including reachable post-task states.
4. Native goal parity, true action admission and observed effects; no hidden resets.
5. End-to-end learned-method smoke with independent practice/evaluation state and
   matching action accounting. Do not launch a scientific cohort if the native task
   remains physically unreachable with this controller set.
6. Freeze new source, cost/support manifest and separate output path, then launch
   capped persistent runs. Existing Drawer results remain separate.

Estimated first physical feasibility checkpoint after approval:1–2hours. Complete
integration/launch ETA depends on that result; no credible overnight guarantee yet.
