# tossing3d

KINDER's `Tossing3D` benchmark, behind this project's `core/` interfaces.

A TidyBot++ mobile manipulator must get a cube from the floor to a goal region on the far
side of an immovable 5 m barrier. The base cannot pass the barrier, so the cube can only
get there through the air: the robot must **toss** it. A tossed cube cannot be retrieved
— hit or miss, it ends up past the barrier and no skill brings it back. That
irreversibility is why this domain is here: it is the concrete case the project's own
proposal names as EES's predicted failure mode.

**This is an integration, not a port.** No dynamics, no controllers and no success
criterion are written here.

## What comes from KINDER, and what is ours

Everything in the left column is upstream's code or upstream's number, used unmodified.

| upstream gives | used for |
| --- | --- |
| `kinder.make("kinder/Tossing3D-o1-v0", ...)` and all its physics | the entire simulator |
| `kinder_models.dynamic3d.shelf.parameterized_skills`' `pick_shelf` | the grasp |
| `kinder_models.dynamic3d.tossing.parameterized_skills`' `move_to_target`, `move_arm_to_conf`, `toss` | the walk and the throw |
| `_check_goals()` | the success criterion this domain's own predicate is checked against |
| `Region.bbox` on `blocks_goal_region` | the scored box, read live and carried in the `State` **on the bin** |
| `MOVE_TO_TARGET_DISTANCE_BOUNDS` / `MOVE_TO_TARGET_ROT_BOUNDS` | the `Pick` sampler's range |
| the windup conf `(0, 50, 180, -110, 0, -100, 90)°` and toss conf `(0, 20, 180, -35, 0, 25, 90)°` | `Toss`, verbatim — nothing is interpolated |
| the `o1` and `o2` task JSONs, the `task_view` camera, `render_fps` | the scene and the clip |
| `test_pick_ground_toss`'s own `1.35` standoff, `rng(123)` pick draw, and 400/200/200/200 step limits | the oracle and the per-controller budgets |

Ours, and therefore ours to defend:

- **The six predicates** (`predicates.py`). Three — `HandEmpty`, `Holding`, `OnGround` —
  are ported from upstream's own `kinder_models.dynamic3d.shelf.state_abstractions`,
  thresholds included, with **one documented deviation**: upstream's `Holding` also
  requires a forward-kinematics end-effector check through a live `PyBulletSim`, which a
  pure-`State` predicate cannot do, so that conjunct is dropped and the predicate is
  correspondingly *weaker* than upstream's. `InBin` reproduces `_check_goals()`
  exactly and is differentially tested against it. `Reachable` and `RobotAtSuccessfulThrowPose` are new.
- **The three lifted skills and their operator models** (`skills.py`). KINDER ships
  symbolic models for `Shelf3D`, `Sweep3D` and base motion, but **none for Tossing3D**,
  so the operator layer is written here — following `tidybot3d_shelf3D.py`'s shape, where
  a `LiftedOperator` is paired to one of upstream's controllers.
- **`THROW_STANDOFF_BOUNDS = (1.10, 1.75)`**, the one genuinely new continuous range.
  Upstream's own `MOVE_TO_TARGET_DISTANCE_BOUNDS` is `(0.5, 0.6)` — a *grasping* standoff
  — and upstream's tossing test simply hardcodes `1.35` with no range at all. This is the
  measured **feasible** range `[0.40, 2.06]` inset at both ends: above 2.06 m `Toss`'s
  windup fails to motion-plan, and below 1.10 m the base can drive through
  `cuboid_barrier` — a real dynamic MuJoCo body upstream's base motion planner does not
  collision-check against — and knock it over. The worst measured colliding standoff is
  1.00 m; `BARRIER_COLLISION_MARGIN` (0.10 m) sets the floor 1.10 m above that. It is
  what the sampler draws from and **not** what `RobotAtSuccessfulThrowPose` accepts —
  see below.
- **`THROW_RANGE = 1.275`**, the distance a throw displaces the cube. `Toss` takes no
  parameters (fixed windup conf, fixed toss conf), so this is a property of upstream's
  controller rather than of the scene, which is what lets the success band be derived
  from live state instead of pinned to one bin position.
- **The `[skill_id, param_0, param_1]` action encoding**, the `State` schema, and the
  `scene` object that carries the episode seed.
- **Nothing about the scene itself.** This domain used to ship its own task JSON; it
  no longer does (see below).
- `H_eval = 3 + 2` (`problem.py`).

## The scene, and the defect it used to have

This domain runs whatever `Tossing3D-o1.json` the installed KINDER registers. It selects
**no** scene of its own and passes no `task_config_path`. That is a deliberate decision
with a real cost, and both halves are worth knowing before touching anything here.

**The defect.** The goal predicate is `["on", "cube_0", "blocks_goal_region"]` — a
*ground region* that the bin merely sits near, never the bin itself. Upstream commit
`1183de7` moved `bin_init_region` from x = 2.0 to x = 2.23 and left `blocks_goal_region`
behind, so the bin came to sit 23 cm past the box that scores. **A cube that landed in
the bin was a scored failure**, and only a throw that missed the bin scored at all;
training against that scene would have rewarded missing. Upstream's own prose
(`docs/envs/Tossing3D.md:8`, "must toss the object into a bin") describes the
pre-`1183de7` scene, and was correct when it was written.

**The fix, and why the workaround it replaced is gone.** This repo used to ship its own
`scripts/task_configs/Tossing3D-o1-coincident.json` — upstream's `o1` with the bin put
back to x = 2.0 — selectable against upstream's through a two-member `Tossing3DTaskConfig`
enum (`STOCK` / `COINCIDENT`). Upstream then fixed it for real, `kindergarden` PR #126,
**by editing `Tossing3D-o1.json` itself rather than adding a variant**. Both enum members
therefore came to load the same scene, and two tests asserting the contrast between them
broke with nobody having edited them. Josh's call was to take upstream's config as the
config rather than vendor a pre-fix scene to keep the comparison alive, so the enum, the
`--task-config` flag and this repo's copy of the scene are all retired.

**The cost, stated rather than left implicit.** `STOCK` meant "whatever the submodule
ships", so its meaning moved with the `reference/kindergarden` pin — which is exactly how
this collapsed unnoticed. Taking upstream's config accepts that coupling for the whole
domain: a pin bump can now change the geometry every measured number was taken under.
`test_the_shipped_scene_still_puts_the_bin_on_the_box_that_scores` reads the installed
KINDER's own task JSON and fails loudly if `bin_init_region` ever comes off
`blocks_goal_region`'s centre again, so the coupling is observable rather than silent.
`test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene` is the live counterpart,
measured off the compiled MuJoCo model after inflation and sampling.

At standoff 1.35, seed 125, on the shipped scene the cube comes to rest at x = 1.9902,
z = 0.0444 — inside the bin, inside the goal box, `_check_goals()` **True**. z = 0.0444
is the bin's interior floor (0.02 m bottom panel plus the cube's 0.025 m half-extent).
On the pre-fix scene the same throw rested at x = 2.2197, also inside the bin, and scored
**False**. **Never compare a number measured before PR #126 against one measured after**:
moving the bin 23 cm changes the flight path, so it changed the physics and not only the
scoring.

## Rewinding: `set_state` versus `snapshot`/`restore`

**`set_state` cannot restore a mid-episode state, and raises rather than pretending.** A
`core.State` here is a *lossy projection* of KINDER's own state — four of the robot's
twenty-two features, six of the cube's sixteen — so there is nothing in it to rebuild an
arm configuration or a velocity from. `set_state` therefore rebuilds the scene from the
seed carried in the state's own `scene` object, and refuses any state with
`steps_taken > 0`. That covers everything the harness actually needs (`reset_to_task`,
`Method.reset_environment`) and refuses everything it does not.

**A genuine rewind does exist, under a different name.** KINDER's `ObjectCentricState`
*is* a full state — it carries velocities and the whole arm configuration, which is why
upstream's own `tidybot3d_shelf3D.py` uses it to build a transition function — and
`ObjectCentricTidyBot3DEnv.set_state` accepts one once the env is constructed with
`allow_state_access=True`. So `Tossing3DEnvironment.snapshot()` / `.restore()` hand back
an opaque handle that really does put MuJoCo back. It is deliberately *not* folded into
`set_state`: `core.Environment.set_state` is documented as the human's privileged
override, and speculatively executing a skill to see whether it does anything is not that.

Restore is faithful **to float32**, not bit-exact: an `ObjectCentricState` is the
observation vector and `ObjectCentricBoxSpace` is float32, while MuJoCo integrates in
float64. Measured over one `move_to_target`, two runs from the same snapshot end up
~1e-7 apart on x and ~2.7e-4 on the cube's quaternion. Four orders of magnitude below
anything a predicate here tests, but a rewound rollout is not byte-reproducible.

## Operator-dynamics fidelity

The repo-wide invariant — *a symbolic operator model must never permit more than the raw
dynamics allow* — is enforced for this domain by
`tests/environments/tossing3d/test_operator_fidelity.py`, which executes every applicable
ground skill speculatively along the oracle's own trajectory and rewinds with
`snapshot`/`restore`.

It is **not** registered in the cross-domain
`tests/environments/test_operator_dynamics_fidelity.py`, for two structural reasons: that
file must stay green on CI, which never installs KINDER, and its budget (8 random walks ×
40 steps, every candidate × 30 draws) is thousands of real MuJoCo rollouts at seconds
each. The narrowing this leaves is stated in the local file: it walks one trajectory
rather than searching, so it cannot find a reachable-but-unvisited symbolic state whose
model is wrong.

**This is not a hypothetical guard.** The throw-pose predicate was first written as a
plain `1.0 <= distance <= 1.8` band. After `Pick` — which drives the base to the *cube*,
off to one side — the base sat 1.76 m from the bin, inside that band, so the oracle
believed it was already at a throw pose, skipped `MoveToThrowPose`, and threw facing 40°
away: the cube landed at (0.9969, −0.7196) and the episode failed. Its lateral conjunct
(on the bin's axis, within upstream's own `WAYPOINT_TOL`) is what rules that out, and is
unchanged since.

**`RobotAtSuccessfulThrowPose` is a success test, not a reachability test, and that is a
correctness fix rather than a rename.** It was `NearBin`, and it accepted every standoff
in `THROW_STANDOFF_BOUNDS` — the interval the sampler draws from. `MoveToThrowPose`'s only
add effect was therefore constant-true, EES's per-skill success classifier saw a single
class, and every draw fell back to uniform: 16/16 attempts labelled success with 0/16
informed draws in a probe run. The standoff conjunct now tests whether the *predicted
landing point* `base_x + THROW_RANGE` falls inside the scored box's own live extent, so
the accepted band `[bin_x + THROW_RANGE − x_max, bin_x + THROW_RANGE − x_min]` is derived
per call and tracks the box wherever it moves.

## The goal region is not a symbolic object

**The symbolic layer assumes the bin's interior *is* the scored region**, so no predicate
and no operator takes a goal region as an argument: `InBin(cube, bin)`,
`RobotAtSuccessfulThrowPose(robot, bin)`, `Pick(robot, cube, barrier, bin)`,
`MoveToThrowPose(robot, cube, bin)`, `Toss(robot, cube, bin, barrier)`.

That is a modelling choice and it is **false in general** — under stock the two are 23 cm
apart, which is exactly the trap the section above describes. What it does *not* change is
fidelity: the box these classifiers test against is still the live `Region.bbox` of
`blocks_goal_region`, carried in the `State` as six extra features on the bin object, so
`InBin` agrees with `_check_goals()` on **both** configs. Only the *symbolic* dependence
went away — an object a planner had to bind and no skill could act on.

Two consequences worth knowing before reading a trace:

- **Under `--task-config stock` the name lies.** The bin's own `x` (2.2305) sits outside
  its own `x_max` (2.15), so `InBin` is true exactly when the cube is *not* in the bin.
  The arithmetic still scores what KINDER scores.
- **Wherever the bin is taken to be movable, moving it moves the goal.** kindergarden#126
  moves the bin, and under this assumption the scored target follows it — "put the cube in
  the bin, wherever the bin is", which is the intended reading of a move-the-bin task
  rather than a problem to work around. The stock config implements the opposite reading:
  a fixed goal the bin merely decorates.

`predicates.py`'s module docstring carries the same statement at source.

## Running it

KINDER installs **into `hitl-pmp` itself**, as the optional `tossing3d` extra — see
CLAUDE.md's `reference/` section for the install, the version ceilings it imposes, the
four environment traps, and the memory cap. (It used to require its own virtualenv; that
split rested on a `requires-python` cap that was measured to exclude neither environment.)

`scripts/with_kinder_env.sh` is a thin alias for `with_env.sh` that adds the
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` pins a reproducible simulator run wants:

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- \
  scripts/with_kinder_env.sh python -m hitl_pmp.cli \
    --env tossing3d --method skill-oracle --num-test-tasks 5 --output-dir /tmp/tossing3d
```

`--output-dir` writes `episode.mp4` (a four-frame storyboard: initial scene, then one
frame after each of `Pick`, `MoveToThrowPose`, `Toss`, each captioned with the cube's
measured position and the `InBin` verdict), plus `stats.json` and
`config_snapshot.json`. For a *smooth* per-tick clip use
`scripts/tossing3d_oracle_demo.py`, which stays separate — one frame per transition is a
property of `core.Renderer`, not of this domain.

The KINDER-backed tests skip cleanly without the simulator (CI still never installs it),
and now run as part of the ordinary local gate. To run just this domain's:

```bash
scripts/with_env.sh python -m pytest tests/environments/tossing3d/ -q
```

## Known gaps

- **No `HumanOracle`.** This is the domain in the repo that most needs one — a tossed
  cube genuinely requires someone to walk over and pick it up — and
  `Metrics.num_human_interventions()` reports `(0.0, 0)` because none is *representable*,
  not because none was needed.
- **`o2` is not supported.** It needs two cubes in the scored region; the symbolic layer
  here is single-cube. `Tossing3DEnvironment.backend()` refuses `--variant o2` outright
  rather than running an under-specified goal.
- **The oracle's weak link is the grasp, not the throw.** `pick_shelf` is marginal on
  some scene seeds — a previously recorded pair, seed 1 losing the cube during the base
  move and seed 3 never releasing it. Per-seed solve counts should be read with that in
  mind.
- **No learning run has been made on this domain in this repo.** Nothing here says
  whether EES or any other method should use it.
