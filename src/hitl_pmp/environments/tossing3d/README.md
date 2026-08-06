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
| `Region.bbox` on `blocks_goal_region` | the goal box, read live and carried in the `State` |
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
  correspondingly *weaker* than upstream's. `InGoalRegion` reproduces `_check_goals()`
  exactly and is differentially tested against it. `Reachable` and `NearBin` are new.
- **The three lifted skills and their operator models** (`skills.py`). KINDER ships
  symbolic models for `Shelf3D`, `Sweep3D` and base motion, but **none for Tossing3D**,
  so the operator layer is written here — following `tidybot3d_shelf3D.py`'s shape, where
  a `LiftedOperator` is paired to one of upstream's controllers.
- **`THROW_STANDOFF_BOUNDS = (0.45, 1.75)`**, the one genuinely new continuous range.
  Upstream's own `MOVE_TO_TARGET_DISTANCE_BOUNDS` is `(0.5, 0.6)` — a *grasping* standoff
  — and upstream's tossing test simply hardcodes `1.35` with no range at all. It used to
  be `(1.20, 1.65)`, the interval `scripts/tossing3d_oracle_demo.py --sweep` happened to
  cover, which is barely wider than the band that solves; both endpoints are now measured
  (see below).
- **The `[skill_id, param_0, param_1]` action encoding**, the `State` schema, and the
  `scene` object that carries the episode seed.
- **The default of the coincident task config** (see below).
- `H_eval = 3 + 2` (`problem.py`).

## The single most misreadable thing about this domain

**On upstream's stock `o1`, a cube that lands in the bin is a scored failure.** The goal
predicate is `["on", "cube_0", "blocks_goal_region"]` — a *ground region* that the bin
merely sits near. Upstream commit `1183de7` moved `bin_init_region` from x = 2.0 to
x = 2.23 and left `blocks_goal_region` behind, so the bin now sits past the box that
scores. Upstream's own prose (`docs/envs/Tossing3D.md:8`, "must toss the object into a
bin") describes the pre-`1183de7` scene.

So the default here is **`--task-config coincident`**:
`scripts/task_configs/Tossing3D-o1-coincident.json`, upstream's own `o1` with
`bin_init_region` put back to x = 2.0 and `blocks_goal_region` left byte-identical.
x = 2.0 is a pairing upstream itself still ships (`Tossing3D-o2.json`), so this is not an
invented scene; and it reverts the edit that caused the drift rather than compensating
for it. Measured live, the two boxes then agree to 0.1 mm.

Training against stock would reward the throw for **missing the bin**, which is why the
default is not merely a preference.

`--task-config stock` stays selectable, and is what every number in
`docs/kinder-environment-validation.md` was measured against. **Never compare a number
taken under one config against one taken under the other**: moving the bin 23 cm nearer
puts it in the flight path, so it changes the physics, not only the scoring. Concretely,
at standoff 1.35, seed 125:

| | coincident (default) | stock |
| --- | --- | --- |
| cube at rest | x = 1.9902, z = 0.0444 | x = 2.2197, z = 0.0444 |
| where that is | inside the bin, inside the goal box | inside the bin, 7 cm past the goal box |
| `_check_goals()` | **True** | **False** |

z = 0.0444 in both cases is the bin's interior floor (0.02 m bottom panel plus the cube's
0.025 m half-extent), i.e. the cube is *in* the bin either way. Only the verdict differs.

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

**This is not a hypothetical guard.** `NearBin` was first written as a plain
`1.0 <= distance <= 1.8` band. After `Pick` — which drives the base to the *cube*, off to
one side — the base sat 1.76 m from the bin, inside that band, so the oracle believed it
was already at a throw pose, skipped `MoveToThrowPose`, and threw facing 40° away: the
cube landed at (0.9969, −0.7196) and the episode failed. `NearBin` now characterises
exactly the pose `move_to_target` produces (on the bin's axis, at a band standoff, within
upstream's own `WAYPOINT_TOL`).

## Running it

KINDER goes in **its own virtualenv**, never `hitl-pmp` — see CLAUDE.md's `reference/`
section for the install, the four environment traps, and the memory cap.

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- \
  /path/to/kinder-venv/bin/python -m hitl_pmp.cli \
    --env tossing3d --method skill-oracle --num-test-tasks 5 --output-dir /tmp/tossing3d
```

`--output-dir` writes `episode.mp4` (a four-frame storyboard: initial scene, then one
frame after each of `Pick`, `MoveToThrowPose`, `Toss`, each captioned with the cube's
measured position and the `InGoalRegion` verdict), plus `stats.json` and
`config_snapshot.json`. For a *smooth* per-tick clip use
`scripts/tossing3d_oracle_demo.py`, which stays separate — one frame per transition is a
property of `core.Renderer`, not of this domain.

The KINDER-backed tests skip cleanly without the simulator (CI never installs it), so run
them under the same venv:

```bash
PYTHONPATH=$(pwd)/src /path/to/kinder-venv/bin/python -m pytest tests/environments/tossing3d/ -q
```

## Known gaps

- **No `HumanOracle`.** This is the domain in the repo that most needs one — a tossed
  cube genuinely requires someone to walk over and pick it up — and
  `Metrics.num_human_interventions()` reports `(0.0, 0)` because none is *representable*,
  not because none was needed.
- **`o2` is not supported.** It needs two cubes in the goal region; the symbolic layer
  here is single-cube. The CLI accepts `--variant o2` because the backend does, but the
  goal would be under-specified, and `--task-config coincident` refuses it outright.
- **The oracle's weak link is the grasp, not the throw.** `pick_shelf` is marginal on
  some scene seeds — a previously recorded pair, seed 1 losing the cube during the base
  move and seed 3 never releasing it. Per-seed solve counts should be read with that in
  mind. (At the coincident config's default standoff the oracle measured 99/100 over
  seeds 0-9 × 10 test tasks, so on this scene the grasp is close to reliable.)
- **What there is to learn is a constant, not a function of state.** `bin_init_region` is
  1 mm wide, so the bin is in the same place every episode; `Toss` has `param_dim = 0`;
  and only `MoveToThrowPose`'s standoff decides success. **Do not read a result on this
  domain as evidence about learning a state-dependent sampler.** Widening
  `bin_init_region` is the change that would make it one.
- **The constant is now hard to find, which it was not before.** `THROW_STANDOFF_BOUNDS`
  used to be `(1.20, 1.65)` — barely wider than the band that solves — and pooled over
  that range the oracle solved **155/330**, so a uniform draw was right about as often as
  not and a learned sampler had almost no headroom over its own prior. The bounds are now
  `(0.45, 1.75)`, 1.30 m wide: the measured feasible range `[0.40, 2.06]`, inset at the
  bottom by `NEAR_BIN_TOLERANCE` and at the top by the pose `Pick` leaves the base in.
  Over five scene seeds at 0.025 m resolution the throw solves **5/5** throughout
  `[1.150, 1.375]` and **0/5** below 1.125 or above 1.425, with soft edges between (2/5 at
  1.125, 3/5 at 1.400, 2/5 at 1.425). So the reliably-solving band is 0.225 m of a 1.30 m
  range. Short standoffs overshoot — the cube lands at x = 2.3–3.0 against a goal box
  ending at 2.15 — and long ones fall short.
- **`RandomSkillsMethod` cannot run here.** After a `Toss` the cube is past the barrier,
  `Reachable` is false, and no ground skill's preconditions hold — the first genuine
  dead-end state in any domain in this repo. `EesMethod` degrades to `no-op (no plan)`;
  `RandomSkillsMethod.get_labeled_action` asserts and the run dies. The paper's own
  lower-bound baseline is therefore unavailable until that is fixed.
- **EES's "no-op" is not inert on this domain.** `EesMethod._noop_action()` returns
  `np.zeros(action_space.shape)` = `[0, 0, 0]`, and `pick_id` is `0`, so a
  `no-op (no plan)` step runs a real `pick_shelf` at distance 0.0. Harmless to the
  symbolic outcome so far, but it burns simulator time and moves the robot.
- **One learning run has now been made**, and it is a null result:
  `docs/experiment-logs/2026-08-06-tossing3d-ees.md`. EES went from 24/90 to 33/90 over
  ~55 online transitions (p = 0.1328, not established) against a 99/100 oracle ceiling.
  **That run predates the widening above** and was taken under `(1.20, 1.65)`, so its
  counts are not reproducible at HEAD and must not be compared against one that is.
