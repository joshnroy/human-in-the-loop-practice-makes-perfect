# `tossing3d/` — KINDER's Tossing3D, integrated

[Environment page](https://prpl-group.com/kinder-site/environments/tossing3d/index.html) ·
[benchmark](https://prpl-group.com/kinder-site/) ·
[source](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)

A TidyBot++ mobile manipulator has to get a cube from the floor into a goal region on
the far side of an immovable barrier. The barrier is 5 m wide and blocks the base, so
the cube can only get there through the air: the robot must **toss** it.

## This is an integration, not a port

Unlike every other domain here, **no dynamics are written in this repo**. The simulator
is KINDER's own: `kinder_backend.py` calls `kinder.register_all_environments()` and
`kinder.make("kinder/Tossing3D-<variant>-v0")`, and every skill is one of KINDER's own
parameterized controllers — `pick_shelf` from `kinder_models.dynamic3d.shelf`, and
`move_to_target` / `move_arm_to_conf` / `toss` from `kinder_models.dynamic3d.tossing`.
`environment.py` implements no physics; it delegates to the backend and assembles a
`State` from what the backend reads back.

What lives here is **adapter code**, and only adapter code. This repo's `Method`s
consume `core.Environment`, `core.Skill` and `core.Predicate`; KINDER exposes a Gym
env, an `ObjectCentricState` and imperative controllers. Everything in this folder
exists to map the second onto the first:

| KINDER gives us | this folder maps it to | why the mapping is needed |
| --- | --- | --- |
| a Gym `env.step(action_vector)` | `environment.py`'s `[skill_id, param0, param1]` encoding | one transition here is one *skill*, so a `Method` chooses skills, not 20 ms control ticks |
| an `ObjectCentricState` | `environment.py`'s flat `State` over five `Object`s | predicates in this repo are pure functions of `State`, with no simulator handle |
| `_check_goals()`, a method on the env | `predicates.py`'s `InGoalRegion` and friends | Fast Downward needs a symbolic layer it can evaluate off-simulator |
| imperative controllers with `sample_parameters` | `skills.py`'s lifted `Skill`s with pre/add/delete effects | EES task-plans over operators; a controller is not one |
| an episode seed | `tasks.py`'s `Task` (a seed plus a `Goal`) | the harness needs a train/test split and a restorable initial state |
| — | `problem.py` | the `Problem` facade and `H_eval = 3 + 2` |

The glue is the reviewable content. The physics, the controllers and the success
criterion are all upstream's, used verbatim.

## Why this domain

**A tossed cube cannot be retrieved.** Whether the toss succeeds or misses, the cube
ends up past the barrier and no skill brings it back. This is the concrete case the
project's V1 proposal names as EES's predicted failure mode — *"cannot reset the
environment under a suboptimal policy... when the goal is reached it can't reset, even
though it did everything right."*

**Read the experiment log before reading any learning curve measured here.**
`PracticeLoop` resets to the sampled train task at the start of every interaction
period, and `run_task_episode` resets per evaluation episode. That is faithful to
predicators and correct for a reproduction, but it hands out a free reset every
`--max-steps-per-interaction` steps — i.e. exactly the resets the irreversibility
hypothesis is about removing. A curve measured here is a *reproduction* result. It
cannot speak to the hypothesis in either direction.

## Installing the optional dependency

KINDER is the **`tossing3d` extra** in `pyproject.toml` — declared, and pinned to exact
upstream commits, but deliberately not a core dependency: it pulls MuJoCo, PyBullet and
OpenCV, none of which the rest of this repo needs, and `kindergarden` caps
`requires-python` at `<3.13` while this project sets no upper bound.
`kinder_backend.py` imports it lazily, so everything else here imports, typechecks and
tests without it — CI never installs the extra, and the eight simulator tests in
`test_kinder_fidelity.py` skip there.

Install it into a **separate environment**, not on top of the `hitl-pmp` dev env — the
extra's job is to record and pin the versions, not to be layered into the environment
every other domain runs in:

```bash
python -m venv kinder-venv && . kinder-venv/bin/activate
pip install -e "/path/to/this/repo[tossing3d]"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Two repos, two pins, because KINDER is a live upstream that moves underneath this
integration — and since the physics and the controllers are all upstream's, the version
*is* the experiment. The leaked PyBullet client `KinderBackend._release` works around is
an *upstream* bug, so a version that fixes or reshapes it would silently change what a
number measured here means:

| package | repo | pin |
| --- | --- | --- |
| `kindergarden` | [`kindergarden`](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden) | `39eb7e08` (2026-07-28) — upstream `main` when this adapter was written |
| `kinder_models` | [`kinder-baselines`](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines), subdirectory `kinder-models` | `4c731dc8` (2026-06-29) |

Both SHAs are **inferred, not recorded** — nothing wrote down the KINDER commit this was
built against. `39eb7e08` was upstream `main` for the whole window the adapter was
written in; the next upstream commit landed between that and the EES sweep, and is
excluded on content (it changes cluttered-retrieval sampling, which Tossing3D does not
use) rather than on timing. `kinder-baselines` has not moved since 2026-06-29. Correct
these if you know better — and if you re-measure against a different KINDER, move the
pin and say so in the experiment log.

`kinder_models` is the one that holds the parameterized controllers this domain drives
(`dynamic3d.shelf`, `dynamic3d.tossing`); they are not part of `kindergarden` itself,
and it pulls `bilevel_planning` in turn. To hack on either, clone and install over the
top:

```bash
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines.git
pip install -e kindergarden -e kinder-baselines/kinder-models
```

`MUJOCO_GL=egl` matters: `kinder.register_all_environments()` forces `osmesa` whenever
`DISPLAY` is unset, which is the case under `scripts/run_sweep.py`.
`KinderBackend._ensure_env` snapshots and restores the caller's choice, so the exported
value wins — but it has to be exported.

## Layout

| file | what it is |
| --- | --- |
| `kinder_backend.py` | **The only module that talks to KINDER/MuJoCo**, and it imports it lazily, so importing this package is always safe. Runs one KINDER controller per call to termination. |
| `environment.py` | The `State` layout, the `[skill_id, param0, param1]` action encoding, and `set_state`'s seed contract. |
| `predicates.py` | `InGoalRegion`, `HandEmpty`, `Holding`, `Reachable`, `AtThrowPose` — pure functions of `State`. |
| `skills.py` | `Pick` (2 params), `MoveToThrowPose` (0), `Toss` (1) as lifted `Skill`s for Fast Downward. |
| `tasks.py` | A task *is* a KINDER episode seed; the initial state is read back from a real reset. |
| `problem.py` | `H_eval = 3 + 2`. |
| `skill_oracle_policy.py` | The privileged solve, with a measured swing constant. |
| `renderer.py` | KINDER's `task_view` camera, one frame per skill (a storyboard, not a smooth video). |
| `cli.py` | `python -m hitl_pmp.cli --env tossing3d --method ees ...` |

Everything except `kinder_backend.py` is pure arithmetic over feature vectors and is
covered by ordinary CI tests. `tests/environments/tossing3d/test_kinder_fidelity.py`
holds the tests that genuinely drive the simulator; those skip without the optional
dependency. They test the **integration boundary**, never KINDER itself: that
`InGoalRegion` returns exactly what upstream's `_check_goals()` returns on every state
a random walk of skills visits, that `take_action` stays total when upstream's inverse
kinematics has no solution, that resetting to a seed is deterministic enough for
`set_state`'s contract, and that every swing the sampler's prior can draw is one the
upstream controllers accept. None of them reimplement anything upstream already does.

## Three things about this domain that look like bugs and are not

**The goal region is not the bin.** In the `o1` variant `blocks_goal_region` is
x ∈ [1.90, 2.10] while `bin_init_region` puts the bin at x = 2.2305 — a 0.30 m bin, so
its footprint is x ∈ [2.08, 2.38]. A toss hard enough to land *in* the bin therefore
fails KINDER's own goal check, and one that stops short of it in the region passes.
This domain uses KINDER's criterion verbatim rather than substituting an "in the bin"
test, so that a number reported here is a number about the benchmark. It is also what
makes the swing dial worth learning: KINDER's own demo toss (swing = 1.0) overshoots.

**One transition is one skill, not one control tick.** A `take_action` runs a KINDER
controller for a few hundred MuJoCo steps. This matches every other domain here, and it
is what makes "online transitions" comparable across domains.

**`terminated` is ignored.** KINDER reports termination the instant its goal check
passes. An interaction period runs its full length regardless — a solved state is
absorbing (the cube is past the barrier), and ending early would make a solved period
cheaper in transitions than a failed one, biasing the x-axis of every learning curve.

## Known limitations

- **`o2` is not supported.** It requires two cubes in the goal region; the symbolic
  layer here is single-cube. The CLI accepts `--variant o2` because the backend does,
  but the goal would be under-specified.
- **Episodes are deterministic given the whole run's history, not per-episode.**
  Resetting to a seed reproduces the initial state bit-for-bit (pinned by
  `test_reset_to_the_same_seed_reproduces_the_same_initial_state`), but a *marginal*
  grasp can still flip on residual MuJoCo solver state left by whatever the simulator
  did before. A run with a fixed `--seed` reproduces; the same episode embedded in a
  different run may not.
- **No `HumanOracle`.** The domain is the one in this repo that most needs one. Until
  it exists, `Metrics.num_human_interventions()` reports `(0.0, 0)` here as everywhere
  else — not because no intervention was needed, but because none is representable.
