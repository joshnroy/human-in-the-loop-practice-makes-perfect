# `tossing3d/` — a port of KINDER's Tossing3D

[Environment page](https://prpl-group.com/kinder-site/environments/tossing3d/index.html) ·
[benchmark](https://prpl-group.com/kinder-site/) ·
[source](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)

A TidyBot++ mobile manipulator has to get a cube from the floor into a goal region on
the far side of an immovable barrier. The barrier is 5 m wide and blocks the base, so
the cube can only get there through the air: the robot must **toss** it.

Unlike every other domain here, this one is not a self-contained simulator written in
this repo — it drives the real KINDER MuJoCo environment and KINDER's own parameterized
controllers.

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

`kindergarden` is deliberately **not** in `pyproject.toml`: it pulls MuJoCo, PyBullet,
`relational_structs`, `prpl_utils` and a numpy pin, none of which the rest of this repo
needs. Install it into a separate environment alongside this package:

```bash
python -m venv kinder-venv && . kinder-venv/bin/activate
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kinder-models.git
pip install -e kindergarden -e kinder-models/kinder-models bilevel_planning
pip install torch pydantic          # what hitl_pmp itself needs
pip install --no-deps -e /path/to/this/repo
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
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
holds the tests that genuinely drive the simulator, including the property test that
pins `InGoalRegion` against KINDER's own `_check_goals`; those skip without the
optional dependency.

## Three things about this port that look like bugs and are not

**The goal region is not the bin.** In the `o1` variant `blocks_goal_region` is
x ∈ [1.90, 2.10] while `bin_init_region` puts the bin at x = 2.2305 — a 0.30 m bin, so
its footprint is x ∈ [2.08, 2.38]. A toss hard enough to land *in* the bin therefore
fails KINDER's own goal check, and one that stops short of it in the region passes.
This port uses KINDER's criterion verbatim rather than substituting an "in the bin"
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
