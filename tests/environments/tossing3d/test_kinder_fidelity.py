"""Simulator-backed tests: does this integration actually agree with KINDER?

**Every test in this file skips cleanly without KINDER**, gated on
`importlib.util.find_spec("kinder")` -- the *import* package name, not the distribution
name `kindergarden`. Locally, and on CI since 2026-08-14, KINDER installs into
`hitl-pmp` itself (the `tossing3d` extra), so this runs in the ordinary gate:

    scripts/with_env.sh python -m pytest tests/environments/tossing3d/ -q

and under a memory cap, because these tests execute real controllers:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- ...

What these check that the offline tests structurally cannot:

1. `predicates.IN_BIN` agrees with KINDER's own `_check_goals()`. That is the
   whole basis for this domain trusting its own symbolic layer rather than the simulator.
2. The goal box in the `State` is the live `Region.bbox`, element for element -- the only
   check that can catch a wrong box, which has shipped once already.
3. `predicates.CubeSymmetry.tilt_from_upright` is upstream's own function, computed
   without `scipy` and without the symmetry group. That closed form is what lets the whole
   symbolic layer run on a machine with no KINDER, and it is algebra rather than a port,
   so it is verified rather than asserted.
4. The oracle reproduces the rest position and step counts recorded below.
5. The `Tossing3D-o1.json` the installed KINDER ships still puts the bin on the box that
   scores. This domain no longer commits a scene of its own, so the scene moves with the
   `reference/kindergarden` pin -- and that check is what makes a pin bump a **loud** event
   rather than a silent change of geometry under every number already measured.

## What the two-skill migration removed from this file

Four tests are deleted rather than ported, all of them for the same reason: they measured
a decomposition this domain no longer runs.

- **`test_oracle_pick_parameters_match_upstreams_own_sampler`** checked that
  `ORACLE_PICK_DISTANCE`/`ORACLE_PICK_ROTATION` really were what
  `PickShelfController.sample_parameters` drew from `np.random.default_rng(123)`.
  `PickCube` has `param_dim=0` and derives both internally, so there is nothing left to
  agree about.
- **`test_the_derived_band_agrees_with_whether_the_throw_actually_scores`**,
  **`test_no_toss_parameterisation_scores_from_beyond_the_accepted_band`** and
  **`test_the_converse_guard_would_catch_a_widened_band`** were the three-way calibration
  guard for `RobotAtSuccessfulThrowPose` -- forward, converse, and a proof the converse
  could fail. The predicate and its `THROW_RANGE` calibration are gone with the pose they
  described. The property they protected (that the sampler's range is a range throws
  actually score from) now lives upstream, in the `TARGET_DISTANCE_BOUNDS` that
  `test_kinder_pin.py` pins this package's own copy against.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_BIN, CubeSymmetry, InBinClassifier
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossing3d.skills import TOSS_DISTANCE_BOUNDS
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

# The DISTRIBUTION is `kindergarden`; the IMPORT package is `kinder`. Keying on the
# distribution name here would skip everything, always.
needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)

pytestmark = needs_kinder

# Upstream's own `test_pick_ground_toss` seed, and the seed every number in
# `docs/kinder-environment-validation.md` was measured at.
CANONICAL_SEED = 125

# Where the oracle's cube comes to rest, at `ORACLE_THROW_STANDOFF`,
# `ORACLE_RELEASE_SPEED_DEG_S` and `ORACLE_GRIPPER_RELEASE_MS`.
#
# > **Re-measured 2026-08-16 under the composed controller.** Every earlier value in this
# > file was taken with `Pick` -> `MoveToThrowPose(1.35)` -> `Toss(speed, ms)` driven as
# > three separate controllers from this package. `MoveToTossLocationAndToss` plans the
# > base motion, the windup and the swing together, and the swing therefore starts from
# > the arm configuration its own motion planner reached rather than from a separately
# > commanded windup. **Those earlier numbers are not restated or recomputed here** -- they
# > are correct for what they measured, and are left in git history and in the experiment
# > logs that cite them:
# >
# > | measured | three-skill | composed |
# > | --- | --- | --- |
# > | resting x | 1.9926 | 2.0175 |
# > | controller steps | 71 / 23 / 16 / 18 (four executions) | 121 / 51 (two) |
# >
# > Reproducibility was checked rather than assumed: three independent runs at
# > `CANONICAL_SEED` returned bit-identical values for both.
#
# Resting x is a bin-carom outcome rather than evidence about throw distance -- first
# contact is the bin, and across three seeds of one parameter cell the pre-migration grid
# measured impact x spreading 2.2 mm while resting x spread 216.5 mm. It is pinned here as
# a *reproducibility* check on one seed, not as a claim about the throw.
REST_X = 2.0175
BIN_FLOOR_Z = 0.0444

# What the two controller executions take, at the same operating point and re-measured in
# the same run. Pinned so that a change in either controller's termination condition is a
# loud event: a controller that silently stopped terminating would otherwise show up only
# as a slower suite.
ORACLE_CONTROLLER_STEPS = [121, 51]

# Two standoffs from inside upstream's own `TARGET_DISTANCE_BOUNDS`, measured 2026-08-16
# at `CANONICAL_SEED` to land on opposite sides of the goal box: 1.25 rests the cube at
# x = 2.1057 (inside) and 1.45 at x = 1.6901 (short). Both ends of the range the sampler
# draws from, so a test that needs a scoring rollout and a missing one needs no standoff
# this domain could not itself have drawn.
STANDOFF_THAT_SCORES = TOSS_DISTANCE_BOUNDS[0]
STANDOFF_THAT_MISSES = TOSS_DISTANCE_BOUNDS[1]


def _env():
    return Tossing3DEnvironment()


def _run_oracle(*, env, standoff: float | None = None):
    """Drive the oracle's whole two-skill plan and return the final state.

    The task is built ONCE, before the rollout: `build_task` rebuilds the scene (see
    `Tossing3DTasks`' docstring), so building it inside the loop would silently reset the
    episode between every step.
    """
    goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
    state = env.reset_to_seed(seed=CANONICAL_SEED)
    kwargs = {} if standoff is None else {"throw_standoff": standoff}
    for _ in range(2):
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal, **kwargs)
        state = env.take_action(action=action.action)
    return state


@pytest.mark.parametrize("standoff", [STANDOFF_THAT_SCORES, STANDOFF_THAT_MISSES])
def test_in_bin_agrees_with_kinders_own_goal_check_on_the_oracles_trajectory(
    *, standoff: float
) -> None:
    """The differential test this domain's trust rests on, checked at every step of a real
    oracle episode.

    Both verdicts flip during the episode -- `False` while the cube is on the floor and in
    the gripper, `True` once it lands -- so agreeing at every step is a real constraint and
    not something a predicate hardwired to `True` would pass. The two standoffs are the
    ends of upstream's own distance bounds and were measured to land on opposite sides of
    the goal box, so the pair also exercises agreement on a `False` final verdict."""
    env = _env()
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        for _ in range(2):
            symbolic = InBinClassifier.holds(state=state, cube=env.cube, target=env.bin)
            assert symbolic == env.is_solved(), (
                f"standoff {standoff}: InBin said {symbolic} while KINDER's "
                f"own _check_goals() said {env.is_solved()}"
            )
            action = SkillOraclePolicy.get_labeled_action(
                state=state, env=env, goal=goal, throw_standoff=standoff
            )
            state = env.take_action(action=action.action)
        assert InBinClassifier.holds(state=state, cube=env.cube, target=env.bin) == env.is_solved()
    finally:
        env.close()


def test_the_two_probe_standoffs_really_do_land_on_opposite_sides_of_the_box() -> None:
    """**The test above is only worth anything if its two cases differ**, and a negative
    -- "they agree" -- passes just as well when both rollouts miss, or both score, as when
    one does each. Asserted separately so that a change making both standoffs behave alike
    fails here, naming the reason, rather than quietly halving the agreement test's
    coverage.

    Which standoff does which is measured, not derived: nothing in the symbolic layer
    predicts it any more, which is exactly what retiring `RobotAtSuccessfulThrowPose`
    gave up."""
    outcomes = {}
    for standoff in (STANDOFF_THAT_SCORES, STANDOFF_THAT_MISSES):
        env = _env()
        try:
            _run_oracle(env=env, standoff=standoff)
            outcomes[standoff] = bool(env.is_solved())
        finally:
            env.close()

    assert outcomes[STANDOFF_THAT_SCORES] is True
    assert outcomes[STANDOFF_THAT_MISSES] is False


def test_the_goal_box_in_the_state_is_the_live_region_bbox_element_for_element() -> None:
    """The only check that can catch a wrong goal box. Pinned against `Region.bbox` read
    back off the compiled model, never against the task JSON's literal -- the JSON range
    is inflated by 0.05 m per side before it becomes a region, and a predicate written
    against the literal is 2/3 of the true width on x."""
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        live = env.backend().goal_region_bbox()
        corners = ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
        in_state = tuple(state.get(obj=env.bin, feature_name=name) for name in corners)
        assert in_state == pytest.approx(live)
        # And it is the inflated box, not the JSON's [1.90, 2.10].
        assert live[0] == pytest.approx(1.85, abs=1e-6)
        assert live[3] == pytest.approx(2.15, abs=1e-6)
    finally:
        env.close()


def test_the_closed_form_cube_tilt_agrees_with_upstreams_symmetry_group() -> None:
    """**The algebra `predicates.CubeSymmetry` rests on, checked rather than argued.**

    Upstream composes the measured rotation with all 24 rotations that map a cube onto
    itself (`scipy`'s octahedral group) and takes the smallest resulting `qx^2 + qy^2`.
    That quantity is `(1 - R[2, 2]) / 2` for the composed rotation, and composing with the
    group sweeps the composed matrix's third column over the six signed body axes -- so
    the minimum is `(1 - max_i |R[2, i]|) / 2`, in closed form, with no group and no
    `scipy`. Writing it that way is what lets the whole symbolic layer -- and therefore
    every offline test in this directory -- run on a machine with no KINDER.

    A closed form derived by hand is exactly the kind of thing that is right for the cases
    someone thought of, so this checks it over 200 uniformly random rotations rather than
    over hand-picked poses, from a fixed-seed generator so a failure is reproducible.
    Compared to floating point (`rel=1e-9`), not to a tolerance chosen to make it pass:
    the two are the same number, and any real disagreement is a wrong derivation rather
    than accumulated error.
    """
    from kinder_models.dynamic3d.cube_symmetry import cube_tilt_from_upright

    rng = np.random.default_rng(20260816)
    # Uniform on the 3-sphere, so the sample covers the whole rotation group rather than
    # clustering near the identity the way per-axis Euler draws would.
    quaternions = rng.normal(size=(200, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    for quaternion in quaternions:
        rotation = tuple(float(value) for value in quaternion)
        assert CubeSymmetry.tilt_from_upright(rotation=rotation) == pytest.approx(
            cube_tilt_from_upright(rotation), rel=1e-9, abs=1e-12
        )


def test_the_closed_form_cube_tilt_agrees_with_upstream_on_the_face_down_rests() -> None:
    """The random sample above almost surely never draws a face-down rest, which is the
    one family the classifier's verdict actually turns on -- and the one where both sides
    return zero, so a broken derivation could still agree there by accident on the random
    draws while disagreeing here. Probed directly, at each of the six faces."""
    from kinder_models.dynamic3d.cube_symmetry import cube_tilt_from_upright

    half = float(np.sqrt(0.5))
    face_down = (
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0),
        (half, 0.0, 0.0, half),
        (-half, 0.0, 0.0, half),
        (0.0, half, 0.0, half),
        (0.0, -half, 0.0, half),
    )
    for rotation in face_down:
        ours = CubeSymmetry.tilt_from_upright(rotation=rotation)
        assert ours == pytest.approx(cube_tilt_from_upright(rotation), abs=1e-12)
        assert ours == pytest.approx(0.0, abs=1e-12)


def test_the_oracle_reproduces_the_recorded_landing_and_step_counts() -> None:
    """The reference numbers, **re-measured 2026-08-16 under the composed controller**:
    the cube comes to rest at x = 2.0175, z = 0.0444 (the bin's interior floor, i.e. the
    cube is *in* the bin), `_check_goals()` True, and the two controller executions
    terminating in 121 / 51 steps.

    See `REST_X`' own note for what the three-skill decomposition measured and why those
    values are left as published rather than replaced. Both numbers were reproduced
    bit-identically across three independent runs before being pinned here."""
    env = _env()
    try:
        # Built ONCE, before the rollout: `build_task` rebuilds the scene.
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        steps: list[int] = []
        for _ in range(2):
            action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
            state = env.take_action(action=action.action)
            steps.extend(env.last_controller_steps())
            assert env.last_skill_error() is None, env.last_skill_error()

        assert steps == ORACLE_CONTROLLER_STEPS
        assert state.get(obj=env.cube, feature_name="x") == pytest.approx(REST_X, abs=1e-3)
        assert state.get(obj=env.cube, feature_name="z") == pytest.approx(BIN_FLOOR_Z, abs=1e-3)
        assert env.is_solved()
        assert IN_BIN.holds(state, (env.cube, env.bin))
    finally:
        env.close()


def test_the_oracle_solves_in_exactly_two_controller_executions() -> None:
    """One execution per skill, and two skills. Stated apart from the step counts above
    because it is the claim that survives a retune: a decomposition that quietly went back
    to driving a windup and a swing separately would keep landing the cube in the bin and
    would report four executions here."""
    env = _env()
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        per_skill = []
        for _ in range(2):
            action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
            state = env.take_action(action=action.action)
            per_skill.append(len(env.last_controller_steps()))
        assert per_skill == [1, 1]
    finally:
        env.close()


def test_the_composed_toss_at_the_shortest_standoff_does_not_disturb_the_barrier() -> None:
    """**The regression guard for a defect that shipped once.** `cuboid_barrier` is a
    real dynamic MuJoCo body rather than a static collision-free waypoint check, and base
    motion planning had `obstacle_geoms` hardcoded empty upstream -- so a short enough
    standoff drove the base straight through the barrier and knocked it over. The old
    `THROW_STANDOFF_BOUNDS` floor of 1.10 m existed to clear the worst colliding standoff
    (1.00 m, measured three ways) by a 0.10 m margin.

    **That floor is gone with the skill that carried it**, and the shortest standoff this
    domain can now draw is `TOSS_DISTANCE_BOUNDS[0]` -- upstream's own, and further out.
    The hazard is upstream's to have designed around, but the consequence of getting it
    wrong is still ours, so the property is re-asserted at whatever the current floor is
    rather than deleted along with the constant.

    Runs the oracle's own pick, then the composed toss at exactly `TOSS_DISTANCE_BOUNDS[0]`,
    so this reads the bound rather than hardcoding it."""
    env = _env()
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        barrier_before = tuple(
            state.get(obj=env.barrier, feature_name=name) for name in ("x", "y", "z")
        )

        # PickCube, which takes no parameters at all.
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
        state = env.take_action(action=action.action)
        assert env.last_skill_error() is None, env.last_skill_error()

        # The composed toss at exactly the sampler's shortest standoff.
        action = SkillOraclePolicy.get_labeled_action(
            state=state, env=env, goal=goal, throw_standoff=TOSS_DISTANCE_BOUNDS[0]
        )
        state = env.take_action(action=action.action)
        assert env.last_skill_error() is None, env.last_skill_error()

        barrier_after = tuple(
            state.get(obj=env.barrier, feature_name=name) for name in ("x", "y", "z")
        )
        assert barrier_after == pytest.approx(barrier_before, abs=1e-4), (
            f"MoveToTossLocationAndToss(distance={TOSS_DISTANCE_BOUNDS[0]}) moved "
            f"cuboid_barrier from {barrier_before} to {barrier_after} -- the base drove "
            "through it"
        )
    finally:
        env.close()


def test_the_shipped_scene_still_puts_the_bin_on_the_box_that_scores() -> None:
    """**The guard that makes a `reference/kindergarden` pin bump loud instead of silent.**

    This domain no longer commits a scene of its own: it runs whatever
    `Tossing3D-o1.json` the installed KINDER registers. That is the decision, and it has
    a cost worth naming rather than leaving implicit -- **the scene now moves with the
    pin.** That coupling is precisely the defect that produced this test's existence: the
    retired `Tossing3DTaskConfig.STOCK` meant "whatever the submodule ships", so its
    meaning moved when the pin moved, and two tests broke with nobody having edited them.

    Accepting the coupling therefore requires making it observable, which is this test.
    It reads the task JSON out of the KINDER that is actually **installed** -- located
    through the import system rather than a hardcoded path, so it is the tree that is
    *run*, not a second checkout that might sit at a different commit -- and pins the one
    property the domain cannot survive losing: `bin_init_region` is centred on
    `blocks_goal_region`'s centre, i.e. the bin sits on the box that scores.

    **Why that property and not the file's bytes.** A byte or hash comparison would fail
    on every pin bump, including ones that change nothing about the geometry, and a test
    that cries wolf gets deleted. The form this refuses is the one that broke the domain:
    upstream `1183de7` moved the bin to x = 2.23 and left the region behind, putting the
    two centres 230 mm apart.

    **Why the tolerance is 5 mm rather than exact, measured rather than chosen.** The two
    known-good forms are not identical, and the difference is instructive. The pin this
    repo recorded before kg#126 merged declared `[[2.0, -0.0005, 2.001, 0.0005]]` -- a
    1 mm-wide sampling range running from 2.0 to 2.001, whose **mean is 2.0005**, i.e.
    0.5 mm past the goal region's centre. kg#126 as **merged** tightened it to
    `[[2.0, 0.0, 2.0, 0.0]]`, zero-width and exactly on centre. So bumping the pin onto
    the merged commit shifts the bin's mean position by 0.5 mm: tiny, but a real dynamics
    change rather than a formatting one, and worth saying out loud because numbers in this
    file were measured under the jitter form. 5 mm admits both, matches
    `test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene`'s live tolerance,
    and excludes 230 mm by a factor of 46.

    The width bound is a second, independent thing worth pinning: it keeps the bin's
    placement effectively deterministic, which is what lets a single scene seed reproduce
    a landing position. Both known forms are <= 1 mm wide.

    Offline as far as the geometry goes -- it reads the JSON, not a compiled model --
    but it needs KINDER installed to find the file, so it lives here with the rest of the
    simulator-gated tests. `test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene`
    is the live counterpart, measured off MuJoCo after inflation and sampling.
    """
    import json

    import kinder

    task_json = (
        Path(kinder.__file__).resolve().parent
        / "envs"
        / "dynamic3d"
        / "tasks"
        / "Tossing3D"
        / "Tossing3D-o1.json"
    )
    assert task_json.is_file(), f"the installed KINDER has no o1 task JSON at {task_json}"
    regions = json.loads(task_json.read_text())["regions"]

    (bin_range,) = regions["bin_init_region"]["ranges"]
    bin_x_start, bin_x_end = bin_range[0], bin_range[2]
    bin_centre = (bin_x_start + bin_x_end) / 2

    (goal_range,) = regions["blocks_goal_region"]["ranges"]
    goal_centre = (goal_range[0] + goal_range[3]) / 2

    assert bin_centre == pytest.approx(goal_centre, abs=0.005), (
        f"{task_json} declares bin_init_region centred at x = {bin_centre} against "
        f"blocks_goal_region centred at x = {goal_centre}. The bin has come off the box "
        "that scores, which is the pre-PR-126 defect: a cube landing IN the bin would "
        "score a FAILURE, and training on this scene would reward missing it. The pin "
        "moved the scene -- decide whether that is wanted before trusting any number "
        "measured after this point."
    )
    assert bin_x_end - bin_x_start <= 0.001, (
        f"{task_json} samples the bin over {bin_x_end - bin_x_start} m on x. Above about "
        "a millimetre the bin's position stops being effectively determined by the scene "
        "seed, and per-seed landing positions recorded in this file stop reproducing."
    )


def test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene() -> None:
    """**The invariant the whole domain rests on, re-measured live rather than asserted
    about JSON:** the bin sits on the box that scores, so "the cube is in the bin" and
    "the cube is in the goal region" are the same event.

    Measured rather than read off the file because neither box is where the file says it
    is -- `blocks_goal_region` is inflated by 0.05 m per side before it becomes a region,
    and the bin's placement is sampled from a sub-millimetre range.

    **This is the fixed state of a scene that used to be broken, and the gap is the whole
    story.** Upstream commit `1183de7` moved `bin_init_region` to x = 2.23 and left
    `blocks_goal_region` behind, putting the bin ~230 mm off the region: a cube landing
    *in* the bin scored a **failure**, and training against it would have rewarded
    missing. `kindergarden` PR #126 put the bin back to x = 2.0. The tolerance below is
    5 mm against a measured ~0.1 mm, and 230 mm is the value it exists to exclude -- so
    this test is red on the pre-fix scene without needing one to compare against.
    """
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        # The scored box rides on the bin object now that the goal region is not modelled
        # separately, but it is still `blocks_goal_region`'s live bbox rather than the
        # bin's own geometry -- so this stays a comparison of two independent sources.
        goal_min = state.get(obj=env.bin, feature_name="x_min")
        goal_max = state.get(obj=env.bin, feature_name="x_max")
        bin_x = state.get(obj=env.bin, feature_name="x")
        centre = (goal_min + goal_max) / 2
        gap = abs(bin_x - centre)
        assert gap < 0.005, (
            f"the bin is {gap} m off the goal region's centre. At ~0.23 m this is the "
            "pre-PR-126 scene, in which a cube landing in the bin scores a failure."
        )
    finally:
        env.close()


def test_the_thrown_cube_really_comes_to_rest_on_one_of_its_faces() -> None:
    """**The evidence behind the composed toss's `OnGround` add effect.** kb#113's
    operator model records `15/15` scoring throws leaving the cube resting on a face, and
    this package added `OnGround` to the toss's add effects on the strength of that. An
    add effect that read false after a scoring throw would have EES label a throw that
    scored a failure, so the claim is worth one live rollout.

    Deliberately checks the *tilt* rather than the classifier, so it says which of the two
    conjuncts holds: a cube in the bin is off the floor by the bin's own interior height,
    which is what `ON_GROUND_TOL` has to absorb, and reading only the classifier's verdict
    would not distinguish "landed flat" from "landed inside the height tolerance"."""
    env = _env()
    try:
        state = _run_oracle(env=env)
        assert env.is_solved()
        rotation = tuple(
            state.get(obj=env.cube, feature_name=name) for name in ("qx", "qy", "qz", "qw")
        )
        from hitl_pmp.environments.tossing3d.predicates import ON_GROUND_TOL

        assert CubeSymmetry.tilt_from_upright(rotation=rotation) < ON_GROUND_TOL
    finally:
        env.close()


def test_a_full_episode_through_the_problem_solves_the_default_scene() -> None:
    """End to end through the harness's own path: `run_task_episode` with the oracle
    policy, on the canonical scene."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

        # core.method.types.Policy is a positional Callable[[State], LabeledAction].
        def policy(state) -> LabeledAction:  # noqa: PLR0917
            return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

        solved, frames, _ = problem.run_task_episode(task=task, policy=policy)
        assert solved
        assert frames == []
    finally:
        env.close()


def test_the_renderer_produces_a_frame_whose_dimensions_ffmpeg_accepts() -> None:
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        frame = Tossing3DRenderer.render_frame(state=state, env=env, label=None)
        assert frame.dtype.name == "uint8"
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape[0] % 16 == 0
        assert frame.shape[1] % 16 == 0
        assert frame.shape[0] == 480 + Tossing3DRenderer.caption_height
    finally:
        env.close()


def test_render_returns_a_copy_not_a_view_into_mujocos_reused_buffer() -> None:
    """`np.asarray` does not copy an already-uint8 array, so without an explicit copy this
    hands back a view into the render context's reused buffer -- harmless one frame at a
    time, silently wrong the moment frames accumulate into a clip."""
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        first = env.backend().render()
        before = first.copy()
        env.backend().render()
        assert np.array_equal(first, before)
    finally:
        env.close()


def test_task_sampling_draws_distinct_scenes_for_train_and_test() -> None:
    """A test task must be a scene the method never practiced on. The split is a split of
    scene seeds, drawn from two streams derived from `seed`."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        train_seed = Tossing3DTasks.draw_scene_seed(rng=tasks.train_rng)
        test_seed = Tossing3DTasks.draw_scene_seed(rng=tasks.test_rng)
        assert train_seed != test_seed
        task = tasks.build_task(scene_seed=train_seed)
        assert task.goal.describe() == "InBin(cube_0, bin_0)"
        assert not task.goal.is_satisfied(state=task.initial_state)
    finally:
        env.close()


def test_set_state_rebuilds_the_scene_so_reset_to_task_really_rewinds() -> None:
    """`PracticeLoop` and `run_task_episode` both restore a task's initial state before
    running it. In this domain that is a genuine simulator rebuild, not a field
    assignment, so it is worth proving that the cube really goes back."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)
        start_x = task.initial_state.get(obj=env.cube, feature_name="x")

        state = env.reset_to_seed(seed=CANONICAL_SEED)
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)
        moved = env.take_action(action=action.action)
        assert moved.get(obj=env.cube, feature_name="z") > 0.1, "the grasp should have lifted it"

        env.set_state(state=task.initial_state)
        assert env.get_current_state().get(obj=env.cube, feature_name="x") == pytest.approx(
            start_x, abs=1e-6
        )
    finally:
        env.close()


def test_a_recorded_episode_is_physics_rate_rather_than_one_frame_per_skill() -> None:
    """The frame count is the whole claim. One `take_action` here is a whole controller
    execution -- tens of MuJoCo ticks -- so rendering once per decision gave a three-frame
    `episode.mp4` of a domain that exists to show a throw. With the gymnasium
    `RenderCollection` wrapper in place, every one of those ticks reaches the clip.

    Asserted as a floor rather than an exact number: the oracle's controllers terminate
    on their own conditions (`ORACLE_CONTROLLER_STEPS` at the canonical seed), and pinning
    the total would make this a second, fragile copy of the step-count test above.
    """
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

        def policy(state) -> LabeledAction:  # noqa: PLR0917
            return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

        solved, frames, _ = problem.run_task_episode(
            task=task, policy=policy, renderer=Tossing3DRenderer
        )

        assert solved
        assert len(frames) > 100, f"expected a physics-rate clip, got {len(frames)} frames"
        assert len({frame.shape for frame in frames}) == 1, "ffmpeg needs one frame size"
        # Recording is per-episode and must not leak into the next, unrendered one.
        assert env.backend().record_substeps is False
    finally:
        env.close()


def test_recording_does_not_change_where_the_cube_comes_to_rest() -> None:
    """Presentation-only, proved rather than asserted in prose: the same episode run with
    and without the recording wrapper has to land the cube in the same place. A wrapper
    that stepped the simulator, or a per-tick render that perturbed it, would show up
    here and nowhere else."""
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    rest = {}
    for label, renderer in (("plain", None), ("recorded", Tossing3DRenderer)):
        env = _env()
        try:
            tasks = Tossing3DTasks(env=env, seed=0)
            problem = Tossing3DProblem(env=env, tasks=tasks)
            task = tasks.build_task(scene_seed=CANONICAL_SEED)

            def policy(state, task=task, env=env) -> LabeledAction:  # noqa: PLR0917
                return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

            problem.run_task_episode(task=task, policy=policy, renderer=renderer)
            final = env.get_current_state()
            rest[label] = (
                final.get(obj=env.cube, feature_name="x"),
                final.get(obj=env.cube, feature_name="z"),
            )
        finally:
            env.close()

    assert rest["plain"][0] == pytest.approx(REST_X, abs=1e-3)
    assert rest["recorded"] == pytest.approx(rest["plain"], abs=1e-6)
