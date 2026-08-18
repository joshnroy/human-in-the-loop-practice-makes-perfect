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
3. The two upstream classifiers that **cannot** be evaluated offline really do what the
   swap claimed: `Holding`'s forward-kinematics conjunct rejects a cube that is closed on
   but nowhere near the gripper, and `MovableInGoalRegion` agrees with `_check_goals()`
   at both edges of the scored box. Those are the two probes that had to move here when
   the classifiers became upstream's.
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
from hitl_pmp.environments.tossing3d.predicates import HOLDING, IN_BIN
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
# Upstream inflates a ground region by this much per side before it becomes a
# `Region` (`kinder/envs/dynamic3d/objects/base.py`,
# `PhysicsObject.ground_placement_threshold`). Restated rather than imported:
# upstream sets it as an instance attribute on a literal, so there is no module
# constant to read. The live goal-box test measures the real box against this,
# so a change upstream fails there rather than going unnoticed.
GROUND_PLACEMENT_THRESHOLD = 0.05

# > **Third and fourth rows, 2026-08-18, at the bumped pins.** The rows above are
# > left exactly as published; this branch bumps `reference/kindergarden` to a scene
# > whose `blocks_goal_region` is `[2.00, 2.05]` rather than `[1.90, 2.10]`, and
# > `reference/kinder-baselines` to the rung whose `pick_cube` plans its own base
# > motion. Nothing below is recomputed from anything above it -- every column is a
# > rollout that was actually run:
# >
# > | measured | three-skill | composed | kb `f12326c` | kb `2ccc7e6` |
# > | --- | --- | --- | --- | --- |
# > | resting x | 1.9926 | 2.0175 | 2.0204 | 2.0202 |
# > | controller steps | 71 / 23 / 16 / 18 | 121 / 51 | 187 / 51 | 208 / 51 |
# >
# > The changes in resting x are a carom outcome and not evidence about the throw
# > (see the note below). The pick's step count rose 121 -> 187 because the composed
# > rung plans its own base motion rather than starting from a pose the caller drove
# > to, and 187 -> 208 at `2ccc7e6`, which chains the pick's three arm plans by their
# > own endpoints instead of by the raw IK solution. **The toss's 51 is unchanged
# > across all three**, which is the check that the swing itself did not move --
# > upstream measured the same, reporting its own canonical rollout bit-identical
# > either side of that commit.
# >
# > Reproducibility was checked rather than assumed at this pin too: 3/3 independent
# > runs at `CANONICAL_SEED` returned `[208, 51]` and x = 2.020231 bit-identically.
REST_X = 2.0202
BIN_FLOOR_Z = 0.0444

# What the two controller executions take, at the same operating point and re-measured in
# the same run. Pinned so that a change in either controller's termination condition is a
# loud event: a controller that silently stopped terminating would otherwise show up only
# as a slower suite.
ORACLE_CONTROLLER_STEPS = [208, 51]

# Two standoffs from inside upstream's own `TARGET_DISTANCE_BOUNDS`, measured 2026-08-16
# at `CANONICAL_SEED` to land on opposite sides of the goal box: 1.25 rests the cube at
# x = 2.1057 (inside) and 1.45 at x = 1.6901 (short). Both ends of the range the sampler
# draws from, so a test that needs a scoring rollout and a missing one needs no standoff
# this domain could not itself have drawn.
# Re-measured 2026-08-18 at the bumped pins, because the scored box narrowed from
# x in [1.85, 2.15] to [1.95, 2.10] and the old lower-bound choice stopped
# scoring: at `CANONICAL_SEED`, 1.25 now rests the cube at x = 2.1061, which is
# 6 mm *past* the box's far edge. Swept across the whole range at 0.025 m, the
# scoring interval is 1.275 to 1.400 (6/9 sampled standoffs score). 1.35 is taken
# rather than an edge of that interval so the choice is not one carom away from
# flipping; it rests at x = 2.0204, mid-box.
STANDOFF_THAT_SCORES = 1.35
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
            symbolic = IN_BIN.holds(state, (env.cube, env.bin))
            assert symbolic == env.is_solved(), (
                f"standoff {standoff}: InBin said {symbolic} while KINDER's "
                f"own _check_goals() said {env.is_solved()}"
            )
            action = SkillOraclePolicy.get_labeled_action(
                state=state, env=env, goal=goal, throw_standoff=standoff
            )
            state = env.take_action(action=action.action)
        assert IN_BIN.holds(state, (env.cube, env.bin)) == env.is_solved()
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
        # And it is the *inflated* box rather than the range the JSON declares --
        # derived from the installed file rather than written down, so a further
        # tightening of the region is caught by the containment tests, which say
        # what broke, instead of here as a bare number mismatch.
        (goal_range,) = _installed_task_json()["regions"]["blocks_goal_region"]["ranges"]
        assert live[0] == pytest.approx(goal_range[0] - GROUND_PLACEMENT_THRESHOLD, abs=1e-6)
        assert live[3] == pytest.approx(goal_range[3] + GROUND_PLACEMENT_THRESHOLD, abs=1e-6)
        assert live[0] < goal_range[0] and live[3] > goal_range[3]
    finally:
        env.close()


def test_the_oracle_reproduces_the_recorded_landing_and_step_counts() -> None:
    """The reference numbers, **re-measured 2026-08-18 at the pins this branch carries**:
    the cube comes to rest at x = 2.0202, z = 0.0444 (the bin's interior floor, i.e. the
    cube is *in* the bin), `_check_goals()` True, and the two controller executions
    terminating in 208 / 51 steps.

    See `REST_X`' own note for the three earlier measurements, why each moved, and why
    they are left as published rather than replaced. Every one of them was reproduced
    bit-identically across three independent runs before being pinned, this one included
    (3/3 at `CANONICAL_SEED`)."""
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


def _installed_task_json() -> dict:
    """`Tossing3D-o1.json`, read out of the KINDER that is actually installed.

    Located through the import system rather than a hardcoded path, so it is the tree that
    is *run* -- a second checkout at a different commit has already caused a wrong SHA to
    be stated as fact.
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
    return json.loads(task_json.read_text())


# How far a margin may go negative and still count as containment. The two boxes were
# *exactly* coincident before `blocks_goal_region` was tightened, and that scene is
# correct -- but subtracting equal floats through two different routes (JSON arithmetic
# versus the compiled model) leaves noise of order 1e-16, so a bare `>= 0` would call a
# valid coincident geometry a failure. 1e-9 is seven orders of magnitude below the
# smallest real margin at the current pin (0.03 m) and seven above the observed noise.
CONTAINMENT_TOLERANCE = 1e-9


def _containment_margins(
    *,
    scored_x: tuple[float, float],
    scored_y: tuple[float, float],
    footprint_x: tuple[float, float],
    footprint_y: tuple[float, float],
) -> dict[str, float]:
    """Per-edge signed clearance from the scored box to the bin's footprint.

    Positive means the scored box is inside the footprint on that edge. **All three
    containment tests call this**, including the converse guard -- an inlined second copy
    would let the guard keep passing while the real assertion was inverted, which is
    exactly the "test that proves nothing" shape this file exists to avoid.
    """
    return {
        "x near": scored_x[0] - footprint_x[0],
        "x far": footprint_x[1] - scored_x[1],
        "y left": scored_y[0] - footprint_y[0],
        "y right": footprint_y[1] - scored_y[1],
    }


def _is_contained(*, margins: dict[str, float]) -> bool:
    return all(margin >= -CONTAINMENT_TOLERANCE for margin in margins.values())


def _live_bin_half_extents(*, env) -> tuple[float, float]:
    """The bin's half-extents off the compiled MuJoCo model, not off the task JSON.

    Read off the raw observation rather than the `core.State`, because this domain's bin
    type carries the scored box (`x_min`..`z_max`) rather than the bin's own bounding box.
    The observation's `bb_x`/`bb_y` come from the compiled MuJoCo model, so this stays a
    measurement of the live scene rather than a re-reading of the task JSON.
    """
    observation = env.backend().observe()
    return (
        float(observation.get(name="bin_0", feature="bb_x")) / 2,
        float(observation.get(name="bin_0", feature="bb_y")) / 2,
    )


def test_the_shipped_scenes_scoring_box_lies_inside_the_bin() -> None:
    """**The guard that makes a `reference/kindergarden` pin bump loud instead of silent.**

    This domain runs whatever `Tossing3D-o1.json` the installed KINDER registers, so the
    scene moves with the pin. Making that coupling observable is this test's whole job.

    ## What is asserted, and why in this direction

    The invariant the domain cannot survive losing is **scoring implies in-bin**: a cube
    may not score from a position outside the bin. So the scored box must lie *inside* the
    bin's footprint.

    **Containment, not containment-with-margin**, and the distinction is deliberate: before
    `blocks_goal_region` was tightened the two boxes were exactly coincident, which is a
    correct scene with zero margin on every edge. Requiring a positive margin would call it
    a failure. The only slack allowed is `CONTAINMENT_TOLERANCE`, which exists for float
    noise rather than for geometry.

    That is the opposite containment from the one this test asserted before, and the reason
    is a real change in the scene rather than a change of mind. It used to require the two
    boxes to share a **centre**, to within 5 mm. That held while `blocks_goal_region` was
    `[1.90, 2.10]` on x, because inflating it by `GROUND_PLACEMENT_THRESHOLD` per side gives
    `[1.85, 2.15]` -- exactly the 0.30 m bin footprint, so the two boxes coincided and every
    containment held trivially. At the pin this branch bumps to, the region is
    `[2.00, 2.05]`, inflating to `[1.95, 2.10]`: strictly smaller than the footprint on
    every side, and centred at 2.025 against a bin at 2.000. **The old centring assertion is
    red on this correct scene, by 25 mm against its own 5 mm tolerance** -- and a test
    asserting the bin fits inside the scoring box would be red too, since a 0.30 m bin
    cannot fit inside a 0.15 m box.

    The consequence is worth stating plainly, because prose elsewhere still assumes
    otherwise: **"the cube is in the bin" and "the cube scores" are no longer the same
    event.** A cube resting at x = 1.90 is inside the bin and does not score. What survives,
    and what this test pins, is the one-way implication.

    ## Why containment and not centring

    Centring is neither necessary nor sufficient: two boxes can share a centre while one
    spills outside the other, and a scored box slightly off-centre but well inside is
    harmless. The form this refuses is the one that actually broke the domain -- upstream
    `1183de7` moved `bin_init_region` to x = 2.23 and left `blocks_goal_region` behind, so
    the scored box sat 230 mm clear of the bin and only a throw that **missed** the bin
    scored. `test_the_containment_guard_would_catch_a_bin_moved_off_the_scored_box` proves
    this arithmetic still rejects that geometry.

    ## Why the JSON and not the file's bytes

    A byte or hash comparison would fail on every pin bump, including ones that change
    nothing about the geometry, and a test that cries wolf gets deleted. Every number here
    is read from the installed task JSON at test time -- none is written down -- so a
    further tightening of the region moves the assertion with it rather than breaking it.

    Offline as far as the geometry goes (it reads the JSON, not a compiled model), but it
    needs KINDER installed to find the file. The live counterpart is
    `test_the_live_scoring_window_lies_inside_the_bins_live_footprint`.
    """
    task_json = _installed_task_json()
    regions = task_json["regions"]

    (bin_range,) = regions["bin_init_region"]["ranges"]
    bin_x = (bin_range[0] + bin_range[2]) / 2
    bin_y = (bin_range[1] + bin_range[3]) / 2

    bin_spec = task_json["objects"]["bin"]["bin_0"]
    half_length, half_width = bin_spec["length"] / 2, bin_spec["width"] / 2
    footprint_x = (bin_x - half_length, bin_x + half_length)
    footprint_y = (bin_y - half_width, bin_y + half_width)

    (goal_range,) = regions["blocks_goal_region"]["ranges"]
    # A ground region is inflated by `ground_placement_threshold` per side before it becomes
    # a `Region`, so the scored box is wider than the range the file declares.
    scored_x = (
        goal_range[0] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[3] + GROUND_PLACEMENT_THRESHOLD,
    )
    scored_y = (
        goal_range[1] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[4] + GROUND_PLACEMENT_THRESHOLD,
    )

    margins = _containment_margins(
        scored_x=scored_x, scored_y=scored_y, footprint_x=footprint_x, footprint_y=footprint_y
    )
    assert _is_contained(margins=margins), (
        f"the scored box {scored_x} x {scored_y} is not inside the bin's footprint "
        f"{footprint_x} x {footprint_y}; per-edge margins were {margins}. A cube can score "
        "from outside the bin, which is the pre-PR-126 defect."
    )


def test_the_live_scoring_window_lies_inside_the_bins_live_footprint() -> None:
    """The same invariant as the test above, measured off the running simulator instead.

    Worth having twice because neither box is where the file says it is: the goal region is
    inflated before it becomes a `Region`, the bin's placement is sampled from its own
    range, and the bin's bounding box comes off the compiled MuJoCo model rather than the
    JSON's `length`/`width`. The JSON test would still pass if upstream changed how a region
    is inflated; this one would not.
    """
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        bin_x = state.get(obj=env.bin, feature_name="x")
        bin_y = state.get(obj=env.bin, feature_name="y")
        live = env.backend().goal_region_bbox()
        scored_x, scored_y = (live[0], live[3]), (live[1], live[4])

        half_length, half_width = _live_bin_half_extents(env=env)
        footprint_x = (bin_x - half_length, bin_x + half_length)
        footprint_y = (bin_y - half_width, bin_y + half_width)

        margins = _containment_margins(
            scored_x=scored_x,
            scored_y=scored_y,
            footprint_x=footprint_x,
            footprint_y=footprint_y,
        )
        assert _is_contained(margins=margins), (
            f"the live scoring window x {scored_x} y {scored_y} is not inside the bin's "
            f"live footprint x {footprint_x} y {footprint_y}; per-edge margins were "
            f"{margins}. Scoring no longer implies in-bin."
        )
    finally:
        env.close()


def test_the_containment_guard_would_catch_a_bin_moved_off_the_scored_box() -> None:
    """The converse direction: prove the two tests above are capable of going red.

    A geometry guard that only ever runs against a correct scene proves nothing -- it would
    pass just as happily if it were asserting a tautology. This reconstructs the actual
    historical defect (upstream `1183de7` put the bin at x = 2.23 and left the scored box
    behind) from the installed JSON's own numbers and checks that it is rejected.

    **It calls `_containment_margins`/`_is_contained`, the same two functions the real
    tests call**, rather than repeating the arithmetic. An inlined copy would keep passing
    if a sign were inverted in the real assertions, so it would prove that *a* containment
    check rejects the bad scene rather than that *this* one does.
    """
    task_json = _installed_task_json()
    (goal_range,) = task_json["regions"]["blocks_goal_region"]["ranges"]
    half_length = task_json["objects"]["bin"]["bin_0"]["length"] / 2

    scored_x = (
        goal_range[0] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[3] + GROUND_PLACEMENT_THRESHOLD,
    )
    (goal_y_min, goal_y_max) = (goal_range[1], goal_range[4])
    scored_y = (
        goal_y_min - GROUND_PLACEMENT_THRESHOLD,
        goal_y_max + GROUND_PLACEMENT_THRESHOLD,
    )
    displaced_bin_x = 2.23
    footprint_x = (displaced_bin_x - half_length, displaced_bin_x + half_length)
    half_width = task_json["objects"]["bin"]["bin_0"]["width"] / 2
    footprint_y = (-half_width, half_width)

    margins = _containment_margins(
        scored_x=scored_x, scored_y=scored_y, footprint_x=footprint_x, footprint_y=footprint_y
    )
    assert not _is_contained(margins=margins), (
        "the pre-PR-126 geometry passes the containment check, so the guard above cannot "
        f"catch the defect it exists for; margins were {margins}"
    )


def test_the_thrown_cube_really_comes_to_rest_on_one_of_its_faces() -> None:
    """**The evidence behind the composed toss's `OnGround` add effect.** kb#113's
    operator model records `15/15` scoring throws leaving the cube resting on a face, and
    this package added `OnGround` to the toss's add effects on the strength of that. An
    add effect that read false after a scoring throw would have EES label a throw that
    scored a failure, so the claim is worth one live rollout.

    Deliberately checks the *tilt* rather than the classifier, so it says which of the two
    conjuncts holds: a cube in the bin is off the floor by the bin's own interior height,
    which is what upstream's `ON_GROUND_TOLERANCE` has to absorb, and reading only its verdict
    would not distinguish "landed flat" from "landed inside the height tolerance"."""
    env = _env()
    try:
        state = _run_oracle(env=env)
        assert env.is_solved()
        rotation = tuple(
            state.get(obj=env.cube, feature_name=name) for name in ("qx", "qy", "qz", "qw")
        )
        # Upstream's own function and upstream's own tolerance, which is the pair
        # `_check_on_ground` itself uses. This package no longer carries a closed-form
        # copy of the symmetry algebra to compare against -- the classifier is upstream's
        # now, so measuring the same quantity it measures is the honest check.
        from kinder_models.dynamic3d.cube_symmetry import cube_tilt_from_upright
        from kinder_models.dynamic3d.utils import ON_GROUND_TOLERANCE

        assert cube_tilt_from_upright(rotation) < ON_GROUND_TOLERANCE
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


def test_holding_uses_upstreams_forward_kinematics_conjunct() -> None:
    """The conjunct this domain could not evaluate before, and therefore dropped.

    Upstream's `Holding` requires the end effector to be within
    `END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE` of the object, computed by forward
    kinematics through a live `PyBulletSim`. Our own version had a closed gripper and a
    lifted cube and nothing else, so it could call a cube "held" that was airborne
    without being grasped.

    Teleporting the cube two metres away while leaving the gripper closed separates the
    two: the weaker version still says held, upstream's does not. This is a
    simulator-backed test because that is exactly what the conjunct needs -- it is one of
    the two probes that could not stay offline after the swap.
    """
    env = _env()
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
        state = env.take_action(action=action.action)
        assert HOLDING.holds(state, (env.robot, env.cube)), "the oracle's Pick should grasp"

        backend = env.backend()
        snapshot = backend.snapshot()
        cube = snapshot.get_object_from_name(backend.cube_name)
        # Still closed on *something*, still well above the floor, but nowhere near the
        # gripper. Only the FK conjunct can tell the difference.
        snapshot.set(cube, "x", float(snapshot.get(cube, "x")) + 2.0)
        atoms = backend.abstract_atoms(state=snapshot)
        assert ("Holding", ("robot", "cube_0")) not in atoms
    finally:
        env.close()


def test_in_bin_agrees_with_kinders_own_goal_check_at_the_boundary() -> None:
    """`InBin`'s boundary probes, which used to be offline.

    `MovableInGoalRegion` reads the scored region off the live env's ground fixture, so
    it cannot be evaluated from a hand-built state any more -- these moved here from
    `test_predicates.py` rather than being deleted. The check is the one that matters:
    upstream's classifier and KINDER's own `_check_goals()` must agree at every point,
    including just inside and just outside the far edge.
    """
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        backend = env.backend()
        x_min, _, _, x_max, _, _ = backend.goal_region_bbox()
        snapshot = backend.snapshot()
        cube = snapshot.get_object_from_name(backend.cube_name)
        snapshot.set(cube, "y", 0.0)
        snapshot.set(cube, "z", 0.0444)

        for x, expected in (
            (x_min - 0.01, False),
            (x_min + 0.01, True),
            ((x_min + x_max) / 2, True),
            (x_max - 0.01, True),
            (x_max + 0.01, False),
        ):
            snapshot.set(cube, "x", float(x))
            backend.restore(snapshot=snapshot)
            atoms = backend.abstract_atoms()
            symbolic = ("MovableInGoalRegion", ("cube_0",)) in atoms
            assert symbolic is expected, f"x={x}: upstream said {symbolic}, expected {expected}"
            assert symbolic == backend.check_goals(), (
                f"x={x}: MovableInGoalRegion said {symbolic} while KINDER's own "
                f"_check_goals() said {backend.check_goals()}"
            )
    finally:
        env.close()
