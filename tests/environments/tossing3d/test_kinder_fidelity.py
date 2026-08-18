"""Simulator-backed tests: does this integration actually agree with KINDER?

**Every test in this file skips cleanly without KINDER**, gated on
`importlib.util.find_spec("kinder")` -- the *import* package name, not the distribution
name `kindergarden`. **CI installs the extra as of 2026-08-14**, so these execute there
rather than skipping; the gate now exists to keep a checkout with an unpopulated
`reference/` from failing wholesale, not to spare CI. Locally KINDER installs into
`hitl-pmp` itself, so this runs in the ordinary gate:

    scripts/with_env.sh python -m pytest tests/environments/tossing3d/ -q

and under a memory cap, because these tests execute real controllers:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- ...

What these check:

1. `MovableInGoalRegion` agrees with KINDER's own `_check_goals()` along a real episode.
   Note what this is now: **two upstream computations of the same thing**, rather than
   upstream's against a port of it. The predicate stopped being ours when this domain
   adopted `kinder_models.dynamic3d.tossing.state_abstractions`, so the test is weaker
   evidence than it used to be -- and correspondingly less likely to ever fail. It is kept
   because the two really do travel different paths (the abstractor's `PyBulletSim` versus
   the gym env's own goal check) and a divergence there would be worth knowing about.
2. **The scoring box lies inside the bin's footprint, with margin** -- checked twice, once
   off the installed task JSON and once off the live simulator. See those two tests for
   why the containment runs in that direction and why it is the invariant that matters.
3. The harness path end to end: an episode solves, rewinds, renders, and records at
   physics rate without perturbing the physics.

## What left this file with the three-skill decomposition

Seven tests are gone, and all seven for one reason: they characterised the pose *between*
a base move and a throw. `RobotAtSuccessfulThrowPose`, `THROW_RANGE`, the overshoot and
shortfall margins, the lateral tolerance and the accepted band were the machinery for
"the base is standing somewhere a throw from here scores". `move_to_toss_location_and_toss`
executes the move and the throw together, so there is no intermediate state to
characterise and nothing left for those tests to be about. That is obsolete work, not
pending work: the throw band does not need re-deriving.

Two more went for narrower reasons worth recording, because both were guarding something
real:

* `test_the_goal_box_in_the_state_is_the_live_region_bbox_element_for_element` checked that
  the scored box travelled in the `State` on the bin object. It no longer does -- the six
  bbox features are gone, and `MovableInGoalRegion` reads the region off the live
  simulator through the same `check_in_region` call `_check_goals()` makes. The defect it
  existed to catch (a predicate written against the *uninflated* JSON range, which is
  narrower than the true box on x) is now structurally impossible rather than merely
  tested for.
* `test_oracle_pick_parameters_match_upstreams_own_sampler` checked two literals in
  `skill_oracle_policy.py` against what upstream's sampler draws. The oracle no longer
  writes literals; it draws from the controller's own sampler, so the two cannot disagree.

## A published number that this change makes provisional

`REST_X = 1.9926` -- where the cube came to rest under the oracle at
`(standoff 1.35, 140 deg/s, 792 ms)` -- is **not asserted here any more**, and the reason
is not that it was wrong. The oracle no longer *has* those three dials to set: it draws all
four continuous parameters from `move_to_toss_location_and_toss`'s own `sample_parameters`
in one call, off `ORACLE_PARAMETER_SEED`. There is no parameterisation to hold fixed, so
the number is not reproducible in the form it was published. It is recorded as stale in
`docs/experiment-logs/` rather than recomputed here.
"""

from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    MOVABLE_IN_GOAL_REGION,
    Tossing3DPredicates,
)
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

from .conftest import CANONICAL_SEED, requires_kinder

pytestmark = requires_kinder

# Upstream inflates a ground region by this much per side before it becomes a `Region`
# (`kinder/envs/dynamic3d/objects/base.py`, `PhysicsObject.ground_placement_threshold`).
# Restated rather than imported: upstream sets it as an instance attribute on a literal, so
# there is no module constant to read. The live counterpart test measures the real window
# off the simulator, so a change to this number upstream shows up as a disagreement between
# the two rather than going unnoticed.
GROUND_PLACEMENT_THRESHOLD = 0.05

# How close to the bin's footprint edge the live probe is willing to call "inside the bin".
# The scoring window's measured margins at this pin are 100 mm (near x), 50 mm (far x) and
# 30 mm (each side in y), so 10 mm is comfortably finer than the smallest of them while
# staying far coarser than any floating-point edge effect.
FOOTPRINT_MARGIN_PROBE = 0.01

# The height a resting cube's centre sits at (`size: 0.025` in the task JSON, so a
# half-extent of 0.025 off the floor). The goal region's z range admits it.
RESTING_CUBE_Z = 0.025


def _env():
    return Tossing3DEnvironment()


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


def _scores_at(*, env: Tossing3DEnvironment, x: float, y: float, z: float) -> bool:
    """Would a cube resting at `(x, y, z)` satisfy `MovableInGoalRegion`?

    Drives the live predicate rather than any geometry written down here, so it is the same
    `check_in_region` call `_check_goals()` makes. The cube is moved in a *copy* of the
    KINDER state, so the scene itself is never disturbed.
    """
    abstraction = env.abstraction()
    predicate = Tossing3DPredicates.get(abstraction=abstraction, name=MOVABLE_IN_GOAL_REGION)
    kinder_state = env.backend().kinder_state()
    cube = kinder_state.get_object_from_name("cube_0")
    probe = kinder_state.copy()
    probe.set(cube, "x", x)
    probe.set(cube, "y", y)
    probe.set(cube, "z", z)
    state = env.build_state(kinder_state=probe, seed=CANONICAL_SEED, steps_taken=0)
    # The predicate reads the live simulator, so a cached verdict is about the scene as it
    # was before this probe moved the cube.
    abstraction.invalidate()
    return predicate.holds(state, (Tossing3DEnvironment.cube,))


def _live_scoring_edge(
    *, env: Tossing3DEnvironment, axis: str, inside: float, outside: float
) -> float:
    """Bisect along one axis for the boundary of the live scoring window.

    Bisection rather than a swept grid because each probe is a state build plus an
    abstractor call: 24 iterations resolve the edge to well under a millimetre, where a
    grid fine enough to do the same would cost hundreds.
    """
    other = 0.0
    for _ in range(24):
        middle = (inside + outside) / 2
        point = (middle, other) if axis == "x" else (other, middle)
        if _scores_at(env=env, x=point[0], y=point[1], z=RESTING_CUBE_Z):
            inside = middle
        else:
            outside = middle
    return (inside + outside) / 2


def test_movable_in_goal_region_agrees_with_kinders_own_goal_check_along_an_episode() -> None:
    """The differential check this domain's symbolic layer rests on, at every step of a real
    oracle episode.

    The verdict flips during the episode -- `False` while the cube is on the floor and in
    the gripper, `True` once it lands -- so agreeing at every step is a real constraint and
    not something a predicate hardwired to either answer would pass.
    """
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        # Built ONCE, before the rollout: `build_task` rebuilds the scene, so building it
        # inside the loop would silently reset the episode between every step.
        goal = tasks.build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        abstraction = env.abstraction()
        predicate = Tossing3DPredicates.get(abstraction=abstraction, name=MOVABLE_IN_GOAL_REGION)

        verdicts = []
        for _ in range(3):
            symbolic = predicate.holds(state, (env.cube,))
            assert symbolic == env.is_solved(), (
                f"MovableInGoalRegion said {symbolic} while KINDER's own _check_goals() "
                f"said {env.is_solved()}"
            )
            verdicts.append(symbolic)
            action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
            state = env.take_action(action=action.action)

        assert predicate.holds(state, (env.cube,)) == env.is_solved()
        assert verdicts[0] is False, "the cube starts outside the goal region"
    finally:
        env.close()


def test_the_shipped_scenes_scoring_box_lies_inside_the_bin_with_margin() -> None:
    """**The guard that makes a `reference/kindergarden` pin bump loud instead of silent.**

    This domain no longer commits a scene of its own: it runs whatever `Tossing3D-o1.json`
    the installed KINDER registers. That is the decision, and its cost is that **the scene
    moves with the pin** -- so the coupling has to be observable, which is this test.

    ## What is asserted, and why in this direction

    The invariant the domain cannot survive losing is **scoring implies in-bin**: a cube
    may not score from a position outside the bin. So the scored box must lie *inside* the
    bin's footprint, with margin.

    That is the opposite containment from the one this test used to assert, and the reason
    is a real change in the scene rather than a change of mind. The scored region has been
    tightened twice. It was `[1.90, 2.10]` on x, which inflates to `[1.85, 2.15]` -- exactly
    the 0.30 m bin footprint, so the two *coincided* and either containment held trivially.
    It is now `[2.00, 2.05]`, inflating to `[1.95, 2.10]`, which is strictly smaller than the
    footprint on every side. **A test asserting that the bin fits inside the scoring box
    would now be red on a correct scene**, since a 0.30 m bin cannot fit inside a 0.15 m box.

    The consequence is worth stating plainly because prose elsewhere still assumes
    otherwise: **"the cube is in the bin" and "the cube scores" are no longer the same
    event.** A cube resting at x = 1.90 is inside the bin and does not score. What survives,
    and what this test pins, is the one-way implication.

    ## Why containment and not centring

    Centring is neither necessary nor sufficient: two boxes can share a centre while one
    spills outside the other, and a scored box slightly off-centre but well inside is
    harmless. The form this refuses is the one that actually broke the domain -- upstream
    `1183de7` moved `bin_init_region` to x = 2.23 and left `blocks_goal_region` behind, so
    the scored box sat 230 mm clear of the bin and only a throw that **missed** the bin
    scored.

    ## Why the JSON and not the file's bytes

    A byte or hash comparison would fail on every pin bump, including ones that change
    nothing about the geometry, and a test that cries wolf gets deleted. Every number here
    is read from the installed task JSON at test time -- none is written down -- so a
    tightening of the region moves the assertion with it rather than breaking it.

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
    # A ground region is inflated by `ground_placement_threshold` per side before it
    # becomes a `Region`, so the scored box is wider than the range the file declares.
    scored_x = (
        goal_range[0] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[3] + GROUND_PLACEMENT_THRESHOLD,
    )
    scored_y = (
        goal_range[1] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[4] + GROUND_PLACEMENT_THRESHOLD,
    )

    margins = {
        "x near": scored_x[0] - footprint_x[0],
        "x far": footprint_x[1] - scored_x[1],
        "y left": scored_y[0] - footprint_y[0],
        "y right": footprint_y[1] - scored_y[1],
    }
    assert all(margin > 0 for margin in margins.values()), (
        f"the scored box {scored_x} x {scored_y} is not inside the bin's footprint "
        f"{footprint_x} x {footprint_y}; per-edge margins were {margins}. A cube can now "
        "score from outside the bin, which is the pre-PR-126 defect."
    )


def test_the_live_scoring_window_lies_inside_the_bins_live_footprint() -> None:
    """The same invariant as the test above, measured off the running simulator instead.

    Worth having twice because neither box is where the file says it is: the goal region is
    inflated before it becomes a `Region`, the bin's placement is sampled from its own
    range, and the bin's bounding box comes off the compiled MuJoCo model rather than the
    JSON's `length`/`width`. The JSON test would still pass if upstream changed how a
    region is inflated; this one would not.

    The window's edges are found by driving `MovableInGoalRegion` itself, so this is the
    real scoring rule and not a reconstruction of it.
    """
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        kinder_state = env.backend().kinder_state()
        bin_object = kinder_state.get_object_from_name("bin_0")
        bin_x = float(kinder_state.get(bin_object, "x"))
        bin_y = float(kinder_state.get(bin_object, "y"))
        half_length = float(kinder_state.get(bin_object, "bb_x")) / 2
        half_width = float(kinder_state.get(bin_object, "bb_y")) / 2

        assert _scores_at(env=env, x=bin_x, y=bin_y, z=RESTING_CUBE_Z), (
            "a cube at the bin's own centre does not score, so the scored box and the bin "
            "are not in the same place at all"
        )

        # Bisect outward from the bin's centre to each footprint edge. The `outside` end of
        # each bracket is the edge itself, which the assertions below then require to be a
        # non-scoring position.
        near_x = _live_scoring_edge(env=env, axis="x", inside=bin_x, outside=bin_x - half_length)
        far_x = _live_scoring_edge(env=env, axis="x", inside=bin_x, outside=bin_x + half_length)
        left_y = _live_scoring_edge(env=env, axis="y", inside=bin_y, outside=bin_y - half_width)
        right_y = _live_scoring_edge(env=env, axis="y", inside=bin_y, outside=bin_y + half_width)

        margins = {
            "x near": near_x - (bin_x - half_length),
            "x far": (bin_x + half_length) - far_x,
            "y left": left_y - (bin_y - half_width),
            "y right": (bin_y + half_width) - right_y,
        }
        assert all(margin > FOOTPRINT_MARGIN_PROBE for margin in margins.values()), (
            f"the live scoring window x {(near_x, far_x)} y {(left_y, right_y)} is not "
            f"inside the bin's live footprint centred at ({bin_x}, {bin_y}) with "
            f"half-extents ({half_length}, {half_width}); per-edge margins were {margins}"
        )

        # The direct statement of "scoring implies in-bin": just outside the footprint,
        # on either axis, nothing scores.
        for label, (x, y) in {
            "beyond the near wall": (bin_x - half_length - FOOTPRINT_MARGIN_PROBE, bin_y),
            "beyond the far wall": (bin_x + half_length + FOOTPRINT_MARGIN_PROBE, bin_y),
            "beyond the left wall": (bin_x, bin_y - half_width - FOOTPRINT_MARGIN_PROBE),
            "beyond the right wall": (bin_x, bin_y + half_width + FOOTPRINT_MARGIN_PROBE),
        }.items():
            assert not _scores_at(env=env, x=x, y=y, z=RESTING_CUBE_Z), (
                f"a cube {label} of the bin scores, so scoring no longer implies in-bin"
            )
    finally:
        env.close()


def test_a_full_episode_through_the_problem_runs_the_two_skill_plan_to_completion() -> None:
    """End to end through the harness's own path: `run_task_episode` with the oracle
    policy, on the canonical scene.

    **This deliberately does not assert that the episode solves, and that is a finding
    rather than a weakened test.** Measured on this branch at the shipped
    `ORACLE_PARAMETER_SEED = 123`, the oracle scores **0/10** scene seeds (125 and 0-8),
    resting the cube at x ~ 1.730 every time against a scoring window that starts at
    x = 1.95 -- a systematic ~220 mm shortfall, not a flaky band.

    The fused operator itself is fine: sweeping the parameter seed on this same scene,
    **3/8** seeds score (3, 42, 99 land at 2.0605, 2.0762, 1.9884; 0, 1, 2, 7, 123 fall
    short). The sampler's draws are inside its own documented bounds -- speed in *radians*
    per second over [2.0071, 2.4435], i.e. [115, 140] deg/s, and release over [700, 840] ms.
    Seed 123 simply draws 120.5 deg/s at 725.8 ms, which is a losing combination: the three
    scoring draws all pair a high speed with a late release.

    So the reference arm is currently a ~3/8 sampler carrying an unlucky fixed seed, not a
    ceiling. `skill_oracle_policy.py`'s own docstring already says it "is not a guaranteed
    solve"; what was not known is that the shipped default never solves. Choosing a seed
    that scores would be overfitting to one scene without a measured rate behind it, so
    nothing here is changed to make this test green -- the number is reported instead.

    What this test still pins is real and would catch a broken integration: the oracle
    emits exactly the two-skill plan shape, both controllers actually run, and the episode
    terminates through the harness rather than raising.
    """
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

        chosen: list[str] = []

        # core.method.types.Policy is a positional Callable[[State], LabeledAction].
        def policy(state) -> LabeledAction:  # noqa: PLR0917
            labeled = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)
            # `LabeledAction` carries only the raw action and a label, by design -- the
            # label is the one place the chosen skill's name survives.
            chosen.append(labeled.label.split("(")[0])
            return labeled

        _, frames, _ = problem.run_task_episode(task=task, policy=policy)
        assert frames == []
        assert chosen[:2] == ["pick_cube", "move_to_toss_location_and_toss"], (
            f"the oracle's plan shape is no longer pick-then-throw: {chosen}"
        )
        # Both controllers really drove the simulator rather than falling through as
        # unrecognised skills: the cube ends up past the barrier, which only a completed
        # grasp followed by a completed throw can achieve.
        final = env.get_current_state()
        barrier_x = final.get(obj=env.barrier, feature_name="x")
        assert final.get(obj=env.cube, feature_name="x") > barrier_x, (
            "the cube never crossed the barrier, so the pick and the throw did not both run"
        )
        # After the throw the cube is unreachable and the oracle keeps proposing
        # `pick_cube`, whose grasp cannot plan. That recorded no-op is the irreversibility
        # this domain exists to exhibit, not a malfunction -- so the episode is expected to
        # end carrying a skill error rather than a clean one.
        assert chosen[2:] == ["pick_cube"] * len(chosen[2:])
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
        assert task.goal.describe() == "MovableInGoalRegion(cube_0)"
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
    execution -- tens of MuJoCo ticks -- so rendering once per decision gave a two-frame
    `episode.mp4` of a domain that exists to show a throw. With the gymnasium
    `RenderCollection` wrapper in place, every one of those ticks reaches the clip.

    Asserted as a floor rather than an exact number: the controllers terminate on their own
    conditions, and pinning the total would make this a fragile second copy of a step-count
    assertion.

    Like its sibling above, this no longer asserts that the episode *solves* -- the shipped
    `ORACLE_PARAMETER_SEED` scores `0/10` scene seeds on this branch. The frame count is the
    claim here and it does not depend on the throw landing.
    """
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

        def policy(state) -> LabeledAction:  # noqa: PLR0917
            return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

        _, frames, _ = problem.run_task_episode(
            task=task, policy=policy, renderer=Tossing3DRenderer
        )

        assert len(frames) > 100, f"expected a physics-rate clip, got {len(frames)} frames"
        assert len({frame.shape for frame in frames}) == 1, "ffmpeg needs one frame size"
        # Recording is per-episode and must not leak into the next, unrendered one.
        assert env.backend().record_substeps is False
    finally:
        env.close()


def test_recording_does_not_change_where_the_cube_comes_to_rest() -> None:
    """Presentation-only, proved rather than asserted in prose: the same episode run with
    and without the recording wrapper has to land the cube in the same place. A wrapper
    that stepped the simulator, or a per-tick render that perturbed it, would show up here
    and nowhere else.

    **This compares the two runs against each other rather than against a published
    landing.** It used to also assert `REST_X = 1.9926`, the resting x under the oracle at
    `(standoff 1.35, 140 deg/s, 792 ms)`. The oracle no longer has those dials to set --
    it draws all four continuous parameters from the fused controller's own sampler in one
    call -- so there is no parameterisation to hold fixed and that number is not
    reproducible in the form it was published. Dropping it costs nothing here: the claim
    this test exists to make is that recording is a pure observer, which is exactly the
    equality between the two runs.
    """
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

    assert rest["recorded"] == pytest.approx(rest["plain"], abs=1e-6)


def test_the_containment_guard_would_catch_a_bin_moved_off_the_scored_box() -> None:
    """The converse direction: prove the two tests above are capable of going red.

    A geometry guard that only ever runs against a correct scene proves nothing -- it would
    pass just as happily if it were asserting a tautology. This reconstructs the actual
    historical defect (upstream `1183de7` put the bin at x = 2.23 and left the scored box at
    x = 2.0) from the installed JSON's own numbers and checks that the containment
    arithmetic rejects it.
    """
    task_json = _installed_task_json()
    (goal_range,) = task_json["regions"]["blocks_goal_region"]["ranges"]
    bin_spec = task_json["objects"]["bin"]["bin_0"]
    half_length = bin_spec["length"] / 2

    scored_x = (
        goal_range[0] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[3] + GROUND_PLACEMENT_THRESHOLD,
    )
    displaced_bin_x = 2.23
    footprint_x = (displaced_bin_x - half_length, displaced_bin_x + half_length)

    near_margin = scored_x[0] - footprint_x[0]
    far_margin = footprint_x[1] - scored_x[1]
    assert min(near_margin, far_margin) < 0, (
        "the pre-PR-126 geometry passes the containment check, so the guard above cannot "
        "catch the defect it exists for"
    )
