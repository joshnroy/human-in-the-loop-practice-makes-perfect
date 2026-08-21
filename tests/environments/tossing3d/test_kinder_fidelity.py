"""Integration smoke tests: does this integration agree with KINDER?

These tests verify the live bridge between `hitl-pmp`'s environment wrapper,
state abstractor, task builder, and renderer against the underlying KINDER
and MuJoCo runtime.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
)
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

# The DISTRIBUTION is `kindergarden`; the IMPORT package is `kinder`.
needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`)",
)

pytestmark = needs_kinder

CANONICAL_SEED = 125
# `blocks_goal_region.target` is `bin_0`, not `ground`: its inflation comes from
# `MujocoObject`'s per-object placement threshold (1cm), not the ground fixture's (5cm).
OBJECT_PLACEMENT_THRESHOLD = 0.01
CONTAINMENT_TOLERANCE = 1e-9
# `bin_init_region` now samples a real range rather than one fixed point, so a claim about
# "the" bin position has to hold across several seeds, not just CANONICAL_SEED.
BIN_POSITION_SEEDS = (CANONICAL_SEED, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _env():
    return Tossing3DEnvironment()


def _installed_task_json() -> dict:
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


def _containment_margins(
    *,
    scored_x: tuple[float, float],
    scored_y: tuple[float, float],
    footprint_x: tuple[float, float],
    footprint_y: tuple[float, float],
) -> dict[str, float]:
    return {
        "x near": scored_x[0] - footprint_x[0],
        "x far": footprint_x[1] - scored_x[1],
        "y left": scored_y[0] - footprint_y[0],
        "y right": footprint_y[1] - scored_y[1],
    }


def _is_contained(*, margins: dict[str, float]) -> bool:
    return all(margin >= -CONTAINMENT_TOLERANCE for margin in margins.values())


def _live_bin_half_extents(*, env) -> tuple[float, float]:
    observation = env.backend().observe()
    return (
        float(observation.get(name="bin_0", feature="bb_x")) / 2,
        float(observation.get(name="bin_0", feature="bb_y")) / 2,
    )


def test_the_goal_box_in_the_state_is_the_live_region_bbox_element_for_element() -> None:
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        live = env.backend().goal_region_bbox()
        corners = ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
        in_state = tuple(state.get(obj=env.bin, feature_name=name) for name in corners)
        assert in_state == pytest.approx(live)

        # blocks_goal_region.target is bin_0: the JSON's "ranges" are in the bin's own
        # local frame, so the bin's live world position has to be added back in before
        # comparing to `live`, which is a world-frame box.
        (goal_range,) = _installed_task_json()["regions"]["blocks_goal_region"]["ranges"]
        bin_x = float(env.backend().observe().get(name="bin_0", feature="x"))
        assert live[0] == pytest.approx(
            bin_x + goal_range[0] - OBJECT_PLACEMENT_THRESHOLD, abs=1e-6
        )
        assert live[3] == pytest.approx(
            bin_x + goal_range[3] + OBJECT_PLACEMENT_THRESHOLD, abs=1e-6
        )
        assert live[0] < bin_x + goal_range[0] and live[3] > bin_x + goal_range[3]
    finally:
        env.close()


@pytest.mark.parametrize(
    "bin_x,bin_y",
    [
        (2.0, 0.0),  # the pre-widening fixed point, still a valid sample
        (1.85, -0.20),  # bin_init_region's corners...
        (1.85, 0.20),
        (2.15, -0.20),
        (2.15, 0.20),
        (1.85, 0.0),  # ...and edge midpoints
        (2.15, 0.0),
        (2.0, -0.20),
        (2.0, 0.20),
    ],
)
def test_the_shipped_scenes_scoring_box_lies_inside_the_bin(*, bin_x: float, bin_y: float) -> None:
    """blocks_goal_region.target is bin_0, so its "ranges" are bin-local: this containment
    property has to hold wherever bin_init_region might place the bin, not just at one
    point -- swept across the corners and edge midpoints of bin_init_region's actual
    sampling range, since the bin can land anywhere in it now (see BIN_POSITION_SEEDS'
    docstring).
    """
    task_json = _installed_task_json()
    regions = task_json["regions"]

    bin_spec = task_json["objects"]["bin"]["bin_0"]
    half_length, half_width = bin_spec["length"] / 2, bin_spec["width"] / 2
    footprint_x = (bin_x - half_length, bin_x + half_length)
    footprint_y = (bin_y - half_width, bin_y + half_width)

    (goal_range,) = regions["blocks_goal_region"]["ranges"]
    scored_x = (
        bin_x + goal_range[0] - OBJECT_PLACEMENT_THRESHOLD,
        bin_x + goal_range[3] + OBJECT_PLACEMENT_THRESHOLD,
    )
    scored_y = (
        bin_y + goal_range[1] - OBJECT_PLACEMENT_THRESHOLD,
        bin_y + goal_range[4] + OBJECT_PLACEMENT_THRESHOLD,
    )

    margins = _containment_margins(
        scored_x=scored_x,
        scored_y=scored_y,
        footprint_x=footprint_x,
        footprint_y=footprint_y,
    )
    assert _is_contained(margins=margins), f"scored box not contained in bin: {margins}"


@pytest.mark.parametrize("seed", BIN_POSITION_SEEDS)
def test_the_live_scoring_window_lies_inside_the_bins_live_footprint(*, seed: int) -> None:
    """Swept across several seeds, not just CANONICAL_SEED: bin_init_region now samples a
    real range rather than one fixed point, so this has to hold wherever the bin actually
    landed, not only at its old, single, always-identical position.
    """
    env = _env()
    try:
        env.reset_to_seed(seed=seed)
        backend = env.backend()
        scored_x = (backend.goal_region_bbox()[0], backend.goal_region_bbox()[3])
        scored_y = (backend.goal_region_bbox()[1], backend.goal_region_bbox()[4])
        obs = backend.observe()
        bin_x = float(obs.get(name="bin_0", feature="x"))
        bin_y = float(obs.get(name="bin_0", feature="y"))
        hx, hy = _live_bin_half_extents(env=env)
    finally:
        env.close()

    margins = _containment_margins(
        scored_x=scored_x,
        scored_y=scored_y,
        footprint_x=(bin_x - hx, bin_x + hx),
        footprint_y=(bin_y - hy, bin_y + hy),
    )
    assert _is_contained(margins=margins), f"live scored window not contained: {margins}"


def test_the_goal_regions_live_bbox_moves_with_the_bins_own_position() -> None:
    """Regression test for the upstream `1183de7`-style desync: a scene edit (there, the
    bin's init region; here, moving the bin itself) must move the *scored* box along with
    it, not leave it sitting at whatever point the scene happened to be built at.

    Before the fix (`blocks_goal_region.target` was `ground`), this failed: the region
    was a fixed site in the world frame, so moving the bin left `goal_region_bbox()`
    completely unchanged. After the fix (`target` is `bin_0`), the region is a site on
    the bin's own body, so MuJoCo tracks its live world position as the bin moves.
    """
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        backend = env.backend()
        before_bbox = backend.goal_region_bbox()
        before_bin_x = float(backend.observe().get(name="bin_0", feature="x"))

        snapshot = backend.snapshot()
        bin_object = snapshot.get_object_from_name(backend.bin_name)
        offset = 0.30
        snapshot.set(bin_object, "x", float(snapshot.get(bin_object, "x")) + offset)
        backend.restore(snapshot=snapshot)

        after_bbox = backend.goal_region_bbox()
        after_bin_x = float(backend.observe().get(name="bin_0", feature="x"))
    finally:
        env.close()

    actual_bin_delta = after_bin_x - before_bin_x
    assert actual_bin_delta == pytest.approx(offset, abs=1e-3), "the bin itself did not move"

    box_delta_min = after_bbox[0] - before_bbox[0]
    box_delta_max = after_bbox[3] - before_bbox[3]
    assert box_delta_min == pytest.approx(actual_bin_delta, abs=1e-3), (
        f"the goal region's x_min did not track the bin's move "
        f"(bin moved {actual_bin_delta}, box x_min moved {box_delta_min})"
    )
    assert box_delta_max == pytest.approx(actual_bin_delta, abs=1e-3), (
        f"the goal region's x_max did not track the bin's move "
        f"(bin moved {actual_bin_delta}, box x_max moved {box_delta_max})"
    )
    # y is untouched by the offset, so the box's y-extent must be unchanged.
    assert after_bbox[1] == pytest.approx(before_bbox[1], abs=1e-6)
    assert after_bbox[4] == pytest.approx(before_bbox[4], abs=1e-6)


def test_the_containment_guard_would_catch_a_bin_moved_off_the_scored_box() -> None:
    margins = _containment_margins(
        scored_x=(1.95, 2.10),
        scored_y=(-0.075, 0.075),
        footprint_x=(1.50, 1.80),
        footprint_y=(-0.15, 0.15),
    )
    assert not _is_contained(margins=margins)


def test_a_full_episode_through_the_problem_solves_the_default_scene() -> None:
    """End to end through the harness's own path: `run_task_episode` with the oracle
    policy, on the canonical scene."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

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


def test_holding_uses_upstreams_forward_kinematics_conjunct() -> None:
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
        snapshot.set(cube, "x", float(snapshot.get(cube, "x")) + 2.0)
        atoms = backend.abstract_atoms(state=snapshot)
        assert ("Holding", ("robot", "cube_0")) not in atoms
    finally:
        env.close()


def test_in_bin_agrees_with_kinders_own_goal_check_at_the_boundary() -> None:
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        backend = env.backend()
        x_min, _, _, x_max, _, _ = backend.goal_region_bbox()
        # blocks_goal_region.target is bin_0, so its y-extent is centred on wherever the
        # bin actually landed -- no longer y=0.0 at every seed now that bin_init_region
        # samples a real range. Sweeping x at a fixed y=0.0 would silently walk outside
        # the (now bin-relative) region whenever a seed's bin isn't at y=0.
        bin_y = float(backend.observe().get(name="bin_0", feature="y"))
        snapshot = backend.snapshot()
        cube = snapshot.get_object_from_name(backend.cube_name)
        snapshot.set(cube, "y", bin_y)
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


# --- reset_movables / reset_cube_and_bin: the partial, robot-untouched reset ------


def _robot_pose(*, state: State) -> tuple[float, float, float, float]:
    return (
        state.get(obj=Tossing3DEnvironment.robot, feature_name="pos_base_x"),
        state.get(obj=Tossing3DEnvironment.robot, feature_name="pos_base_y"),
        state.get(obj=Tossing3DEnvironment.robot, feature_name="pos_base_rot"),
        state.get(obj=Tossing3DEnvironment.robot, feature_name="pos_gripper"),
    )


def test_reset_movables_leaves_the_robot_pose_exactly_where_it_was() -> None:
    """The whole point of this partial reset: unlike set_state (a seed-based whole-
    scene rebuild, which also re-initializes the robot's own pose), this must leave
    the robot's live joint/base configuration untouched.

    **`abs=1e-6`, not bit-exact.** `sim.forward()` recomputes the whole scene's
    kinematics after cube/bin are moved, and `KinderBackend.restore`'s own docstring
    already documents this domain's ~1e-7-relative float32 round-trip noise -- this
    is that same noise, not evidence the robot moved. Measured here at ~9e-9 absolute
    on a single call; the bound is set two orders of magnitude above that, not at it.
    """
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
        state = env.take_action(action=action.action)  # PickCube: moves the robot for real
        before = _robot_pose(state=state)

        assert env.reset_movables() is True
        after = _robot_pose(state=env.get_current_state())
        assert after == pytest.approx(before, abs=1e-6)
    finally:
        env.close()


def test_reset_movables_moves_the_cube_to_a_fresh_ground_pose() -> None:
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        before = env.get_current_state().get(obj=env.cube, feature_name="x")

        env.reset_movables()
        state = env.get_current_state()
        after = state.get(obj=env.cube, feature_name="x")
        assert after != pytest.approx(before, abs=1e-6)
        assert ON_GROUND.holds(state, (env.cube,))
        assert REACHABLE.holds(state, (env.cube, env.barrier))
        assert not IN_BIN.holds(state, (env.cube, env.bin))
    finally:
        env.close()


def test_reset_movables_moves_the_bin_too_now_that_its_region_is_a_real_range() -> None:
    """As of the kindergarden#166 pin, bin_init_region is a genuine box rather than a
    single fixed point -- see KinderBackend.reset_cube_and_bin's own docstring for the
    correction this test pins. Since blocks_goal_region is now parented on bin_0 (also
    #166/#272), this reset relocates the SCORED window along with the bin, not just the
    bin's own visible position."""
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        before_bin = (
            env.get_current_state().get(obj=env.bin, feature_name="x"),
            env.get_current_state().get(obj=env.bin, feature_name="y"),
        )

        env.reset_movables()
        state = env.get_current_state()
        after_bin = (
            state.get(obj=env.bin, feature_name="x"),
            state.get(obj=env.bin, feature_name="y"),
        )
        assert after_bin != pytest.approx(before_bin, abs=1e-6)
    finally:
        env.close()


def test_reset_movables_breaks_a_grasp_since_the_robot_is_never_touched() -> None:
    """Empirical grounding for the operator's HandEmpty(robot) precondition (see
    Tossing3DSkillProvider.human_cube_bin_reset_skill's own docstring), and a
    correction to the naive prediction: teleporting the cube away from wherever the
    gripper is does flip Holding to False (the cube is no longer there to hold), but
    it does NOT flip HandEmpty to True -- the gripper itself is still physically
    closed (nothing here opens it), and upstream's HandEmpty apparently reads gripper
    aperture rather than "is Holding false". So the reachable-if-uncontracted result
    is a state where BOTH are False -- neither holding nor empty-handed, something no
    ordinary pick_cube/toss transition ever produces -- which is a *stronger* reason
    this operator requires HandEmpty(robot) as a precondition than "Holding would go
    stale", not a weaker one. This calls the backend primitive directly, bypassing
    the operator's precondition, specifically to demonstrate why it is required."""
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
        state = env.take_action(action=action.action)
        assert HOLDING.holds(state, (env.robot, env.cube)), "the oracle's Pick should grasp"

        env.reset_movables()
        state = env.get_current_state()
        assert not HOLDING.holds(state, (env.robot, env.cube))
        assert not HAND_EMPTY.holds(state, (env.robot,)), (
            "if this starts holding, the HandEmpty(robot) precondition guard above "
            "needs re-checking against upstream's classifier, not just this test"
        )
    finally:
        env.close()


def test_reset_movables_leaves_the_scenes_seed_and_steps_taken_unchanged() -> None:
    """Neither take_action's forward-dynamics bump nor set_state's always-zero: this
    is a third kind of state change, and its own bookkeeping stays put -- see
    Tossing3DEnvironment.reset_movables's own docstring for why."""
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        before_seed = state.get(obj=env.scene, feature_name="seed")
        before_steps = state.get(obj=env.scene, feature_name="steps_taken")

        env.reset_movables()
        state = env.get_current_state()
        assert state.get(obj=env.scene, feature_name="seed") == before_seed
        assert state.get(obj=env.scene, feature_name="steps_taken") == before_steps
    finally:
        env.close()


def test_reset_movables_leaves_the_barrier_untouched() -> None:
    """Out of scope for this reset: only cube_0/bin_0 are repositioned, never
    cuboid_barrier -- see KinderBackend.reset_cube_and_bin's own docstring for why
    this is a deliberately narrower scope than upstream's own `_initialize_object_
    poses` (which would also touch the barrier)."""
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        before = (
            state.get(obj=env.barrier, feature_name="x"),
            state.get(obj=env.barrier, feature_name="y"),
            state.get(obj=env.barrier, feature_name="z"),
        )
        env.reset_movables()
        state = env.get_current_state()
        after = (
            state.get(obj=env.barrier, feature_name="x"),
            state.get(obj=env.barrier, feature_name="y"),
            state.get(obj=env.barrier, feature_name="z"),
        )
        assert after == pytest.approx(before, abs=1e-9)
    finally:
        env.close()
