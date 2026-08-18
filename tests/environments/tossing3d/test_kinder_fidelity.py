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
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HOLDING
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
GROUND_PLACEMENT_THRESHOLD = 0.05
CONTAINMENT_TOLERANCE = 1e-9


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

        (goal_range,) = _installed_task_json()["regions"]["blocks_goal_region"]["ranges"]
        assert live[0] == pytest.approx(goal_range[0] - GROUND_PLACEMENT_THRESHOLD, abs=1e-6)
        assert live[3] == pytest.approx(goal_range[3] + GROUND_PLACEMENT_THRESHOLD, abs=1e-6)
        assert live[0] < goal_range[0] and live[3] > goal_range[3]
    finally:
        env.close()


def test_the_shipped_scenes_scoring_box_lies_inside_the_bin() -> None:
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
    scored_x = (
        goal_range[0] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[3] + GROUND_PLACEMENT_THRESHOLD,
    )
    scored_y = (
        goal_range[1] - GROUND_PLACEMENT_THRESHOLD,
        goal_range[4] + GROUND_PLACEMENT_THRESHOLD,
    )

    margins = _containment_margins(
        scored_x=scored_x,
        scored_y=scored_y,
        footprint_x=footprint_x,
        footprint_y=footprint_y,
    )
    assert _is_contained(margins=margins), f"scored box not contained in bin: {margins}"


def test_the_live_scoring_window_lies_inside_the_bins_live_footprint() -> None:
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
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
