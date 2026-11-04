"""The toss feasibility check plans a stand exactly as the dispatched toss controller does.

`TossDirectionSelector` accepts a toss stand only when the base planner finds a path to
it. That check used to rebuild the planner's arguments by hand, and dropped the 3-D check
of the carried arm and cube, the planning scene's bounding boxes and static obstacles
(and in the kitchen disabled the barrier outright). These tests hold the check's
arguments to the controller's own, on recorded dispatch states of both scenes, and pin
upstream's controller to the arguments Tossing3D's builds for it.
"""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from hitl_pmp.environments.tossing3d.toss_direction import (
    NoFeasibleTossDirectionError,
    TossDirectionSelector,
)
from hitl_pmp.environments.tossing3d.types import TossFeasibilityGeometry

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="needs KINDER",
)
pytestmark = pytest.mark.xdist_group("tossing3d_toss_check_mirrors_controller")

FIXTURES = Path(__file__).parent / "fixtures" / "toss_check"
STANDOFFS = (1.4, 1.8, 2.2, 2.5)
DIRECTIONS = (0, 90, 180, 270)


def load(*, name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def environment(*, fixture: dict[str, Any]) -> Any:
    """The fixture's scene, its recorded state restored at MuJoCo's float64 precision."""
    if fixture["environment"] == "kitchen_recycling3d":
        from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
        from hitl_pmp.environments.kitchen_recycling3d.environment import (
            KitchenRecycling3DEnvironment,
        )

        env: Any = KitchenRecycling3DEnvironment(
            kitchen_config=KitchenConfig.model_validate(fixture["kitchen_config"]),
            scene_bg=False,
            visual_furniture=False,
        )
    else:
        from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

        env = Tossing3DEnvironment()
    env.reset_to_seed(seed=0)
    env.restore_plain_snapshot(plain=fixture["state"], dtype=np.float64)
    return env


class PlanningCalls:
    """Every `run_base_motion_planning` call made while active, wherever the function was
    imported to, each reduced to what the planner can observe of its arguments: the
    callback by its answers at fixed probe poses, geometry by value."""

    def __init__(self, *, plan: bool = True) -> None:
        import kinder_models.dynamic3d.utils as utils

        self.original = utils.run_base_motion_planning
        self.plan = plan
        self.calls: list[dict[str, Any]] = []
        self.patched: list[Any] = []

    def __enter__(self) -> "PlanningCalls":
        def spy(**kwargs: Any) -> Any:
            self.calls.append(PlanningCalls.observe(kwargs=kwargs))
            return self.original(**kwargs) if self.plan else None

        for module in list(sys.modules.values()):
            if getattr(module, "run_base_motion_planning", None) is self.original:
                module.run_base_motion_planning = spy  # type: ignore[attr-defined]
                self.patched.append(module)
        return self

    def __exit__(self, *exc: object) -> None:
        for module in self.patched:
            module.run_base_motion_planning = self.original

    @staticmethod
    def observe(*, kwargs: dict[str, Any]) -> dict[str, Any]:
        from spatialmath import SE2

        observed: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "state":
                observed[key] = {obj.name: np.asarray(value[obj]).tobytes() for obj in value}
            elif key == "target_base_pose":
                observed[key] = (float(value.x), float(value.y), float(value.theta()))
            elif key == "static_obstacles":
                observed[key] = sorted(repr(geom) for geom in value)
            elif key == "bounding_boxes":
                observed[key] = None if value is None else sorted(value.items())
            elif key != "body_collision_fn":
                observed[key] = value
        target = kwargs["target_base_pose"]
        rng = np.random.default_rng(0)
        (x_low, x_high), (y_low, y_high) = kwargs["x_bounds"], kwargs["y_bounds"]
        poses = [SE2(target.x, target.y, target.theta())] + [
            SE2(rng.uniform(x_low, x_high), rng.uniform(y_low, y_high), rng.uniform(-np.pi, np.pi))
            for _ in range(40)
        ]
        body = kwargs.get("body_collision_fn")
        observed["body_collision_fn"] = (
            None if body is None else [bool(body(pose)) for pose in poses]
        )
        return observed


class Controller:
    """The toss controller exactly as a dispatched toss grounds and resets it."""

    @staticmethod
    def planning_call(*, env: Any, standoff: float, direction: int) -> dict[str, Any]:
        backend = env.backend()
        snapshot = env.get_current_state().object_centric
        controller = backend._ground_controller(
            module="tossing",
            key="move_to_toss_location_and_toss",
            object_names=(backend.robot_name, backend.cube_name, backend.barrier_name),
            state=snapshot,
        )
        params = [standoff, TossDirectionSelector.rotation(direction_deg=direction), 3.0, 500.0]
        with PlanningCalls() as calls:
            try:
                controller.reset(snapshot, np.asarray(params))
            except BaseException as exc:  # noqa: BLE001 (TrajectorySamplingFailure included)
                if isinstance(exc, KeyboardInterrupt):
                    raise
            finally:
                sim = controller.controller._pybullet_sim
                if sim is not None:
                    sim.close()
        assert calls.calls, "the controller reached no base plan (its retract failed)"
        return calls.calls[0]


def check_calls(*, env: Any, standoff: float) -> list[dict[str, Any]]:
    TossDirectionSelector.clear_plan_cache()
    with PlanningCalls() as calls, contextlib.suppress(NoFeasibleTossDirectionError):
        TossDirectionSelector.select_for_state(state=env.get_current_state(), standoff=standoff)
    return calls.calls


def direction_of(*, env: Any, call: dict[str, Any], standoff: float) -> int:
    from kinder_models.dynamic3d.utils import (
        get_overhead_object_se2_pose,
        get_target_robot_pose_from_parameters,
    )

    snapshot = env.get_current_state().object_centric
    bin_pose = get_overhead_object_se2_pose(snapshot, snapshot.get_object_from_name("bin_0"))
    for direction in DIRECTIONS:
        distance, rotation = np.asarray(
            [standoff, TossDirectionSelector.rotation(direction_deg=direction)], dtype=np.float32
        )
        pose = get_target_robot_pose_from_parameters(bin_pose, distance, rotation)
        if np.allclose((pose.x, pose.y, pose.theta()), call["target_base_pose"], atol=1e-9):
            return direction
    raise AssertionError(f"no direction plans to {call['target_base_pose']}")


@needs_kinder
@pytest.mark.parametrize("name", ["kitchen_kc0_step174.json", "tossing3d_tc0_step114.json"])
def test_the_check_plans_with_the_dispatched_controllers_own_arguments(*, name: str) -> None:
    """Every argument the check hands the base planner is the controller's, the 3-D
    collision callback compared by its answers; only smoothing is off in the check."""
    env = environment(fixture=load(name=name))
    try:
        compared = 0
        for standoff in STANDOFFS:
            for call in check_calls(env=env, standoff=standoff):
                direction = direction_of(env=env, call=call, standoff=standoff)
                controller = Controller.planning_call(
                    env=env, standoff=round(standoff, 6), direction=direction
                )
                smoothing = call.pop("smooth_amt", None)
                assert call == controller, (standoff, direction)
                assert smoothing == 0
                compared += 1
        assert compared >= 4
    finally:
        env.close()


@needs_kinder
def test_tossing3ds_controller_plans_with_upstreams_arguments() -> None:
    """Tossing3D's controller builds its base plan's arguments itself, so the check can
    reuse them; they are upstream's, call for call, at the pinned commit."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )
    from kinder_models.dynamic3d.utils import PyBulletSim

    from hitl_pmp.environments.tossing3d.toss_controller import Tossing3DTossController

    env = environment(fixture=load(name="tossing3d_tc0_step114.json"))
    try:
        snapshot = env.get_current_state().object_centric
        ours = Tossing3DTossController.for_snapshot(snapshot=snapshot)
        upstream = MoveToTossLocationAndTossController(
            ours.objects, pybullet_sim=PyBulletSim(snapshot)
        )
        for standoff in STANDOFFS:
            for direction in DIRECTIONS:
                params = np.asarray(
                    [standoff, TossDirectionSelector.rotation(direction_deg=direction)],
                    dtype=np.float32,
                )
                observed = []
                for controller in (ours, upstream):
                    # Arguments only: the planner itself is not run.
                    with PlanningCalls(plan=False) as calls:
                        try:
                            controller._plan_base_motion(snapshot, params, 0.025, np.pi / 8, None)
                        except BaseException as exc:  # noqa: BLE001
                            if isinstance(exc, KeyboardInterrupt):
                                raise
                    observed.append(calls.calls)
                assert observed[0] == observed[1], (standoff, direction)
    finally:
        env.close()


def test_the_plan_cache_tells_apart_states_with_one_planar_geometry(*, monkeypatch) -> None:
    """The base planner's 3-D check reads what the planar geometry does not (the arm,
    the held cube's grasp), so two snapshots with one geometry are asked about apart."""

    class Named(BaseModel, frozen=True):
        name: str

    robot = Named(name="robot")
    snapshots = [{robot: np.array([0.0, float(joint)])} for joint in (0.1, 0.2)]
    asked: list[Any] = []

    def stub(*, snapshot: Any, distance: float, rotation: float) -> str | None:
        asked.append(snapshot)
        return None

    monkeypatch.setattr(TossDirectionSelector, "base_plan_failure", stub)
    TossDirectionSelector.clear_plan_cache()
    geometry = TossFeasibilityGeometry(
        robot_pose=(0.0, 0.0, 0.0),
        robot_size=(0.5, 0.5),
        bin_pose=(2.0, 0.0, 0.0),
        obstacles=(),
        sampling_x_bounds=(-3.0, 3.0),
        sampling_y_bounds=(-3.0, 3.0),
    )
    for snapshot in [*snapshots, *snapshots]:
        TossDirectionSelector.select(geometry=geometry, snapshot=snapshot, standoff=1.5)
    assert asked == snapshots
