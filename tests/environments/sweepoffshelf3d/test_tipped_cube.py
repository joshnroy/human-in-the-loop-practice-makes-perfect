"""A cube tipped onto a side face, its own x axis vertical, sits at Euler gimbal lock.

scipy's 'xyz' Euler yaw is 0 there, while KINDER's `get_overhead_object_se2_pose`
(spatialmath rpy) reads +-180 deg. `KinderSkills.facing_params` subtracted the first and
`PickShelfController.reset` adds back the second, so KINDER drove the base to the mirror
image of the base the grasp was checked from: its base planner found nothing, every
candidate failed, and the skill refused at 0 control steps. Separately, the grasp faces
were read off the cube's x axis, which for a tipped cube points up or down.

The replay states (`data/gimbal_lock_states.jsonl`) are the exact recorded states of
three such refusals from the 2026-10-23 shelf zero-step diagnosis: EES seed 0 seq 18
and determinized always-ask seed 0 seq 96 (ReturnCubeToShelfFromBox), and EES seed 0
seq 74 (PickCubeFromShelf, refused over 100 times in a row)."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import SkillTraces

STATES = Path(__file__).resolve().parent / "data/gimbal_lock_states.jsonl"
NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="needs KINDER",
)

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [pytest.mark.xdist_group("sweepoffshelf3d_test_tipped_cube")]

YAWS_DEG = (0.0, 17.0, 30.0, 90.0, 135.0, -60.0, 179.0)


def _tipped(*, yaw_deg: float, down: bool = False) -> np.ndarray:
    """(w, x, y, z): the cube's own x axis straight up (or down), turned `yaw_deg`."""
    pitch = np.pi / 2 if down else -np.pi / 2
    x, y, z, w = Rotation.from_euler("ZY", [np.deg2rad(yaw_deg), pitch]).as_quat()
    return np.array([w, x, y, z])


# The recorded cube quaternions of the three replay states, (w, x, y, z).
RECORDED = (
    np.array([-0.0004957522978574497, -0.7071066056811712, -0.000495754449192737, 0.7071066091]),
    np.array([-0.0006943797331053658, 0.7071064370430075, -0.0006943785922027949, -0.70710644]),
    np.array([-0.0006545380088544532, 0.707106478247792, 0.0006545380088545053, 0.7071064782]),
)
QUATERNIONS = (
    *(_tipped(yaw_deg=y) for y in YAWS_DEG),
    *(_tipped(yaw_deg=y, down=True) for y in YAWS_DEG),
    *RECORDED,
    np.array([1.0, 0.0, 0.0, 0.0]),  # upright, for contrast
)


class _Object:
    def __init__(self, *, name: str) -> None:
        self.name = name

    @staticmethod
    def is_instance(_: Any) -> bool:
        return True


class _State:
    """An ObjectCentricState with one object, as KINDER's helpers read it."""

    def __init__(self, *, centre: np.ndarray, quaternion: np.ndarray) -> None:
        self.obj = _Object(name="cube_0")
        w, x, y, z = (float(v) for v in quaternion)
        self.values = {"x": centre[0], "y": centre[1], "qw": w, "qx": x, "qy": y, "qz": z}

    def get_object_from_name(self, name: str) -> _Object:  # noqa: PLR0917, KINDER calls it so
        assert name == self.obj.name
        return self.obj

    def get(self, _: Any, key: str) -> float:  # noqa: PLR0917, KINDER calls it so
        return float(self.values[key])


class _Core:
    def __init__(self, *, centre: np.ndarray, quaternion: np.ndarray) -> None:
        self.pose = (list(centre), list(quaternion))

    def get_object(self, _: str) -> Any:
        pose = self.pose

        class _Body:
            @staticmethod
            def get_pose() -> tuple[list[float], list[float]]:
                return pose

        return _Body()


class _Shelf:
    def __init__(self, *, centre: np.ndarray, quaternion: np.ndarray) -> None:
        self.core = _Core(centre=centre, quaternion=quaternion)


def _kinder(*, centre: np.ndarray, quaternion: np.ndarray) -> KinderSkills:
    kinder = object.__new__(KinderSkills)
    state = _State(centre=centre, quaternion=quaternion)
    kinder.state = lambda: state  # type: ignore[method-assign]
    kinder.shelf = _Shelf(centre=centre, quaternion=quaternion)  # type: ignore[assignment]
    return kinder


def _horizontal_axes(*, quaternion: np.ndarray) -> list[np.ndarray]:
    axes = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    return [axes[:, i] for i in range(3) if abs(axes[2, i]) < 0.5]


@NEEDS_KINDER
@pytest.mark.parametrize("quaternion", QUATERNIONS, ids=lambda q: np.round(q, 3).tolist().__str__())
@pytest.mark.parametrize("heading_deg", [0.0, 40.0, -110.0])
def test_kinder_drives_to_the_base_facing_params_was_given(*, quaternion, heading_deg):
    """`PickShelfController` turns (distance, rotation) back into a base with KINDER's
    own yaw of the cube; that base must be the one `facing_params` was asked for."""
    from kinder_models.dynamic3d.utils import (
        get_overhead_object_se2_pose,
        get_target_robot_pose_from_parameters,
    )

    centre = np.array([1.25, -0.07, 0.02])
    heading = np.deg2rad(heading_deg)
    base = np.array([centre[0] - 0.6 * np.cos(heading), centre[1] - 0.6 * np.sin(heading), heading])
    kinder = _kinder(centre=centre, quaternion=quaternion)
    state = kinder.state()
    params = kinder.facing_params(base=base, centre=centre, cube="cube_0")
    target = get_target_robot_pose_from_parameters(
        get_overhead_object_se2_pose(state, state.get_object_from_name("cube_0")), *params
    )
    assert np.allclose([target.x, target.y], base[:2], atol=1e-6), (target, base)


@pytest.mark.parametrize("quaternion", QUATERNIONS, ids=lambda q: np.round(q, 3).tolist().__str__())
def test_floor_grasps_close_on_a_pair_of_the_cubes_side_faces(*, quaternion, monkeypatch):
    """Every floor/box grasp's pads close along one of the cube's horizontal axes (a
    rolled grasp's tilted up out of the horizontal plane), however the cube lies: on a
    tipped cube the x axis is vertical, so faces read off it put the pads across the
    cube at an arbitrary angle."""
    centre = np.array([1.25, -0.07, 0.02])
    picks = FloorPicks(kinder=_kinder(centre=centre, quaternion=quaternion))
    monkeypatch.setattr(picks, "neighbour_side", lambda **_: 0.0)
    hands = picks.hands(cube="cube_0", centre=centre)
    assert hands
    horizontal = _horizontal_axes(quaternion=quaternion)
    for _, hand in hands:
        closing = np.array([hand[0, 0], hand[1, 0], 0.0])
        closing /= np.linalg.norm(closing)
        assert max(abs(float(closing @ axis)) for axis in horizontal) > 0.999, closing


@pytest.mark.parametrize("quaternion", QUATERNIONS, ids=lambda q: np.round(q, 3).tolist().__str__())
def test_shelf_grasps_close_on_a_pair_of_the_cubes_side_faces(*, quaternion):
    centre = np.array([1.53, -0.2, 0.29])
    hands = _kinder(centre=centre, quaternion=quaternion).shelf_hands(cube="cube_0", centre=centre)
    assert hands
    horizontal = _horizontal_axes(quaternion=quaternion)
    for hand, _ in hands:
        closing = hand[:3, 0]
        assert max(abs(float(closing @ axis)) for axis in horizontal) > 0.999, closing


# --- replay of the recorded refusals -----------------------------------------------------


@pytest.fixture(name="env", scope="module")
def _env():
    # The states were captured in the lab2 scene; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


@NEEDS_KINDER
@pytest.mark.parametrize(
    "state_id",
    [
        "gimbal-ees0-seq18-box-return",
        "gimbal-det-always-ask0-seq96-box-return",
        "gimbal-ees0-seq74-shelf-pick-livelock",
    ],
)
def test_a_recorded_zero_step_refusal_on_a_tipped_cube_now_succeeds(*, env, states, state_id):
    state = states[state_id]
    StateCapture.restore(env=env, state=state)
    case = state.cases[0]
    env.take_action(action=StateCapture.action(case=case))
    trace = SkillTraces.last(env=env)
    assert trace is not None
    assert trace.controller_steps > 0, env.last_skill_error()
    assert trace.outcome == "success", (trace.outcome, env.last_skill_error())
