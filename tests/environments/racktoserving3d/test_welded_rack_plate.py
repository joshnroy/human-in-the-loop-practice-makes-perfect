"""The rack and the plate are welded to the counter (Josh, 2026-11-06).

Before, both were free bodies: a sweep stroke, a food drag or a pick could slide the
rack ("rack drags") or the plate. Each is now a MuJoCo mocap body, as KINDER's own fixed
`Bin` is: no degrees of freedom, so no contact can move it, yet its pose is still read
and written (reset, restore, replay) through the same object API.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator"),
    pytest.mark.xdist_group("racktoserving3d_test_welded_rack_plate"),
]

WELDED = ("rack_0", "box_0")


def _poses(*, backend) -> dict[str, np.ndarray]:
    """Each welded body's world pose, as the simulator places it and as KINDER reads it."""
    model, data = backend.model(), backend.data()
    out = {}
    for name in WELDED:
        body = model.body(name).id
        position, quaternion = backend.core().get_object(name).get_pose()
        out[name] = np.r_[data.xpos[body], data.xquat[body], position, quaternion]
    return out


def _assert_unchanged(*, before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    for name in WELDED:
        np.testing.assert_allclose(after[name], before[name], rtol=0, atol=1e-9, err_msg=name)


@pytest.fixture(name="env", scope="module")
def _env():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment()
    env.reset_to_seed(seed=0)
    yield env
    env.close()


def test_the_rack_and_the_plate_have_no_degrees_of_freedom(*, env) -> None:
    model = env.backend().model()
    for name in WELDED:
        body = model.body(name).id
        assert model.body_dofnum[body] == 0, name
        assert model.body_mocapid[body] >= 0, name


def test_food_dropped_and_thrown_at_them_leaves_the_rack_and_plate_still(*, env) -> None:
    env.reset_to_seed(seed=0)
    backend = env.backend()
    before = _poses(backend=backend)
    rack, plate = backend.support(name="rack_0"), backend.support(name="box_0")
    config = backend.config
    on_counter = config.counter_height + config.dumpling_height / 2 + 0.002
    upright = np.array([1.0, 0, 0, 0])
    placements = {
        # Dropped from 10 cm onto the rack and onto the plate.
        "cube_0": (np.array([rack.x_min + 0.06, 0.0, rack.top_z + 0.10]), upright),
        "cube_1": (np.array([(plate.x_min + plate.x_max) / 2, 0.0, plate.top_z + 0.10]), upright),
        # Thrown along the counter into the rack's side and the plate's rim.
        "cube_2": (
            np.array([(rack.x_min + rack.x_max) / 2, rack.y_min - 0.06, on_counter]),
            upright,
        ),
        "cube_3": (np.array([plate.x_min - 0.05, 0.12, on_counter]), upright),
        "cube_4": (np.array([1.0, 0.35, on_counter]), upright),
    }
    backend._commit(placements=placements)
    model, data = backend.model(), backend.data()
    for name, velocity in (("cube_2", (0.0, 2.0, 0.0)), ("cube_3", (2.0, 0.0, 0.0))):
        dof = model.jnt_dofadr[model.joint(f"{name}_joint").id]
        data.qvel[dof : dof + 3] = velocity
    env.rack_controllers().shelf.hold(count=300, label="food_into_rack_and_plate")
    _assert_unchanged(before=before, after=_poses(backend=backend))


def test_a_sweep_stroke_leaves_the_rack_and_plate_still(*, env) -> None:
    env.reset_to_seed(seed=0)
    backend = env.backend()
    before = _poses(backend=backend)
    env.take_action(action=np.array([float(SKILL_IDS["PickScraper"]), 0, 0, 0, 0]))
    assert backend.held(name="wiper_0"), env._last_skill_error
    env.take_action(action=np.array([float(SKILL_IDS["Sweep"]), 0, 0, 0, 0.08]))
    assert env.last_controller_steps() > 0, env._last_skill_error
    _assert_unchanged(before=before, after=_poses(backend=backend))


def test_resets_and_human_interventions_keep_the_welded_pose(*, env) -> None:
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    initial = env.reset_to_seed(seed=0)
    backend = env.backend()
    before = _poses(backend=backend)
    env.reset_movables(destination="human_batch_return")
    _assert_unchanged(before=before, after=_poses(backend=backend))
    env.reset_movables(destination="human_tool_recovery")
    _assert_unchanged(before=before, after=_poses(backend=backend))
    env.reset_to_seed(seed=7)
    _assert_unchanged(before=before, after=_poses(backend=backend))
    env.set_state(state=initial)
    _assert_unchanged(before=before, after=_poses(backend=backend))
    other = RackServingEnvironment()
    try:
        other.reset_to_seed(seed=3)
        _assert_unchanged(before=before, after=_poses(backend=other.backend()))
    finally:
        other.close()


def test_a_fresh_copy_of_the_simulator_data_sees_the_same_pose(*, env) -> None:
    # Scratch kinematics (IK, placement and clearance checks) build a fresh MjData
    # from the model and copy only qpos; a mocap body then sits at the model's own
    # body pose, so that pose must be the live one.
    import mujoco

    env.reset_to_seed(seed=0)
    backend = env.backend()
    model = backend.model()
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = backend.data().qpos
    mujoco.mj_forward(model, scratch)
    for name in WELDED:
        body = model.body(name).id
        live = backend.data()
        np.testing.assert_allclose(scratch.xpos[body], live.xpos[body], rtol=0, atol=1e-9)
        np.testing.assert_allclose(scratch.xquat[body], live.xquat[body], rtol=0, atol=1e-9)


def test_motion_state_carries_the_welded_pose(*, env) -> None:
    env.reset_to_seed(seed=0)
    backend = env.backend()
    state = backend.motion_state()
    poses = _poses(backend=backend)
    for name in WELDED:
        np.testing.assert_allclose(state[name][-7:], poses[name][:7], atol=1e-9, err_msg=name)


def test_the_replay_renderer_places_a_recorded_rack_and_plate(*, env) -> None:
    from hitl_pmp.environments.racktoserving3d.recording import RackServingReplayRenderer

    env.reset_to_seed(seed=0)
    state = env.backend().motion_state()
    # A recording made before the weld may carry a rack that slid: draw it where it was.
    state["rack_0"][-7] += 0.01
    motion = {
        "stride": RackServingReplayRenderer.motion_stride(),
        "fps": RackServingReplayRenderer.MOTION_FPS,
    }
    renderer = RackServingReplayRenderer(scene={"motion": motion}, width=160, height=120)
    try:
        frame = renderer.show(state=state)
        assert frame.shape == (120, 160, 3)
        back = renderer.current_state()
        for name in WELDED:
            np.testing.assert_allclose(back[name][-7:], state[name][-7:], atol=1e-6, err_msg=name)
    finally:
        renderer.close()
