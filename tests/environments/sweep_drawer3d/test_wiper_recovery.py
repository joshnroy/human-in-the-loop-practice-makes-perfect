"""An invalid initial wiper placement must be recovered by executed robot motion."""

import importlib.util

import numpy as np
import pytest

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra",
)


@needs_kinder
def test_wiper_reacquisition_requires_an_observed_lift() -> None:
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.stock_skills import StockSweepSkills
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    session = SweepDrawerSession(seed=28)
    try:
        StockSweepSkills.attempt(session=session, phase="attempt_1")
        before = session.position(name=S.WIPER).copy()
        prims = SweepDrawerSelfReset(session=session).primitives
        prims.recover_wiper()
        assert session.position(name=S.WIPER)[2] > before[2] + 0.05
        ee = np.asarray(prims.scene.ee_now().position)
        assert np.linalg.norm(ee - session.position(name=S.WIPER)) < 0.15
    finally:
        session.close()


@needs_kinder
def test_invalid_initial_wiper_gets_a_supported_target_without_moving_the_world() -> None:
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    session = SweepDrawerSession(seed=28)
    try:
        before = session.position(name=S.WIPER).copy()
        assert before[2] < 0.02
        reset = SweepDrawerSelfReset(session=session)
        reset.primitives.choose_wiper_parking_pose()
        target, _ = session.wiper_parking_pose()
        assert target[2] == pytest.approx(S.COUNTER_TOP, abs=0.005)
        assert np.linalg.norm(target[:2] - before[:2]) > 0.1
        np.testing.assert_array_equal(session.position(name=S.WIPER), before)
        assert session.ticks == 0
    finally:
        session.close()


@needs_kinder
def test_valid_initial_wiper_parking_pose_is_preserved() -> None:
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    session = SweepDrawerSession(seed=0)
    try:
        SweepDrawerSelfReset(session=session).primitives.choose_wiper_parking_pose()
        target, quat = session.wiper_parking_pose()
        initial, initial_quat = session.initial_pose(name=S.WIPER)
        np.testing.assert_array_equal(target, initial)
        assert quat == initial_quat
    finally:
        session.close()


@needs_kinder
def test_floor_wiper_is_physically_picked_and_returned_to_the_counter() -> None:
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.stock_skills import StockSweepSkills

    session = SweepDrawerSession(seed=28)
    try:
        StockSweepSkills.attempt(session=session, phase="attempt_1")
        start = session.ticks
        result = SweepDrawerSelfReset(session=session).run()
        assert result.wiper_on_counter
        assert result.wiper_xy_error < 0.02
        assert result.success
        assert session.ticks > start
        assert any(s.name == "recover_wiper" and s.success for s in session.steps)
    finally:
        session.close()


@needs_kinder
def test_planning_rejects_arm_poses_outside_physical_joint_limits() -> None:
    import mujoco

    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    session = SweepDrawerSession(seed=0)
    try:
        scene = SweepDrawerSelfReset(session=session).primitives.scene
        assert scene.within_arm_limits(arm=S.HOME)
        for number in (2, 4, 6):
            joint = mujoco.mj_name2id(
                session.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_joint_{number}"
            )
            for bound, direction in zip(
                session.mj_model.jnt_range[joint], (-1.0, 1.0), strict=True
            ):
                arm = np.array(S.HOME)
                arm[number - 1] = bound + direction * 0.04
                assert not scene.within_arm_limits(arm=arm)
                assert scene.in_collision(joints=scene.fingers(arm=arm), bodies=set())
    finally:
        session.close()


@needs_kinder
def test_collision_wiper_tracks_observed_released_pose() -> None:
    from pybullet_helpers.geometry import get_pose

    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    session = SweepDrawerSession(seed=0)
    try:
        scene = SweepDrawerSelfReset(session=session).primitives.scene
        scene.sync()
        actual = get_pose(scene.wiper_body, scene.cid)
        np.testing.assert_allclose(actual.position, session.position(name=S.WIPER), atol=1e-6)
    finally:
        session.close()


@needs_kinder
def test_wiper_leaves_counter_contact_before_stowing_for_return() -> None:
    import json
    from pathlib import Path

    import mujoco

    from hitl_pmp.environments.sweep_drawer3d.primitives import WiperHold
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    session = SweepDrawerSession(seed=11)
    try:
        # The old setup depended on task-controller collision overrides removed
        # from production. Restore its recorded contact state, then test the
        # recovery itself with current native physics and collision checks.
        fixture = json.loads(Path(__file__).with_name("held_wiper_contact_seed11.json").read_text())
        assert session.replay_header() == fixture["header"]
        state = session.state.copy()
        for name, values in fixture["objects"].items():
            state.data[state.get_object_from_name(name)] = np.asarray(values)
        core = session.env.unwrapped._object_centric_env
        core.set_state(state)
        session.mj_data.qpos[:] = fixture["qpos"]
        session.mj_data.qvel[:] = fixture["qvel"]
        session.mj_data.ctrl[:] = fixture["ctrl"]
        mujoco.mj_forward(session.mj_model, session.mj_data)
        session._state = core._get_state()
        reset = SweepDrawerSelfReset(session=session)
        assert WiperHold.in_hand(
            gripper=np.asarray(reset.primitives.scene.ee_now().position),
            wiper=session.position(name=S.WIPER),
        )
        reset.primitives.park_wiper()
        home, _ = session.wiper_parking_pose()
        observed = session.position(name=S.WIPER)
        assert abs(observed[2] - S.COUNTER_TOP) < 0.01
        assert np.linalg.norm(observed[:2] - home[:2]) < 0.02
    finally:
        session.close()
