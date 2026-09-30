"""Approved native planner corrections preserve real collision rejection."""

import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
from hitl_pmp.environments.sweep_drawer3d.stock_skills import StockSweepSkills
from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S


def _plan(*, state):
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as native
    from kinder_models.dynamic3d.utils import WORLD_X_BOUNDS, WORLD_Y_BOUNDS

    target = native.get_target_robot_pose_from_parameters(
        native.get_overhead_object_se2_pose(state, state.get_object_from_name(S.WIPER)), 0.7, -np.pi
    )
    return native.run_base_motion_planning(
        state=state,
        target_base_pose=target,
        x_bounds=WORLD_X_BOUNDS,
        y_bounds=WORLD_Y_BOUNDS,
        seed=0,
    )


def test_valid_counter_wiper_does_not_block_chassis_projection():
    session = SweepDrawerSession(seed=0)
    try:
        before = session.mj_data.qpos.copy()
        assert _plan(state=session.state) is not None
        np.testing.assert_array_equal(before, session.mj_data.qpos)
    finally:
        session.close()


@pytest.mark.parametrize("where", ["floor", "arm"])
def test_real_chassis_and_arm_collisions_still_block_navigation(*, where):
    session = SweepDrawerSession(seed=0)
    try:
        state = session.state.copy()
        robot = state.get_object_from_name(S.ROBOT)
        wiper = state.get_object_from_name(S.WIPER)
        base = np.array([state.get(robot, "pos_base_x"), state.get(robot, "pos_base_y")])
        if where == "floor":
            position = (*base, 0.08)
        else:
            yaw = state.get(robot, "pos_base_rot")
            arm = base + 0.12 * np.array([np.cos(yaw), np.sin(yaw)])
            position = (*arm, 0.48)
        for key, value in zip(("x", "y", "z"), position, strict=True):
            state.set(wiper, key, value)
        assert _plan(state=state) is None
    finally:
        session.close()


def test_native_wiper_grasp_has_collision_free_approach_and_retraction():
    from bilevel_planning.trajectory_samplers.trajectory_sampler import TrajectorySamplingFailure
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as native

    session = SweepDrawerSession(seed=1)
    sim = None
    try:
        StockSweepSkills.run(
            session=session,
            skill="open_drawer",
            label="open",
            phase="diagnostic",
            params=np.array([0.8, -np.pi]),
        )
        state = session.state
        sim = native.PyBulletSim(state)
        controller = native.create_lifted_controllers(
            session.env.action_space, None, pybullet_sim=sim
        )["pick_wiper"].ground(
            tuple(state.get_object_from_name(n) for n in (S.ROBOT, S.WIPER, S.DRAWER, *S.CUBES))
        )
        try:
            controller.reset(state, np.array([0.7, -np.pi]))
        except TrajectorySamplingFailure:
            pytest.fail("Native grasp target or its approach/retraction remains collision-blocked")
    finally:
        if sim is not None:
            sim.close()
        session.close()


def test_carried_wiper_keeps_object_collision_checks():
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as native
    from kinder_models.dynamic3d.utils import WORLD_X_BOUNDS, WORLD_Y_BOUNDS
    from spatialmath import SE2

    session = SweepDrawerSession(seed=0)
    try:
        state = session.state.copy()
        wiper = state.get_object_from_name(S.WIPER)
        cube = state.get_object_from_name(S.CUBES[0])
        # Place a cube through the wiper handle; treating the tool as carried
        # must not exempt the tool from object collisions.
        for key in ("x", "y", "z"):
            state.set(cube, key, state.get(wiper, key))
        state.set(cube, "z", state.get(wiper, "z") + 0.0075)
        base = session.base()
        result = native.run_base_motion_planning(
            state=state,
            target_base_pose=SE2(*base),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=[S.WIPER],
        )
        assert result is None
    finally:
        session.close()


def test_sweep_approach_clears_valid_seed_cube_layout():
    from bilevel_planning.trajectory_samplers.trajectory_sampler import TrajectorySamplingFailure
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as native

    session = SweepDrawerSession(seed=10001)
    sim = None
    try:
        for skill, distance in (("open_drawer", 0.8), ("pick_wiper", 0.7)):
            result = StockSweepSkills.run(
                session=session,
                skill=skill,
                label=skill,
                phase="diagnostic",
                params=np.array([distance, -np.pi]),
            )
            assert result.success
        state = session.state
        sim = native.PyBulletSim(state)
        controller = native.create_lifted_controllers(
            session.env.action_space, None, pybullet_sim=sim
        )["sweep"].ground(
            tuple(state.get_object_from_name(n) for n in (S.ROBOT, S.WIPER, S.DRAWER, *S.CUBES))
        )
        try:
            controller.reset(state, np.array([0.70, -np.pi]))
        except TrajectorySamplingFailure:
            pytest.fail("Sweep approach remains collision-blocked by a valid-start cube")
    finally:
        if sim is not None:
            sim.close()
        session.close()


def test_open_drawer_blocks_native_base_endpoint():
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as native
    from kinder_models.dynamic3d.utils import WORLD_X_BOUNDS, WORLD_Y_BOUNDS
    from spatialmath import SE2

    session = SweepDrawerSession(seed=10001)
    try:
        for skill, distance in (("open_drawer", 0.8), ("pick_wiper", 0.7)):
            assert StockSweepSkills.run(
                session=session,
                skill=skill,
                label=skill,
                phase="diagnostic",
                params=np.array([distance, -np.pi]),
            ).success
        state = session.state
        assert (
            native.run_base_motion_planning(
                state=state,
                target_base_pose=SE2(1.30226, -0.12266, np.pi),
                x_bounds=WORLD_X_BOUNDS,
                y_bounds=WORLD_Y_BOUNDS,
                seed=0,
                disable_collision_objects=[S.WIPER],
            )
            is None
        )
    finally:
        session.close()


def test_drawer_contact_allowance_keeps_arm_and_other_drawers_blocked():
    import pybullet as p
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as native
    from kinder_models.dynamic3d.sweep3D.drawer_collision import target_drawer_contact_constraint
    from pybullet_helpers.geometry import Pose, set_pose

    session = SweepDrawerSession(seed=10001)
    sim = None
    try:
        state = session.state
        sim = native.PyBulletSim(state)
        controller = native.create_lifted_controllers(
            session.env.action_space, None, pybullet_sim=sim
        )["open_drawer"].ground(
            tuple(state.get_object_from_name(n) for n in (S.ROBOT, S.WIPER, S.DRAWER, *S.CUBES))
        )
        controller.reset(state, np.array([0.8, -np.pi]))
        assert controller._current_arm_joint_plan is not None
        sim.robot.set_joints(controller._current_arm_joint_plan[-1])
        all_bodies = set(sim._sweep_drawer_body_names)
        others, allowed_contact = target_drawer_contact_constraint(sim, all_bodies, S.DRAWER)
        assert others
        assert allowed_contact(sim.get_robot_joints())
        target_bodies = all_bodies - others
        assert any(
            p.getClosestPoints(sim.robot.robot_id, body, 0, physicsClientId=sim.physics_client_id)
            for body in target_bodies
        )
        body = next(iter(target_bodies))
        arm_pose = p.getLinkState(sim.robot.robot_id, 5, physicsClientId=sim.physics_client_id)[0]
        set_pose(body, Pose(tuple(arm_pose)), sim.physics_client_id)
        assert not allowed_contact(sim.get_robot_joints())
    finally:
        if sim is not None:
            sim.close()
        session.close()
