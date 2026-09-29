"""Millimeter-level mount mismatch must not corrupt floor tool IK."""

import mujoco
import numpy as np
import pybullet
from pybullet_helpers.geometry import Pose, multiply_poses

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPlanningScene
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_planning_mount_matches_native_and_preserves_physical_state() -> None:
    session = SweepSimpleSession(seed=0)
    scene = FloorPlanningScene(session=session)
    try:
        root = mujoco.mj_name2id(session.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot_gen3/base_link")
        actual = session.mj_data.xpos[root].copy()
        planned = pybullet.getLinkState(
            scene.robot.robot_id, 0, computeForwardKinematics=True, physicsClientId=scene.cid
        )[4]
        np.testing.assert_allclose(planned, actual, atol=1e-6)
        before = session.mj_data.qpos.copy()
        base = (1.8, 1.9, 0.3)
        bx, by, yaw = session.base()
        original_base = Pose.from_rpy((bx, by, 0.0), (0.0, 0.0, yaw))
        quaternion = session.mj_data.xquat[root][[1, 2, 3, 0]]
        relative = multiply_poses(original_base.invert(), Pose(tuple(actual), tuple(quaternion)))
        expected = multiply_poses(
            Pose.from_rpy((base[0], base[1], 0.0), (0.0, 0.0, base[2])), relative
        )
        scene.sync(base=base)
        planned = pybullet.getLinkState(
            scene.robot.robot_id, 0, computeForwardKinematics=True, physicsClientId=scene.cid
        )[4]
        np.testing.assert_allclose(planned, expected.position, atol=1e-6)
        np.testing.assert_array_equal(session.mj_data.qpos, before)
        np.testing.assert_array_equal(session.mj_data.xpos[root], actual)
    finally:
        scene._sim.close()
        session.close()


def test_native_chassis_mesh_world_geometry_and_hypothetical_pose() -> None:
    """Collision proxy follows native CAD and moves rigidly with hypothetical base."""
    from pybullet_helpers.geometry import get_pose

    session = SweepSimpleSession(seed=0)
    scene = FloorPlanningScene(session=session)
    try:
        assert len(scene._native_chassis) == 2
        assert scene.chassis_body not in scene.bodies()
        original_base = session.base()
        original_qpos = session.mj_data.qpos.copy()
        for geom, body in scene._native_chassis:
            actual = session.mj_data.geom_xpos[geom]
            np.testing.assert_allclose(get_pose(body, scene.cid).position, actual, atol=1e-6)
            model = session.mj_model
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start : start + count]
            world = vertices @ session.mj_data.geom_xmat[geom].reshape(3, 3).T + actual
            lo, hi = pybullet.getAABB(body, physicsClientId=scene.cid)
            # AABB is conservative after rotating the local mesh bounds.
            assert np.all(np.asarray(lo) <= world.min(axis=0) + 1e-6)
            assert np.all(np.asarray(hi) >= world.max(axis=0) - 1e-6)
            count_planned, planned_vertices = pybullet.getMeshData(
                body, -1, flags=pybullet.MESH_DATA_SIMULATION_MESH, physicsClientId=scene.cid
            )
            assert count_planned > 0
            planned_world = (
                np.asarray(planned_vertices) @ session.mj_data.geom_xmat[geom].reshape(3, 3).T
                + actual
            )
            np.testing.assert_allclose(planned_world.min(axis=0), world.min(axis=0), atol=1e-5)
            np.testing.assert_allclose(planned_world.max(axis=0), world.max(axis=0), atol=1e-5)
        scene.sync(base=(original_base[0] + 0.2, original_base[1] - 0.1, original_base[2]))
        for geom, body in scene._native_chassis:
            np.testing.assert_allclose(
                get_pose(body, scene.cid).position,
                session.mj_data.geom_xpos[geom] + [0.2, -0.1, 0],
                atol=1e-6,
            )
        np.testing.assert_array_equal(session.mj_data.qpos, original_qpos)
    finally:
        scene._sim.close()
        session.close()
