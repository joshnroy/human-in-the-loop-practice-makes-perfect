"""Millimeter-level mount mismatch must not corrupt floor tool IK."""

import mujoco
import numpy as np
import pybullet
import pytest
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


def test_closed_grasp_pad_centers_match_native_articulation() -> None:
    session = SweepSimpleSession(seed=0)
    scene = FloorPlanningScene(session=session)
    model, data = session.mj_model, session.mj_data
    try:
        for angle in (0.35, 0.63):
            for side in ("left", "right"):
                for name, value in (
                    ("driver", angle),
                    ("spring_link", angle),
                    ("follower", -angle),
                    ("coupler", 0.0),
                ):
                    joint = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{side}_{name}_joint"
                    )
                    data.qpos[model.jnt_qposadr[joint]] = value
            mujoco.mj_forward(model, data)
            session.state.set(session.state.get_object_from_name("robot"), "pos_gripper", 1.0)
            scene.sync()
            scene.robot.set_joints(scene.planning_fingers(arm=session.arm(), state=0.5))
            for side, link in (("left", 14), ("right", 19)):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{side}_pad")
                boxes = [
                    geom
                    for geom in range(model.ngeom)
                    if model.geom_bodyid[geom] == body
                    and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
                ]
                assert len(boxes) == 2
                native_center = data.geom_xpos[boxes].mean(axis=0)
                planned_center = pybullet.getLinkState(
                    scene.robot.robot_id,
                    link,
                    computeForwardKinematics=True,
                    physicsClientId=scene.cid,
                )[4]
                np.testing.assert_allclose(planned_center, native_center, atol=0.001)
    finally:
        scene._sim.close()
        session.close()


def test_observed_soft_limit_roundoff_does_not_broaden_planned_limits(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorded 10-microradian soft-limit overshoot must not invent an obstacle."""
    measured = np.array([-0.81943, 2.24001, 2.97831, -0.7399, -2.01082, 2.01093, 1.70926])
    session = SweepSimpleSession(seed=0)
    monkeypatch.setattr(SweepSimpleSession, "arm", lambda self: measured.copy())
    scene = FloorPlanningScene(session=session)
    try:
        assert not scene.within_arm_limits(arm=measured)
        assert scene.plan_arm(goal=measured, bodies=set()) is None
        assert not scene.in_collision(joints=scene.planning_fingers(arm=measured), bodies=set())
        planned = measured.copy()
        planned[0] += 0.1
        assert scene.in_collision(joints=scene.planning_fingers(arm=planned), bodies=set())
        measured[1] += 0.001
        assert not scene.within_arm_limits(arm=measured)
        assert scene.plan_arm(goal=measured, bodies=set()) is None
        assert not scene.in_collision(joints=scene.planning_fingers(arm=measured), bodies=set())
        proposed = measured.copy()
        proposed[0] += 0.1
        assert scene.in_collision(joints=scene.planning_fingers(arm=proposed), bodies=set())
    finally:
        scene._sim.close()
        session.close()


def test_native_distance_refines_mesh_padding_without_mutating_physics() -> None:
    """A measured 1.93-mm native clearance was reported as30-micron proxy overlap."""
    session = SweepSimpleSession(seed=0)
    scene = FloorPlanningScene(session=session)
    try:
        arm = np.array([-0.82547, 2.21671, 2.81530, -0.92952, -8.28174, 1.72315, 1.50567])
        joints = scene.planning_fingers(arm=arm, state=0.5)
        chassis_geom, proxy = scene._native_chassis[0]
        before = session.mj_data.qpos.copy()
        distance = scene.native_chassis_distance(link=6, chassis_geom=chassis_geom, joints=joints)
        assert distance == pytest.approx(1e-6)
        assert not scene.in_collision(joints=joints, bodies={proxy})
        np.testing.assert_array_equal(session.mj_data.qpos, before)
        scene.robot.set_joints(joints)
        padded = pybullet.getClosestPoints(
            scene.robot.robot_id, proxy, distance=0.0, linkIndexA=6, physicsClientId=scene.cid
        )
        assert padded and padded[0][8] < 0.0
        penetrating = arm.copy()
        penetrating[0] -= 0.05
        penetrating_joints = scene.planning_fingers(arm=penetrating, state=0.5)
        native_overlap = scene.native_chassis_distance(
            link=6, chassis_geom=chassis_geom, joints=penetrating_joints
        )
        assert native_overlap is not None and native_overlap < -0.009
        assert scene.in_collision(joints=penetrating_joints, bodies={proxy})
        np.testing.assert_array_equal(session.mj_data.qpos, before)
    finally:
        scene._sim.close()
        session.close()


def test_exact_contact_pose_uses_threshold_distance_without_changing_native_state() -> None:
    """MuJoCo 3.3.7 can return zero for a separated mesh pair at a large query cap."""
    session = SweepSimpleSession(seed=0)
    model, data = session.mj_model, session.mj_data
    # Full-precision v111 rejection: unlike its rounded replay, this pose exposes
    # the positive-distance solver ambiguity. No physics steps are needed.
    base = (1.3360242983535768, 0.39041569104544893, -4.974209976088986)
    arm = np.array([
        -0.8263251185417175,
        2.2201499938964844,
        2.814652919769287,
        -0.923022449016571,
        -8.281150817871094,
        1.7281091213226318,
        1.508354902267456,
    ])
    fingers = (
        0.6294240741200892,
        0.6292491778980899,
        0.6245892227718545,
        0.6238522908238139,
        -0.6218662570634816,
        -0.6254938868813136,
    )
    joint_values = dict(
        zip(("robot_joint_x", "robot_joint_y", "robot_joint_th"), base, strict=True)
    )
    joint_values.update({f"robot_joint_{i}": value for i, value in enumerate(arm, start=1)})
    joint_values.update({
        f"robot_{name}_joint": value
        for name, value in zip(
            (
                "left_driver",
                "right_driver",
                "left_spring_link",
                "right_spring_link",
                "left_follower",
                "right_follower",
            ),
            fingers,
            strict=True,
        )
    })
    joint_values.update({
        "robot_left_coupler_joint": 0.0005482488952691177,
        "robot_right_coupler_joint": 0.0006636535957275588,
    })
    for name, value in joint_values.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    scene = FloorPlanningScene(session=session)
    try:
        before = data.qpos.copy()
        before_geom = data.geom_xpos.copy()
        before_flags = (int(model.opt.enableflags), int(model.opt.disableflags))
        wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_spherical_wrist_2_link")
        wrist_geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == wrist
            and model.geom_contype[g] + model.geom_conaffinity[g] > 0
        ]
        assert len(wrist_geoms) == 1
        chassis_geom, proxy = scene._native_chassis[0]
        wrist_geom = wrist_geoms[0]
        assert not any({c.geom1, c.geom2} == {wrist_geom, chassis_geom} for c in data.contact)
        # This upstream numerical quirk is version-specific; the threshold and
        # negative-control assertions below remain valid after a MuJoCo upgrade.
        if mujoco.__version__ == "3.3.7":
            assert mujoco.mj_geomDistance(model, data, wrist_geom, chassis_geom, 0.01, None) == 0.0
        assert mujoco.mj_geomDistance(model, data, wrist_geom, chassis_geom, 1e-6, None) == 1e-6
        joints = arm.tolist() + list(fingers)
        assert scene.native_chassis_distance(
            link=6, chassis_geom=chassis_geom, joints=joints
        ) == pytest.approx(1e-6)
        assert not scene.in_collision(joints=joints, bodies={proxy})
        penetrating = list(joints)
        penetrating[0] -= 0.05
        distance = scene.native_chassis_distance(
            link=6, chassis_geom=chassis_geom, joints=penetrating
        )
        assert distance is not None and distance < -0.009
        assert scene.in_collision(joints=penetrating, bodies={proxy})
        np.testing.assert_array_equal(data.qpos, before)
        np.testing.assert_array_equal(data.geom_xpos, before_geom)
        assert (int(model.opt.enableflags), int(model.opt.disableflags)) == before_flags
    finally:
        scene._sim.close()
        session.close()
