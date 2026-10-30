"""Native geometry, reset and IK invariants for the potsticker diagnostic."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController
from hitl_pmp.environments.pouring3d.potsticker_native import make_potsticker_env


def test_repeated_prediction_observation():
    """The inner tracking loop and outer cutoff can observe the same time."""
    fake = SimpleNamespace(
        joint_pour_active=False,
        body_pose=lambda **_: (np.zeros(3), Rotation.identity()),
        scene=SimpleNamespace(
            get_object=lambda _: SimpleNamespace(height=1.0),
            cube_names=["food"],
            cube_locations=lambda: ["pitcher"],
        ),
        model=SimpleNamespace(body=lambda _: SimpleNamespace(id=0)),
        data=SimpleNamespace(time=1.0, xipos=np.array([[0.0, 0.0, 0.99]])),
        anticipation_seconds=0.6,
        mouth_history={"food": (0.9, 0.98, 0.1)},
        outbound=set(),
        feedback_trace=[],
        phase="feeding",
    )
    assert PotstickerController.observe_mouth(fake) == 1
    assert PotstickerController.observe_mouth(fake) == 1
    assert not fake.outbound  # A prediction is never labeled an actual crossing.
    fake.data.time = 1.1
    fake.data.xipos[0, 2] = 0.98
    assert PotstickerController.observe_mouth(fake) == 0
    fake.data.time = 1.2
    fake.data.xipos[0, 2] = 1.01
    assert PotstickerController.observe_mouth(fake) == 1
    assert fake.outbound == {"food"}


def test_joint_pour_does_not_observe_food():
    # No scene, simulator positions, or mouth history are available to this mode.
    fake = SimpleNamespace(joint_pour_active=True)
    assert PotstickerController.observe_mouth(fake) == 0


def test_joint_pour_uses_fixed_payload_model():
    fake = SimpleNamespace(
        joint_pour_active=True,
        joint_payload_mass=0.5,
        joint_payload_com=np.array([0.1, 0, 0]),
        hand_pose=lambda: (np.zeros(3), Rotation.identity()),
        model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0, 0, -10.0]))),
    )
    force, moment = PotstickerController.payload_wrench(fake)
    np.testing.assert_allclose(force, [0, 0, 5])
    np.testing.assert_allclose(moment, [0, -0.5, 0])


def test_joint_pour_tracks_without_live_pitcher_pose():
    targets = []
    motors = []

    def ik(**target):
        targets.append(target)
        return np.zeros(7)

    fake = SimpleNamespace(
        joint_pour_active=True,
        pour_height_drop=0.16,
        wiggle_direction=lambda: np.array([0, -1, 0]),
        rotation_offset=Rotation.from_euler("z", 0.2),
        pour_orientation=lambda **_: Rotation.from_euler("y", np.pi / 2),
        inverse_kinematics=ik,
        motor_step=lambda **command: motors.append(command),
        phase="feedback tilt pitcher",
        data=SimpleNamespace(qpos=np.zeros(7)),
        arm_q=np.arange(7),
    )
    PotstickerController.tracked_step(
        fake, center=np.array([0.0, 0.0, 1.0]), angle=90, offset=0.008
    )
    np.testing.assert_allclose(targets[0]["position"], [0, -0.008, 0.84])
    assert len(motors) == 1


@pytest.mark.parametrize("count", [3, 10])
def test_payload_and_reset(*, count):
    env = make_potsticker_env(count=count, requested=count, seed=2, scene_bg=False)
    try:
        c = PotstickerController.create(env=env)
        assert c.ready()
        assert c.scene.requested_count == count
        assert len(c.scene.cube_names) == count
        bodies = [c.model.body(n).id for n in c.scene.cube_names]
        assert np.allclose(c.model.body_mass[bodies], 0.020)
        before = c.data.qpos.copy()
        time = c.data.time
        payload_q = []
        for body in bodies:
            start = c.model.jnt_qposadr[c.model.body_jntadr[body]]
            payload_q.extend(range(start, start + 7))
        c.scene.respawn_payload_in_pitcher()
        fixed = np.ones(c.model.nq, dtype=bool)
        fixed[payload_q] = False
        np.testing.assert_array_equal(before[fixed], c.data.qpos[fixed])
        assert c.data.time == time
        # Human reset may place a top-layer piece partly above the lip; gravity
        # must settle it into the cavity, without altering the fixed-state audit.
        for _ in range(30):
            env.step(np.zeros(env.action_space.shape))
        assert c.scene.cube_locations() == ["pitcher"] * count
        # Fast IK scratch computes the same transforms and obstacle costs as full
        # forward dynamics, including at a perturbed arm configuration.
        for delta in (0.0, 0.05):
            c.scratch.qpos[:] = c.data.qpos
            c.scratch.qpos[c.arm_q] += delta
            mujoco.mj_forward(c.model, c.scratch)
            geom = c.scratch.geom_xpos.copy()
            site = c.scratch.site_xpos.copy()
            residual = c.obstacle_residual().copy()
            c.update_ik_scratch()
            np.testing.assert_array_equal(geom, c.scratch.geom_xpos)
            np.testing.assert_array_equal(site, c.scratch.site_xpos)
            np.testing.assert_array_equal(residual, c.obstacle_residual())
        # Harness-only fixtures exercise every exact-count goal (including 6..10)
        # and verify that the nonrequested remainder must still be retained.
        origin = c.body_pose(name="tray")[0]
        for i, name in enumerate(c.scene.cube_names):
            obj = c.scene.get_object(name)
            offset = np.array([(i % 3 - 1) * 0.065, (i // 3 - 1.5) * 0.035, 0.0045])
            obj.set_pose(origin + offset, [1, 0, 0, 0])
            obj.set_velocity([0.0] * 3, [0.0] * 3)
            mujoco.mj_forward(c.model, c.data)
            c.scene.requested_count = i + 1
            assert c.scene._check_goals()
            c.scene.requested_count = count if i + 1 != count else 1
            if c.scene.requested_count != i + 1:
                assert not c.scene._check_goals()
    finally:
        env.close()
