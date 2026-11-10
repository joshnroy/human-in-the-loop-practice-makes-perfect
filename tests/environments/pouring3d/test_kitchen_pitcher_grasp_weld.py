"""A verified jug grasp survives carrying and releases at every hand-off boundary."""

import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
    pytest.mark.xdist_group("kitchen_pitcher_grasp_weld"),
]


def _environment():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    env.reset_to_seed(seed=0, requested=1)
    return env


def _anchored(*, controller):
    weld = controller.model.equality("pitcher_home_weld").id
    return bool(controller.data.eq_active[weld])


def _pick(*, env):
    env.take_action(action=np.array([0, 0, 1, 0, 0, 0], dtype=float))
    assert env.last_info["success"], env.last_info
    assert env.controller().scene.pitcher_attached()
    assert not _anchored(controller=env.controller())


def test_the_verified_grasp_survives_a_physical_pour_and_releases_for_placement(
    *, monkeypatch, record_property
):
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

    env = _environment()
    drift = []
    motor_step = KitchenController.motor_step

    def watched_step(self, *, target_q, grip):  # noqa: PLR0917 — replaces an instance method
        motor_step(self, target_q=target_q, grip=grip)
        if not self.scene.pitcher_attached():
            return
        assert not _anchored(controller=self)
        hand = self.model.body(f"{self.scene.robot_name}_bracelet_link").id
        jug = self.model.body("pitcher").id
        hand_rotation = Rotation.from_matrix(self.data.xmat[hand].reshape(3, 3))
        jug_rotation = Rotation.from_matrix(self.data.xmat[jug].reshape(3, 3))
        position = hand_rotation.inv().apply(self.data.xpos[jug] - self.data.xpos[hand])
        relative = hand_rotation.inv() * jug_rotation
        target = self.model.eq_data[self.scene.grasp_weld(name="pitcher")]
        target_rotation = Rotation.from_quat(target[6:10][[1, 2, 3, 0]])
        drift.append((
            np.linalg.norm(position - target[3:6]),
            (relative.inv() * target_rotation).magnitude(),
        ))

    monkeypatch.setattr(KitchenController, "motor_step", watched_step)
    try:
        _pick(env=env)
        c = env.controller()
        env.take_action(action=np.array([1, 0, 1, 1, 0, 0], dtype=float))
        # Scoring is deliberately not fixed: the held jug must survive even if a
        # physical pour misses its requested count.
        assert env.last_info["error"] is None, env.last_info
        assert c.held == "pitcher"
        assert c.scene.pitcher_attached()
        env.take_action(action=np.array([2, 0, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"], env.last_info
        assert not c.scene.pitcher_attached()
        assert _anchored(controller=c)
        assert c.held is None
        assert c.vessel_home(name="pitcher")
        assert len(drift) > 100
        peak = np.max(drift, axis=0)
        record_property("max_grasp_position_error_m", peak[0])
        record_property("max_grasp_rotation_error_deg", np.rad2deg(peak[1]))
        assert peak[0] < 0.001
        # MuJoCo welds remain compliant under load; bound slip throughout motion,
        # rather than demanding the zero residual of a kinematic attachment.
        assert np.rad2deg(peak[1]) < 2.0
    finally:
        env.close()


def test_a_failed_human_reset_keeps_the_grasp_and_a_successful_one_releases_it():
    env = _environment()
    try:
        _pick(env=env)
        c = env.controller()
        env.human_reset_success_probability = 0.0
        assert env.reset_movables()
        assert env.last_info["human_reset_failed"]
        assert c.held == "pitcher"
        assert c.scene.pitcher_attached()
        assert not _anchored(controller=c)
        env.human_reset_success_probability = 1.0
        assert env.reset_movables()
        assert c.held is None
        assert not c.scene.pitcher_attached()
        assert _anchored(controller=c)
        assert c.vessel_home(name="pitcher")
        assert env.human_interventions == 2
    finally:
        env.close()


def test_an_episode_reset_clears_the_grasp_and_anchoring_rejects_a_held_weld():
    env = _environment()
    try:
        c = env.controller()
        c.scene.attach_pitcher()
        with pytest.raises(ValueError, match="welded to the gripper"):
            c.scene.weld_pitcher_home()
        assert not _anchored(controller=c)
        c.scene.reset(seed=0)
        assert not c.scene.pitcher_attached()
        assert _anchored(controller=c)
    finally:
        env.close()


@pytest.mark.parametrize("lift_height", [0.0, 0.01])
def test_a_failed_lift_releases_the_grasp_before_held_is_set(*, monkeypatch, lift_height):
    import mujoco

    from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

    env = _environment()

    def failed_lift(self, *, handle, down, origin):  # noqa: PLR0917 — replaces an instance method
        self.pitcher_grasp_acquired()
        assert self.scene.pitcher_attached()
        assert self.held is None
        self.data.qpos[self.free_qpos(name="pitcher")][2] += lift_height
        mujoco.mj_forward(self.model, self.data)
        raise ValueError("injected lift failure")

    monkeypatch.setattr(PotstickerController, "grasp_and_lift_pitcher", failed_lift)
    try:
        c = env.controller()
        expected = c.data.qpos[c.free_qpos(name="pitcher")].copy()
        expected[2] += lift_height
        with pytest.raises(ValueError, match="injected lift failure"):
            c.grasp_and_lift_pitcher(
                handle=np.zeros(3), down=Rotation.identity(), origin=np.zeros(3)
            )
        assert c.held is None
        assert not c.scene.pitcher_attached()
        assert _anchored(controller=c) == (lift_height == 0)
        np.testing.assert_array_equal(c.data.qpos[c.free_qpos(name="pitcher")], expected)
    finally:
        env.close()
