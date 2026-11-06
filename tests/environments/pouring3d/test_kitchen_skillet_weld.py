"""The kitchen skillet is welded to the stove (Josh, 2026-11-06), so `TrayHome` is gone.

It was a free 2.57 kg body. The weld holds it where the episode placed it: a push, a
forceful Pour and a whole pick-pour-place leave it home, and the pick-pour-place reaches
the "pour 1" goal {Split1, HandEmpty, PitcherHome} on the measured state.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenScene
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.kitchen.tasks import KitchenTasks
from hitl_pmp.environments.pouring3d.kitchen.types import ClosedLoopPourParameters

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
    pytest.mark.xdist_group("kitchen_skillet_weld"),
]


def _pose(*, env: KitchenEnvironment) -> tuple[np.ndarray, np.ndarray]:
    c = env.controller()
    pan = c.model.body("tray").id
    return c.data.xpos[pan].copy(), c.data.xquat[pan].copy()


def _moved(*, before: tuple[np.ndarray, np.ndarray], env: KitchenEnvironment) -> tuple:
    position, quaternion = _pose(env=env)
    angle = 2 * np.arccos(min(1.0, abs(float(np.dot(before[1], quaternion)))))
    return float(np.linalg.norm(position - before[0])), float(np.rad2deg(angle))


def _run(*, env: KitchenEnvironment, name: str, params: np.ndarray | None = None) -> None:
    provider = KitchenSkillProvider(env=env)
    ground = provider.ground(name=name)
    action = provider.compute_action(
        ground_skill=ground,
        params=np.zeros(0) if params is None else params,
        state=env.get_current_state(),
    )
    env.take_action(action=action)


def test_the_skillet_weld_is_active_from_the_first_state():
    import mujoco

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        weld = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_EQUALITY, KitchenScene.SKILLET_WELD)
        assert weld >= 0
        assert c.data.eq_active[weld] == 1
        assert c.vessel_home(name="tray")
    finally:
        env.close()


def test_a_hard_push_does_not_move_the_skillet():
    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        before = _pose(env=env)
        pan = c.model.body("tray").id
        c.data.xfrc_applied[pan] = [100.0, 100.0, 0.0, 0.0, 0.0, 20.0]
        c.pause(phase="push the skillet", seconds=1.0)
        c.data.xfrc_applied[pan] = 0.0
        distance, angle = _moved(before=before, env=env)
        # Measured 2026-11-06 with the weld: well under these; a free pan slides away.
        assert distance < 0.002, distance
        assert angle < 0.5, angle
        c.pause(phase="release", seconds=0.5)
        distance, angle = _moved(before=before, env=env)
        assert distance < 0.0005, distance
        assert c.vessel_home(name="tray")
    finally:
        env.close()


def test_pick_pour_place_reaches_the_pour_one_goal_and_the_skillet_stays_home():
    env = KitchenEnvironment(requested_count=1)
    try:
        before = _pose(env=env)
        goal = {
            (a.predicate.name, tuple(o.name for o in a.objects))
            for a in KitchenTasks(env=env).goal().atoms
        }
        assert goal == {("Split1", ()), ("HandEmpty", ("robot",)), ("PitcherHome", ("pitcher",))}
        _run(env=env, name="PickPitcher")
        assert env.last_info["success"], env.last_info
        _run(env=env, name="Pour1", params=np.array([1.0]))
        _run(env=env, name="PlacePitcherAtHome")
        assert env.last_info["success"], env.last_info
        c = env.controller()
        for _ in range(12):
            if c.outcome_settled():
                break
            c.pause(phase="observe rest", seconds=1.0)
        atoms = env.observe().abstract_atoms
        assert goal <= atoms, sorted(atoms)
        distance, angle = _moved(before=before, env=env)
        assert distance < 0.001, distance
        assert angle < 0.2, angle
    finally:
        env.close()


def test_a_forceful_pour_does_not_move_the_skillet():
    env = KitchenEnvironment(requested_count=3)
    try:
        before = _pose(env=env)
        _run(env=env, name="PickPitcher")
        assert env.last_info["success"], env.last_info
        fastest = ClosedLoopPourParameters.LIMITS[1]
        _run(env=env, name="Pour3", params=np.array([fastest]))
        distance, angle = _moved(before=before, env=env)
        assert distance < 0.001, distance
        assert angle < 0.2, angle
        assert env.controller().vessel_home(name="tray")
    finally:
        env.close()
