"""A pick from the pan takes the named dumpling and no other.

In a table-weld Curtis run (p = 0, seed 1) `PickShrimpFromTray(potsticker_1)` closed its
right pad on potsticker_1 and its left pad on potsticker_0, which leant on potsticker_1
against the pan wall. Both were lifted, potsticker_0 by friction alone, and the pick
reported success. A dumpling that is off every surface and not the held one has no
location atom, so no robot skill mentioned potsticker_0 and the planner asked the human
43 times. The fixture holds the recorded MuJoCo state at the tick before that pick.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

pytestmark = pytest.mark.xdist_group("kitchen_pan_pick_single_dumpling")

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_pan_pick_pinches_a_second_dumpling.json"
CASE = json.loads(FIXTURE.read_text())["cases"]["curtis_p0_seed1"]
PICK = SKILL_NAMES.index("PickShrimpFromTray")
LOCATION_ATOMS = ("InPitcher", "InTray", "OnFloor", "OnSurface")


class Recorded:
    """A static-method container, never instantiated."""

    @staticmethod
    def restored(*, neighbour_on_floor: bool = False) -> KitchenEnvironment:
        env = KitchenEnvironment(requested_count=CASE["requested"])
        env.reset_to_seed(seed=CASE["scene_seed"], requested=CASE["requested"])
        c = env.controller()
        c.data.qpos[:] = CASE["qpos"]
        c.data.qvel[:] = CASE["qvel"]
        c.data.time = CASE["time"]
        if neighbour_on_floor:
            pose = c.free_qpos(name=c.scene.cube_names[CASE["neighbour"]])
            c.data.qpos[pose] = [-1.5, 0.5, 0.03, 1.0, 0.0, 0.0, 0.0]
            c.data.qvel[c.free_dofs(name=c.scene.cube_names[CASE["neighbour"]])] = 0.0
        mujoco.mj_forward(c.model, c.data)
        c.scene.payload_rest.clear()
        c.scene.weld_skillet()
        c.scene.weld_pitcher_home()
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        env.current_state = env.observe()
        return env

    @staticmethod
    def location_atoms(*, env: KitchenEnvironment) -> dict[str, list[str]]:
        atoms = env.observe().abstract_atoms
        return {
            name: sorted(a for a, objects in atoms if a in LOCATION_ATOMS and objects[0] == name)
            for name in env.controller().scene.cube_names
        }

    @staticmethod
    def pick(*, env: KitchenEnvironment) -> None:
        env.take_action(action=np.array([PICK, CASE["dumpling"], 1, 0, 0, 0], dtype=float))


@pytest.fixture(scope="module")
def after_the_recorded_pick():
    """The recorded pick dispatched once from the recorded state, and how high the
    neighbour rose above where it lay at any tick of it."""
    env = Recorded.restored()
    try:
        c = env.controller()
        neighbour = c.scene.cube_names[CASE["neighbour"]]
        names = list(c.scene.cube_names)
        assert c.scene.cube_locations() == ["tray", "tray", "pitcher"]
        start = c.body_pose(name=neighbour)[0][2]
        heights: list[float] = []
        observed = c.after_step

        def watch() -> None:
            if observed is not None:
                observed()
            heights.append(float(c.body_pose(name=neighbour)[0][2]) - start)

        c.after_step = watch
        Recorded.pick(env=env)
        c.after_step = observed
        yield env, names, max(heights)
    finally:
        env.close()


def test_the_recorded_pick_reproduces_the_double_pinch(*, after_the_recorded_pick):
    """Guards the fixture: without the neighbour leaving the pan floor with the hand,
    the tests below would pass on a pick that never met the defect."""
    _, _, rise = after_the_recorded_pick
    assert rise > 0.05, rise


def test_a_pan_pick_that_lifts_a_second_dumpling_reports_failure(*, after_the_recorded_pick):
    env, names, _ = after_the_recorded_pick
    assert env.last_info["success"] is False, env.last_info
    assert env.last_info["error"] == (f"Pick lifted a second dumpling: {names[CASE['neighbour']]}")


def test_after_it_the_hand_is_empty_and_no_dumpling_is_pinched(*, after_the_recorded_pick):
    env, names, _ = after_the_recorded_pick
    c = env.controller()
    assert c.held is None
    assert c.scene.attached_payload() is None
    assert "air" not in c.scene.cube_locations(), c.scene.cube_locations()
    assert {name: c.finger_contacts(name=name) for name in names} == dict.fromkeys(names, set())


def test_after_it_every_dumpling_has_exactly_one_location_atom(*, after_the_recorded_pick):
    env, names, _ = after_the_recorded_pick
    where = Recorded.location_atoms(env=env)
    assert {name: len(atoms) for name, atoms in where.items()} == dict.fromkeys(names, 1), where
    # Lowered and released where they were taken: both are back in the pan.
    assert where[names[CASE["dumpling"]]] == ["InTray"], where
    assert where[names[CASE["neighbour"]]] == ["InTray"], where


def test_the_same_dumpling_without_its_neighbour_is_still_picked():
    env = Recorded.restored(neighbour_on_floor=True)
    try:
        c = env.controller()
        name = c.scene.cube_names[CASE["dumpling"]]
        c.pause(phase="wait", seconds=2)
        assert c.scene.cube_locations() == ["floor", "tray", "pitcher"]
        start = c.body_pose(name=name)[0]
        Recorded.pick(env=env)
        assert env.last_info["success"] is True, env.last_info
        assert c.held == name
        assert c.body_pose(name=name)[0][2] - start[2] >= 0.10
        assert Recorded.location_atoms(env=env) == {
            c.scene.cube_names[0]: ["OnFloor"],
            name: [],
            c.scene.cube_names[2]: ["InPitcher"],
        }
    finally:
        env.close()
