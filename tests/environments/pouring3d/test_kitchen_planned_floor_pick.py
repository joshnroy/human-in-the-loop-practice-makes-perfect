"""A dumpling on the floor is picked with planned free-space motion, as one in the pan is.

In pouring-threefix-20261109-3586431e0 the scripted floor pick drove the base in a straight
line to a pose in front of the dumpling: past the stove, that line ran into the stovetop
and left the base pressed against it ("Base tracking failed in position for shrimp pick"),
a dead end for every later skill. Elsewhere it refused "IK infeasible in position for
shrimp pick". The fixture holds the recorded MuJoCo state before each of those picks.
Pick and place are assumed to work in this project; the arm, wrist, palm and base may
still never touch the stove, the prep table or the pan, only the fingers may.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_dumpling_on_the_floor.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
PICK = SKILL_NAMES.index("PickShrimpFromFloor")
SCENE = ("tray", "stovetop", "prep_table")


class Recorded:
    """A static-method container, never instantiated."""

    @staticmethod
    def restored(*, case: str) -> KitchenEnvironment:
        recorded = CASES[case]
        env = KitchenEnvironment(requested_count=recorded["requested"])
        env.reset_to_seed(seed=recorded["scene_seed"], requested=recorded["requested"])
        c = env.controller()
        c.data.qpos[:] = recorded["qpos"]
        c.data.qvel[:] = recorded["qvel"]
        c.data.time = recorded["time"]
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        env.current_state = env.observe()
        assert c.scene.cube_locations()[recorded["dumpling"]] == "floor"
        return env

    @staticmethod
    def robot_contacts(*, env: KitchenEnvironment) -> list[tuple[str, str, str]]:
        """Watch every tick: robot parts other than the fingers touching the scene."""
        c = env.controller()
        hits: list[tuple[str, str, str]] = []
        observed = c.after_step

        def watch() -> None:
            if observed is not None:
                observed()
            m, d = c.model, c.data
            for contact in d.contact[: d.ncon]:
                bodies = [m.body(int(m.geom_bodyid[g])).name or "" for g in contact.geom]
                robot = [b for b in bodies if b.startswith("robot")]
                scene = [b for b in bodies if b in SCENE]
                if robot and scene and not robot[0].startswith(("robot_left_", "robot_right_")):
                    hits.append((c.phase, robot[0], scene[0]))

        c.after_step = watch
        return hits


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_dumpling_on_the_floor_is_picked_and_lifted_without_the_robot_touching_the_kitchen(
    *, case
):
    env = Recorded.restored(case=case)
    try:
        c = env.controller()
        index = CASES[case]["dumpling"]
        name = c.scene.cube_names[index]
        start = c.body_pose(name=name)[0]
        hits = Recorded.robot_contacts(env=env)
        env.take_action(action=np.array([PICK, index, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"] is True, env.last_info
        assert c.held == name
        assert c.body_pose(name=name)[0][2] - start[2] >= 0.10
        assert not hits, sorted(set(hits))
        assert any(e.get("event") == "planned pick from the floor" for e in c.events)
    finally:
        env.close()


def test_a_pick_from_the_floor_with_no_approach_refuses_to_plan_before_any_motion(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
    from hitl_pmp.environments.pouring3d.kitchen.pick_plannable import KitchenPickPlannable

    case = "ees_p0.1_s2"
    env = Recorded.restored(case=case)
    try:
        c = env.controller()
        monkeypatch.setattr(
            KitchenController, "pan_pick_chain", lambda self, **_: "no grasp IK (test)"
        )
        steps, qpos = c.steps, c.data.qpos.copy()
        with pytest.raises(ValueError, match="^No planned pick from the floor: ") as refused:
            c.pick_payload_with_retry(index=CASES[case]["dumpling"], location="floor")
        assert KitchenPickPlannable.is_planning_refusal(error=str(refused.value))
        assert "no grasp IK (test)" in str(refused.value)
        assert c.steps == steps
        np.testing.assert_array_equal(c.data.qpos, qpos)
    finally:
        env.close()
