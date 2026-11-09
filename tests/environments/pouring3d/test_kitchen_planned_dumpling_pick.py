"""A dumpling by the skillet wall is picked: the pick's free-space motion is planned.

In the p = 0 runs of pouring-planned-20261109-c5758a285 three picks from the skillet
refused "No feasible continuation from prepared arm postures" and stranded the run: the
refused dumplings lie 13 to 21 mm from the skillet's inner wall. The fixture holds the
recorded MuJoCo state at each refusal. Pick and place are assumed to work in this project;
the arm, wrist and palm may still never touch the pan or the hob, only the fingers may.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_dumplings_by_the_skillet_wall.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
PICK = SKILL_NAMES.index("PickShrimpFromTray")
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
        assert c.scene.cube_locations()[recorded["dumpling"]] == "tray"
        return env

    @staticmethod
    def arm_contacts(*, env: KitchenEnvironment) -> list[tuple[str, str, str]]:
        """Watch every tick: robot parts other than the fingers touching the pan or hob."""
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
def test_a_dumpling_by_the_skillet_wall_is_picked_and_lifted_without_the_arm_touching_the_pan(
    *, case
):
    env = Recorded.restored(case=case)
    try:
        c = env.controller()
        index = CASES[case]["dumpling"]
        name = c.scene.cube_names[index]
        start = c.body_pose(name=name)[0]
        assert c.vessel_home(name="tray")
        hits = Recorded.arm_contacts(env=env)
        env.take_action(action=np.array([PICK, index, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"] is True, env.last_info
        assert c.held == name
        assert c.body_pose(name=name)[0][2] - start[2] >= 0.10
        assert not hits, sorted(set(hits))
        # The fingers may touch the pan, but never drag it off home: the human reset
        # and every later pick need it there.
        assert c.vessel_home(name="tray")
    finally:
        env.close()


def test_a_pick_from_the_pan_with_no_approach_refuses_to_plan_before_any_motion(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
    from hitl_pmp.environments.pouring3d.kitchen.pick_plannable import KitchenPickPlannable

    case = sorted(CASES)[0]
    env = Recorded.restored(case=case)
    try:
        c = env.controller()
        monkeypatch.setattr(
            KitchenController, "pan_pick_chain", lambda self, **_: "no grasp IK (test)"
        )
        steps, qpos = c.steps, c.data.qpos.copy()
        with pytest.raises(ValueError, match="^No planned pick from the pan: ") as refused:
            c.pick_from_pan(index=CASES[case]["dumpling"])
        assert KitchenPickPlannable.is_planning_refusal(error=str(refused.value))
        # Every grasp orientation was tried, and none moved the robot.
        assert str(refused.value).count("no grasp IK (test)") == 18
        assert c.steps == steps
        np.testing.assert_array_equal(c.data.qpos, qpos)
    finally:
        env.close()
