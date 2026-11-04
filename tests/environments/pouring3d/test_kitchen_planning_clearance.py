"""Every configuration a kitchen skill commands keeps the planning margin, so any state a
skill ends in, a mid-skill failure included, is a valid start for the motion planner.

Recorded 2026-11-04 (cycle 2 of a four-cycle kitchen run): a dumpling slipped out 1.6 s
into DropShrimpIntoPitcher's scripted "raise held shrimp in aisle". That segment was
checked only in MuJoCo, and it had left the upper arm 14.4 mm from the stovetop in the
planning scene, inside the 15 mm margin. BiRRT refuses such a start, so the next
PickPitcher's base drive was refused and no motion could be planned from there again.
"""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.planning.clearance import PlanningClearance
from hitl_pmp.environments.pouring3d.kitchen.planning.utils import PLANNING_MARGIN_M
from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

pytestmark = pytest.mark.xdist_group("kitchen_planning_clearance")

DUMPLING = 1
# MuJoCo qpos as recorded at the start of that drop (the dumpling held by its ends) and
# where the failed drop left the robot.
DROP_START = [
    0.69973, -1.05087, 0.74, 1.0, -0.0, -0.0, 0.00026, 0.8542, 0.84219, 0.74, 0.70722, 0.0,
    0.0, -0.70699, 0.71277, -1.05771, 0.75507, 0.29933, -0.41405, -0.68861, 0.51457,
    0.90449, 0.8015, 0.89963, -0.75811, -0.63254, -0.15654, 0.0257, 0.69454, -1.0356,
    0.75507, 0.58281, -0.78889, -0.15259, 0.12129, 0.3, 0.78714, -0.0, 2.20416, -1.18556,
    2.34929, -1.01294, -3.81872, 1.71736, 4.04815, 0.35676, 3e-05, 0.35555, -0.34334,
    0.35678, 0.00026, 0.35587, -0.34592,
]  # fmt: skip
STRANDED = [
    0.69973, -1.05087, 0.74, 1.0, -0.0, -0.0, 0.00026, 0.8542, 0.84219, 0.74, 0.70722, 0.0,
    0.0, -0.70699, 0.713, -1.05817, 0.75508, 0.30999, -0.42849, -0.6801, 0.50771, 0.88255,
    0.76628, 0.76517, -0.41447, -0.90868, 0.04804, -0.01418, 0.69453, -1.0356, 0.75507,
    0.5829, -0.7889, -0.15241, 0.12099, 0.3, 0.78714, 0.0, 2.2012, -1.18697, 2.40008,
    -1.02609, -3.83706, 1.7362, 4.08879, 0.39756, 0.0001, 0.39699, -0.38126, 0.39756,
    0.0001, 0.39699, -0.38126,
]  # fmt: skip


class Recorded:
    @staticmethod
    def restore(*, env: KitchenEnvironment, qpos: list[float]) -> None:
        c = env.controller()
        assert c.model.nq == len(qpos)
        c.data.qpos[:] = qpos
        c.data.qvel[:] = 0
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()

    @staticmethod
    def start_clearance(*, env: KitchenEnvironment) -> float:
        """The planner's start check for the robot as it stands: nothing exempt but what
        the hand holds."""
        c = env.controller()
        guard = PlanningClearance.for_controller(controller=c)
        try:
            return guard.closest(
                controller=c, base=c.data.qpos[c.base_q], arm=c.data.qpos[c.arm_q], contact=False
            ).distance
        finally:
            guard.close()


class Ticks:
    """Every control tick: the commanded and the executed configuration's clearance in
    the planning scene (contact exemptions as for the skill's own checks)."""

    @staticmethod
    def record(*, env: KitchenEnvironment) -> tuple[dict[str, list], PlanningClearance]:
        c = env.controller()
        guard = PlanningClearance.for_controller(controller=c)
        seen: dict[str, list] = {"skill": [], "phase": [], "commanded": [], "executed": []}
        current = {"skill": ""}
        sink = c.callback

        def tick(**kwargs):  # noqa: PLR0917 -- controller callback signature
            if sink is not None:
                sink(**kwargs)
            commanded = guard.closest(controller=c, base=c.base_target, arm=c.last_q)
            executed = guard.closest(
                controller=c, base=c.data.qpos[c.base_q], arm=c.data.qpos[c.arm_q]
            )
            seen["skill"].append(current["skill"])
            seen["phase"].append(c.phase)
            seen["commanded"].append(commanded)
            seen["executed"].append(executed)

        c.callback = tick
        seen["current"] = [current]
        return seen, guard


def test_the_recorded_stranding_is_inside_the_margin_and_a_fresh_episode_is_not():
    env = KitchenEnvironment(requested_count=1)
    assert Recorded.start_clearance(env=env) >= PLANNING_MARGIN_M
    Recorded.restore(env=env, qpos=STRANDED)
    assert Recorded.start_clearance(env=env) < PLANNING_MARGIN_M
    env.close()


def test_an_ik_solution_inside_the_margin_is_re_solved_wider_then_refused(*, monkeypatch):
    """The stranded arm is what MuJoCo's own IK check accepted. Handed that solution for
    the raise, the controller re-solves with wider MuJoCo clearances and, when every
    solution still leaves the margin, refuses before any motor command."""
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    Recorded.restore(env=env, qpos=DROP_START)
    c.held, c.phase = c.scene.cube_names[DUMPLING], "raise held shrimp in aisle"
    stranded = np.asarray(STRANDED)[c.arm_q]
    previous, clearances = c.last_q.copy(), []

    def solve(self, *, position, rotation):  # noqa: PLR0917 -- patched method
        clearances.append(self.obstacle_clearance)
        self.last_q = stranded.copy()
        return stranded.copy()

    monkeypatch.setattr(PotstickerController, "inverse_kinematics", solve)
    hand, rotation = c.hand_pose()
    steps = c.steps
    with pytest.raises(
        ValueError,
        match="^Insufficient obstacle clearance in raise held shrimp in aisle: planning scene "
        "half_arm_2_link/collider:stovetop:14 14.4 mm",
    ):
        c.inverse_kinematics(position=hand, rotation=rotation)
    assert clearances == [0.015, 0.025, 0.035]
    assert c.obstacle_clearance == 0.015
    np.testing.assert_array_equal(c.last_q, previous)
    assert c.steps == steps
    env.close()


def test_every_commanded_configuration_of_a_canonical_cycle_keeps_the_margin():
    """PickPitcher, Pour1, PlacePitcherAtHome, PickShrimpFromTray, DropShrimpIntoPitcher
    from a fresh episode: every commanded configuration keeps the margin, and every skill
    ends where the planner could start."""
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

    env = KitchenEnvironment(requested_count=1)
    provider = KitchenSkillProvider(env=env)
    c = env.controller()
    seen, guard = Ticks.record(env=env)
    ends, outcomes = {}, {}
    for name, params in (
        ("PickPitcher", ()),
        ("Pour1", (1.0,)),
        ("PlacePitcherAtHome", ()),
        ("PickShrimpFromTray", ()),
        ("DropShrimpIntoPitcher", ()),
    ):
        index = next((i for i, at in enumerate(c.scene.cube_locations()) if at != "pitcher"), 0)
        seen["current"][0]["skill"] = name
        ground = provider.ground(name=name, request=1, shrimp=index)
        env.take_action(action=provider.encode(ground_skill=ground, params=np.array(params)))
        outcomes[name] = (env.last_info["success"], env.last_info.get("error"))
        ends[name] = guard.closest(
            controller=c, base=c.data.qpos[c.base_q], arm=c.data.qpos[c.arm_q], contact=False
        )
        if not outcomes[name][0] and name != "DropShrimpIntoPitcher":
            break
    worst: dict[str, tuple[float, str, str, float]] = {}
    for skill, phase, commanded, executed in zip(
        seen["skill"], seen["phase"], seen["commanded"], seen["executed"], strict=True
    ):
        best = worst.get(skill, (1.0, "", "", 1.0))
        worst[skill] = (
            min(best[0], commanded.distance),
            phase if commanded.distance < best[0] else best[1],
            commanded.describe() if commanded.distance < best[0] else best[2],
            min(best[3], executed.distance),
        )
    for skill, (commanded, phase, where, executed) in worst.items():
        print(
            f"{skill}: commanded {commanded * 1000:.1f} mm ({phase}: {where}); "
            f"executed {executed * 1000:.1f} mm; ends {ends[skill].distance * 1000:.1f} mm; "
            f"outcome {outcomes[skill]}"
        )
    guard.close()
    # Every skill ran; the drop is judged by where its dumpling landed, not by its
    # 20 s settle check: in this sequence a dumpling already in the jug rocks on for
    # longer than that with the arm 77 mm away (measured 2026-11-04), which says
    # nothing about clearance.
    assert len(outcomes) == 5, outcomes
    assert all(outcomes[n][0] for n in list(outcomes)[:4]), outcomes
    assert outcomes["DropShrimpIntoPitcher"][0] or outcomes["DropShrimpIntoPitcher"][1] == (
        "Returned potstickers did not settle within 20 seconds"
    ), outcomes
    assert c.scene.cube_locations() == ["pitcher"] * 3
    assert all(w[0] >= PLANNING_MARGIN_M for w in worst.values()), worst
    assert all(e.distance >= PLANNING_MARGIN_M for e in ends.values()), ends
    env.close()


def test_grasping_the_jug_never_counts_the_dumplings_riding_in_it():
    """Four-cycle run, 2026-11-04: PickPitcher's lift was refused at 13.3 mm from a
    dumpling inside the jug it was lifting. Riders move with the jug."""
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    guard = PlanningClearance.for_controller(controller=c)
    assert c.scene.cube_locations() == ["pitcher"] * 3
    for phase in ("descend to pitcher handle", "close fingers on pitcher handle"):
        c.phase, c.held = phase, None
        _, touchable = guard.exempt(controller=c)
        assert set(c.scene.cube_names) <= touchable, phase
    c.phase, c.held = "lift pitcher by handle", "pitcher"
    ignored, _ = guard.exempt(controller=c)
    assert set(c.scene.cube_names) <= ignored
    guard.close()
    env.close()
