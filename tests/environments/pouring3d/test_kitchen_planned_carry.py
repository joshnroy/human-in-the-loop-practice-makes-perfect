"""Pour's carry from the lift to the pour start is planned upright and executed clear."""

import numpy as np
import pytest
from kitchen_clearance import Clearance

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_planned_carry")

PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
POUR1 = np.array([1, 0, 1, 91, 0.01, 10], dtype=float)
CARRY_PHASES = {
    "planned tuck jug for carry",
    "planned carry jug to pour pose",
    "planned arm to pour start",
}
# The planner keeps the rigidly modelled jug within 10 degrees of vertical. In MuJoCo the
# handle grasp is compliant and the jug already hangs about 5 degrees after the lift,
# so the executed bound is looser: measured 11.5 degrees on 2026-11-02 (3.6 degrees of
# hang after the lift plus compliance in motion), held here to 15.
EXECUTED_TILT_LIMIT_DEG = 15.0


def test_the_carry_is_refused_whole_or_executed_upright_and_clear(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        KitchenMotionPlanner,
    )

    env = KitchenEnvironment(requested_count=1)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    c = env.controller()

    # No upright path exists for a zero tilt budget (the jug hangs a few degrees in the
    # hand): the whole carry is refused before anything moves.
    monkeypatch.setattr(KitchenMotionPlanner, "tilt_limit_deg", 0.0)
    steps = c.steps
    env.take_action(action=POUR1)
    assert env.last_info["error"].startswith("No collision-free upright ")
    assert c.steps == steps
    assert c.held == "pitcher"
    monkeypatch.undo()

    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

    waits: list[tuple[str, float]] = []
    pause = KitchenController.pause

    def recorded(self, *, phase: str, seconds: float, grip: float | None = None) -> None:  # noqa: PLR0917
        waits.append((phase, seconds))
        pause(self, phase=phase, seconds=seconds, grip=grip)

    monkeypatch.setattr(KitchenController, "pause", recorded)
    seen = Clearance.record(env=env)
    env.take_action(action=POUR1)
    assert env.last_info["error"] is None, env.last_info
    # The outcome is judged 30 s after the jug is back upright (approved 2026-11-02).
    settle = {"settle potstickers in pan", "wait for potstickers to settle"}
    assert sum(seconds for phase, seconds in waits if phase in settle) == 30
    assert {row[0] for row in seen} >= CARRY_PHASES
    Clearance.check(seen=seen)
    carry = [row for row in seen if row[0] in CARRY_PHASES]
    tilt = max(row[2] for row in carry)
    print(f"largest MuJoCo jug tilt over the carry: {tilt:.1f} deg")
    assert tilt <= EXECUTED_TILT_LIMIT_DEG
    assert min(row[3] for row in carry) == 3, "a dumpling left the jug during the carry"


class PourStart:
    """The state the closed-loop pour begins from, captured where `pour_prepared` starts."""

    class Captured(Exception):
        """Stops the skill once the pour start has been captured."""

    @staticmethod
    def scripted_prepare(self):  # noqa: PLR0917 -- patched bound method
        """The kitchen's prepare_to_pour before the planned carry, verbatim."""
        from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

        if np.linalg.norm(self.data.qpos[self.base_q][:2] - self.base_home[:2]) < 0.3:
            hand, rotation = self.hand_pose()
            self.move(
                position=hand + [-0.15, 0, 0.25],
                rotation=rotation,
                phase="fold and raise pitcher for transport",
                seconds=6,
                grip=1,
            )
        return PotstickerController.prepare_to_pour(self)

    @staticmethod
    def capture(*, monkeypatch, scripted: bool) -> dict:
        from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

        state: dict = {}

        def stop(self, **_):  # noqa: PLR0917 -- patched bound method
            jug, jug_rotation = self.body_pose(name="pitcher")
            hand, hand_rotation = self.hand_pose()
            joint = self.model.body_jntadr[self.model.body("pitcher").id]
            address = self.model.jnt_dofadr[joint]
            state.update(
                hand=hand,
                hand_rotation=hand_rotation,
                jug=jug,
                jug_tilt=np.rad2deg(np.arccos(np.clip(jug_rotation.apply([0, 0, 1])[2], -1, 1))),
                jug_spin=np.rad2deg(np.linalg.norm(self.data.qvel[address + 3 : address + 6])),
                q=np.rad2deg(self.data.qpos[self.arm_q]),
                posture=self.posture,
                wrist_headroom=np.rad2deg(self.upper[5] - self.data.qpos[self.arm_q][5]),
            )
            raise PourStart.Captured

        with monkeypatch.context() as patch:
            if scripted:
                patch.setattr(KitchenController, "prepare_to_pour", PourStart.scripted_prepare)
            patch.setattr(KitchenController, "closed_loop_pour", stop)
            env = KitchenEnvironment(requested_count=1)
            env.take_action(action=PICK_PITCHER)
            assert env.last_info["success"] is True, env.last_info
            with pytest.raises(PourStart.Captured):
                from hitl_pmp.environments.pouring3d.kitchen.types import (
                    ClosedLoopPourParameters,
                )

                env.controller().run_closed_loop_pour(
                    parameters=ClosedLoopPourParameters(ramp_rate_deg_s=1.0),
                    requested_count=1,
                )
        return state


# Decision 1(b), Josh 2026-11-02: the planned carry ends with the scripted carry's own
# closed-loop correction, levelling the jug from its observed pose. Parity with the
# scripted start is not exact (the scripted IK is path-dependent and its base carry is
# closed-loop on a compliant grasp); hand position and jug tilt are held, the rest is
# printed as the residual gap. Measured 2026-11-02 with three levelling rounds: hand
# 2.3 mm and 0.97 deg apart, joints within 1.3 deg, posture 1.00 on both; the jug stands
# 0.7 deg off vertical against the scripted 3.3 deg (it is levelled further than the
# scripted carry, which corrects once before its pivot). Gap 2.5 deg, held to 3.
JUG_TILT_TOLERANCE_DEG = 3.0


def test_the_planned_pour_starts_with_the_jug_levelled_like_the_scripted_start(*, monkeypatch):
    scripted = PourStart.capture(monkeypatch=monkeypatch, scripted=True)
    planned = PourStart.capture(monkeypatch=monkeypatch, scripted=False)
    hand_gap = np.linalg.norm(planned["hand"] - scripted["hand"])
    turn = np.rad2deg((planned["hand_rotation"] * scripted["hand_rotation"].inv()).magnitude())
    jug_gap = np.linalg.norm(planned["jug"] - scripted["jug"])
    print(
        f"pour start, planned vs scripted: hand {hand_gap * 1000:.1f} mm, {turn:.2f} deg; "
        f"jug {jug_gap * 1000:.1f} mm, tilt {planned['jug_tilt']:.1f} vs "
        f"{scripted['jug_tilt']:.1f} deg, spin {planned['jug_spin']:.2f} vs "
        f"{scripted['jug_spin']:.2f} deg/s; posture {planned['posture']:.2f} vs "
        f"{scripted['posture']:.2f}; joint-6 headroom {planned['wrist_headroom']:.1f} vs "
        f"{scripted['wrist_headroom']:.1f} deg; q {np.round(planned['q'], 1)} vs "
        f"{np.round(scripted['q'], 1)}"
    )
    assert hand_gap < 0.005
    assert abs(planned["jug_tilt"] - scripted["jug_tilt"]) < JUG_TILT_TOLERANCE_DEG
