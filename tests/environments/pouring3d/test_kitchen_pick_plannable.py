"""`PickPlannable`: a dry run of the dumpling pick's own planner, restored exactly."""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.kitchen.pick_plannable import PICK_PLANNABLE
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

pytestmark = pytest.mark.xdist_group("kitchen_pick_plannable")

DUMPLING = 2
PICK = 3  # PickShrimpFromTray in practice.skills.SKILL_NAMES
PLANNING_REFUSALS = (
    "No feasible continuation from prepared arm postures",
    "No collision-free empty-arm preparation path",
    "Base path lacks",
)


# At the old 0.84 m hob a dumpling 6 cm from the pan centre, on the -y side, has no
# collision-free hover for any of the four grasp candidates (measured 2026-11-02).
UNPLANNABLE_OFFSET = (0.0, -0.06)


def test_every_dumpling_pick_requires_its_plan_but_the_human_reset_does_not():
    skills = {s.name: s for s in KitchenSkillProvider.domain(count=3)}
    for name in ("PickShrimpFromTray", "PickShrimpFromFloor", "PickShrimpFromSurface"):
        pick = skills[name]
        required = {(a.predicate.name, a.variables) for a in pick.preconditions}
        assert (PICK_PLANNABLE, pick.parameters[:2]) in required
    reset = skills[HUMAN_RESET]
    assert PICK_PLANNABLE not in {a.predicate.name for a in reset.preconditions}
    # Skills that move dumplings forget the verdict; it is re-read on observation.
    for name in (HUMAN_RESET, "Pour1", "Pour2", "Pour3"):
        assert PICK_PLANNABLE in {p.name for p in skills[name].ignore_effects}
    for name in ("PickPitcher", "PlacePitcherAtHome", "DropShrimpIntoPitcher"):
        assert PICK_PLANNABLE not in {p.name for p in skills[name].ignore_effects}


class Scenes:
    """A dumpling settled in the pan at a fixed offset from its centre."""

    @staticmethod
    def dumpling_in_pan(*, hob: float, offset: tuple[float, float]) -> KitchenEnvironment:
        env = KitchenEnvironment(geometry=KitchenGeometry(hob_height_m=hob), requested_count=1)
        c = env.controller()
        m, d = c.model, c.data
        joint = int(m.jnt_qposadr[m.body(c.scene.cube_names[DUMPLING]).jntadr[0]])
        pan = d.xpos[m.body("tray").id]
        d.qpos[joint : joint + 3] = [pan[0] + offset[0], pan[1] + offset[1], pan[2] + 0.03]
        d.qpos[joint + 3 : joint + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        c.pause(phase="settle test dumpling", seconds=2, grip=0)
        env.current_state = env.observe()
        assert c.scene.cube_locations()[DUMPLING] == "tray"
        return env

    @staticmethod
    def snapshot(*, c: KitchenController) -> dict[str, object]:
        d = c.data
        arrays = {
            name: np.array(getattr(d, name), copy=True)
            for name in (
                "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "time", "xpos", "xquat",
                "site_xpos", "qfrc_applied", "userdata", "efc_force",
            )
        }  # fmt: skip
        scalars = {
            "steps": c.steps,
            "phase": c.phase,
            "held": c.held,
            "posture": c.posture,
            "grip": c.grip,
            "last_q": c.last_q.copy(),
            "base_target": c.base_target.copy(),
            "approach_plan": len(c.payload_approach_plan),
            "approach_path": len(c.shrimp_approach_path),
            "recovery": (
                c.recovery_finger_offset,
                c.recovery_open_grip,
                c.recovery_pitch_degrees,
                c.recovery_approach_clearance,
                c.recovery_yaw_offset,
            ),
            "timestep": c.scene._robot_env.timestep,
            "ncon": d.ncon,
        }
        return {**arrays, **scalars}

    @staticmethod
    def assert_same(*, before: dict[str, object], after: dict[str, object]) -> None:
        assert before.keys() == after.keys()
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, after[key]), key
            else:
                assert value == after[key], key


@pytest.fixture(scope="module")
def plannable() -> KitchenEnvironment:
    return Scenes.dumpling_in_pan(hob=0.74, offset=(0.0, 0.0))


def test_the_dry_run_restores_the_simulation_and_controller_exactly(*, plannable):
    c = plannable.controller()
    ticks: list[int] = []
    c.callback = lambda **_: ticks.append(1)
    before = Scenes.snapshot(c=c)
    verdict = c.payload_pick_plan(index=DUMPLING, location="tray")
    assert verdict.plannable
    Scenes.assert_same(before=before, after=Scenes.snapshot(c=c))
    # No frame is recorded and no control step is charged for a dry run.
    assert ticks == []
    c.callback = plannable._motion_sink


def test_the_atom_is_cached_per_state_and_skipped_where_there_is_no_pick(*, plannable, monkeypatch):
    calls: list[int] = []
    original = PotstickerController.payload_pick_plan

    def counted(self, **kwargs):  # noqa: PLR0917 -- patched bound method
        calls.append(kwargs["index"])
        return original(self, **kwargs)

    monkeypatch.setattr(PotstickerController, "payload_pick_plan", counted)
    plannable._pick_plans.clear()  # the fixture's own observation filled it
    first = plannable.observe().abstract_atoms
    second = plannable.observe().abstract_atoms
    assert first == second
    # Only the dumpling in the pan is planned; the two in the jug have no pick.
    assert calls == [DUMPLING]
    for name in plannable.controller().scene.cube_names:
        assert (PICK_PLANNABLE, ("robot", name)) in first


def test_the_verdict_matches_what_dispatch_then_plans(*, plannable, monkeypatch):
    c = plannable.controller()
    verdict = c.payload_pick_plan(index=DUMPLING, location="tray")
    assert verdict.plannable and verdict.goal is not None
    dispatched: list[tuple[tuple[float, ...], np.ndarray]] = []
    original = KitchenController.prepare_empty_arm

    def spy(self, **kwargs):  # noqa: PLR0917 -- patched bound method
        original(self, **kwargs)
        if kwargs.get("continuation") is not None:
            dispatched.append((self.recovery_candidate(), self.last_q.copy()))

    monkeypatch.setattr(KitchenController, "prepare_empty_arm", spy)
    plannable.take_action(action=np.array([PICK, DUMPLING, 1, 0, 0, 0], dtype=float))
    # The first plan dispatch finds is the dry run's: same candidate, same joints.
    candidate, goal = dispatched[0]
    assert candidate == verdict.candidate
    assert np.array_equal(goal, np.asarray(verdict.goal))
    error = plannable.last_info["error"]
    assert error is None or not error.startswith(PLANNING_REFUSALS)


def test_an_unplannable_verdict_is_the_refusal_dispatch_then_raises():
    env = Scenes.dumpling_in_pan(hob=0.84, offset=UNPLANNABLE_OFFSET)
    c = env.controller()
    verdict = c.payload_pick_plan(index=DUMPLING, location="tray")
    assert not verdict.plannable
    assert verdict.error is not None and verdict.error.startswith(PLANNING_REFUSALS)
    assert (PICK_PLANNABLE, ("robot", c.scene.cube_names[DUMPLING])) not in (
        env.observe().abstract_atoms
    )
    env.take_action(action=np.array([PICK, DUMPLING, 1, 0, 0, 0], dtype=float))
    assert env.last_info["error"] == verdict.error
    assert c.held is None
