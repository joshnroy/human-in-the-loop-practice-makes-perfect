"""`PickPlannable`: false exactly where the dispatched pick has already refused to plan."""

import subprocess
import sys

import mujoco
import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.pick_plannable import (
    PICK_PLANNABLE,
    KitchenPickPlannable,
)
from hitl_pmp.environments.pouring3d.kitchen.predicates import KitchenPredicates as P
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

pytestmark = pytest.mark.xdist_group("kitchen_pick_plannable")

DUMPLING = 2
PICK = 3  # PickShrimpFromTray in practice.skills.SKILL_NAMES
REFUSAL = "No feasible continuation from prepared arm postures"
PICK_ACTION = np.array([PICK, DUMPLING, 1, 0, 0, 0], dtype=float)
# At a 0.84 m hob a dumpling 6 cm from the pan centre, on the -y side, has no
# collision-free hover for any of the four grasp candidates (measured 2026-11-02).
UNPLANNABLE_OFFSET = (0.0, -0.06)


def test_every_dumpling_pick_requires_the_atom_but_the_human_reset_does_not():
    skills = {s.name: s for s in KitchenSkillProvider.domain(count=3)}
    for name in ("PickShrimpFromTray", "PickShrimpFromFloor", "PickShrimpFromSurface"):
        pick = skills[name]
        required = {(a.predicate.name, a.variables) for a in pick.preconditions}
        assert (PICK_PLANNABLE, pick.parameters[:2]) in required
    reset = skills[HUMAN_RESET]
    assert PICK_PLANNABLE not in {a.predicate.name for a in reset.preconditions}
    for name in (HUMAN_RESET, "Pour1", "Pour2", "Pour3"):
        assert PICK_PLANNABLE in {p.name for p in skills[name].ignore_effects}
    for name in ("PickPitcher", "PlacePitcherAtHome", "DropShrimpIntoPitcher"):
        assert PICK_PLANNABLE not in {p.name for p in skills[name].ignore_effects}


@pytest.mark.parametrize(
    ("error", "planning"),
    [
        ("No feasible continuation from prepared arm postures", True),
        ("No collision-free empty-arm preparation path: residual 0.0410, reachable poses 0", True),
        ("IK infeasible in prepare arm in clear aisle: residual 0.0120; joint degrees []", True),
        ("IK infeasible in position for shrimp pick: residual 0.0120; joint degrees []", True),
        ("IK infeasible in approach above raised shrimp: residual 0.02; joint degrees []", True),
        ("IK infeasible in approach shrimp: residual 0.0120; joint degrees []", True),
        ("Insufficient obstacle clearance in approach shrimp: robot_bracelet_link/x", True),
        ("Insufficient obstacle clearance in position for shrimp pick: a/b", True),
        # Contact motion and physical outcomes are attempts, never refusals.
        ("IK infeasible in descend to shrimp body: residual 0.0120; joint degrees []", False),
        ("Insufficient obstacle clearance in descend to shrimp body: a/b", False),
        ("Motor tracking failed in descend to shrimp body", False),
        ("Motor tracking failed in position for shrimp pick", False),
        ("Base tracking failed in clear aisle for shrimp pick", False),
        ("Physical shrimp grasp lost", False),
        ("Shrimp grasp lacks both finger contacts", False),
        ("Shrimp not retained through lift", False),
        ("Per-skill control-step limit exceeded", False),
        ("Shrimp pick requires an empty hand and the named source", False),
    ],
)
def test_exactly_the_enumerated_planning_errors_count_as_refusals(*, error, planning):
    assert KitchenPickPlannable.is_planning_refusal(error=error) is planning


def test_the_cli_import_path_never_imports_mujoco():
    # CI's unit job has no OSMesa: importing MuJoCo at module level kills collection.
    probe = "import sys, hitl_pmp.cli; sys.exit('mujoco' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


class Planning:
    """The symbolic model alone: a dumpling in the pan, the jug at home."""

    @staticmethod
    def atom(name: str, *objects: object) -> GroundAtom:  # noqa: PLR0917
        return GroundAtom(predicate=P.get(name=name), objects=objects)

    @staticmethod
    def plan(*, plannable: bool, offer_reset: bool) -> list[str]:
        from hitl_pmp.planning.fast_downward import FastDownwardPlanner

        provider = KitchenSkillProvider(env=KitchenEnvironment(requested_count=1))
        o = provider.env.objects()
        pick = provider.ground(name="PickShrimpFromTray")
        shrimp, pitcher = o.shrimp()[0], o.pitcher
        init = set(pick.preconditions) | {
            Planning.atom("PitcherHome", pitcher),
            *(Planning.atom("InPitcher", s, pitcher) for s in o.shrimp()[1:]),
            *(
                Planning.atom("PayloadOrder", a, b)
                for a, b in zip(o.shrimp(), o.shrimp()[1:], strict=False)
            ),
        }
        if not plannable:
            init = {a for a in init if a.predicate.name != PICK_PLANNABLE}
        reset = provider.ground(name=HUMAN_RESET)
        skills = provider.skills() + ((reset.skill,) if offer_reset else ())
        plan = FastDownwardPlanner.plan(
            skills=skills,
            objects=provider.objects(),
            predicates=provider.predicates(),
            types=provider.types(),
            init_atoms=frozenset(init),
            goal=frozenset({Planning.atom("InPitcher", shrimp, pitcher)}),
            ground_skill_costs={reset: 100.0},
        )
        return [g.skill.name for g in plan]


def test_with_the_atom_false_the_human_pour_back_is_the_only_route():
    from hitl_pmp.planning.fast_downward import PlanningFailure

    assert Planning.plan(plannable=True, offer_reset=True) == [
        "PickShrimpFromTray",
        "DropShrimpIntoPitcher",
    ]
    assert Planning.plan(plannable=False, offer_reset=True) == [HUMAN_RESET]
    with pytest.raises(PlanningFailure):
        Planning.plan(plannable=False, offer_reset=False)


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
        # Dropped 3 cm, a dumpling still tumbles for a few seconds; a refusal is
        # keyed on its pose, so the scene must start at rest.
        # (measured: at rest after about 8 s).
        c.pause(phase="settle test dumpling", seconds=10, grip=0)
        rest = d.qpos[joint : joint + 3].copy()
        c.pause(phase="confirm test dumpling at rest", seconds=1, grip=0)
        assert np.linalg.norm(d.qpos[joint : joint + 3] - rest) < 1e-3
        env.current_state = env.observe()
        assert c.scene.cube_locations()[DUMPLING] == "tray"
        return env

    @staticmethod
    def nudge(*, env: KitchenEnvironment, dx: float) -> None:
        c = env.controller()
        m, d = c.model, c.data
        joint = int(m.jnt_qposadr[m.body(c.scene.cube_names[DUMPLING]).jntadr[0]])
        d.qpos[joint] += dx
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        c.pause(phase="settle nudged dumpling", seconds=3, grip=0)

    @staticmethod
    def plannable(*, env: KitchenEnvironment) -> bool:
        name = env.controller().scene.cube_names[DUMPLING]
        return (PICK_PLANNABLE, ("robot", name)) in env.observe().abstract_atoms

    @staticmethod
    def fake_pick(*, monkeypatch, error: str, touch: bool = False) -> None:
        """Replace the dispatched pick: tick the simulation, then fail with `error`."""

        def pick(self, **_):  # noqa: PLR0917 -- patched bound method
            if touch:
                monkeypatch.setattr(type(self), "finger_contacts", lambda *_, **__: {"left"})
            self.pause(phase="position for shrimp pick", seconds=0.5, grip=0)
            raise ValueError(error)

        monkeypatch.setattr(PotstickerController, "pick_payload_with_retry", pick)


@pytest.fixture
def in_pan() -> KitchenEnvironment:
    return Scenes.dumpling_in_pan(hob=0.74, offset=(0.0, 0.0))


def test_an_untried_dumpling_is_plannable_wherever_it_is(*, in_pan):
    names = in_pan.controller().scene.cube_names
    atoms = in_pan.observe().abstract_atoms
    assert {(PICK_PLANNABLE, ("robot", n)) for n in names} <= atoms
    fresh = KitchenEnvironment(requested_count=1)
    assert {(PICK_PLANNABLE, ("robot", n)) for n in names} <= fresh.observe().abstract_atoms


def test_a_real_planning_refusal_makes_the_atom_false():
    env = Scenes.dumpling_in_pan(hob=0.84, offset=UNPLANNABLE_OFFSET)
    assert Scenes.plannable(env=env)
    env.take_action(action=PICK_ACTION)
    assert env.last_info["success"] is False
    assert KitchenPickPlannable.is_planning_refusal(error=env.last_info["error"])
    assert not Scenes.plannable(env=env)


def test_a_planning_message_records_only_without_contact_or_motion(*, in_pan, monkeypatch):
    Scenes.fake_pick(monkeypatch=monkeypatch, error=REFUSAL)
    in_pan.take_action(action=PICK_ACTION)
    assert not Scenes.plannable(env=in_pan)

    touched = Scenes.dumpling_in_pan(hob=0.74, offset=(0.0, 0.0))
    Scenes.fake_pick(monkeypatch=monkeypatch, error=REFUSAL, touch=True)
    touched.take_action(action=PICK_ACTION)
    assert Scenes.plannable(env=touched)


def test_a_physical_slip_is_an_attempt_not_a_refusal(*, in_pan, monkeypatch):
    Scenes.fake_pick(monkeypatch=monkeypatch, error="Physical shrimp grasp lost")
    in_pan.take_action(action=PICK_ACTION)
    assert in_pan.last_info["error"] == "Physical shrimp grasp lost"
    assert Scenes.plannable(env=in_pan)


def test_moving_the_dumpling_clears_the_refusal(*, in_pan, monkeypatch):
    Scenes.fake_pick(monkeypatch=monkeypatch, error=REFUSAL)
    in_pan.take_action(action=PICK_ACTION)
    assert not Scenes.plannable(env=in_pan)
    Scenes.nudge(env=in_pan, dx=0.02)
    assert Scenes.plannable(env=in_pan)
    # Cleared, not merely masked: the original pose no longer refuses either.
    Scenes.nudge(env=in_pan, dx=-0.02)
    assert Scenes.plannable(env=in_pan)


def test_the_memory_belongs_to_one_environment_instance(*, in_pan, monkeypatch):
    Scenes.fake_pick(monkeypatch=monkeypatch, error=REFUSAL)
    in_pan.take_action(action=PICK_ACTION)
    assert not Scenes.plannable(env=in_pan)
    other = Scenes.dumpling_in_pan(hob=0.74, offset=(0.0, 0.0))
    assert Scenes.plannable(env=other)


def test_a_physical_failure_repeated_identically_from_the_same_pose_is_a_refusal(
    *, in_pan, monkeypatch
):
    # EES seed 1 dispatched one pick 23 times into the same tracking failure: a
    # deterministic failure from an unchanged state is as certain as a planning refusal.
    error = "Motor tracking failed in descend to shrimp body"
    Scenes.fake_pick(monkeypatch=monkeypatch, error=error, touch=True)
    in_pan.take_action(action=PICK_ACTION)
    assert Scenes.plannable(env=in_pan)
    in_pan.take_action(action=PICK_ACTION)
    assert not Scenes.plannable(env=in_pan)


def test_a_different_error_or_a_moved_dumpling_is_not_a_repeat(*, in_pan, monkeypatch):
    Scenes.fake_pick(monkeypatch=monkeypatch, error="Shrimp grasp lacks both finger contacts")
    in_pan.take_action(action=PICK_ACTION)
    Scenes.fake_pick(monkeypatch=monkeypatch, error="Physical shrimp grasp lost")
    in_pan.take_action(action=PICK_ACTION)
    assert Scenes.plannable(env=in_pan)
    Scenes.nudge(env=in_pan, dx=0.02)
    in_pan.take_action(action=PICK_ACTION)
    assert Scenes.plannable(env=in_pan)
