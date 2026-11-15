"""`PlacePlannable`: false exactly where the dispatched placement has already refused to plan.

Curtis seed 8 of pouring-cost-10seed-20261112 (human-reset cost 10) dispatched
PlacePitcherAtHome 117 times in a row from one state, each a 0-step "No collision-free
base path to placement", until practice ended `repeated_zero_step`: nothing symbolic
said the placement could not plan there, and each 0-step failure only lowered the
skill's forecast practice cost. As `PickPlannable` does for the dumpling picks, the
first refusal now makes the atom false in that state, so the planner can choose the
human pour-back, which re-asserts it.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.kitchen.place_plannable import PLACE_PLANNABLE
from hitl_pmp.environments.pouring3d.kitchen.predicates import KitchenPredicates as P
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

pytest.importorskip("kinder")

pytestmark = pytest.mark.xdist_group("kitchen_place_plannable")

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_floor_dumpling_in_base_footprint.json"
RECORDED = json.loads(FIXTURE.read_text())
PLACE = np.array([2, 0, 1, 0, 0, 0], dtype=float)  # PlacePitcherAtHome in SKILL_NAMES
REFUSAL = "No collision-free base path to placement"
ATOM = (PLACE_PLANNABLE, ("robot", "pitcher"))


def test_the_placement_requires_the_atom_and_the_human_reset_restores_it():
    skills = {s.name: s for s in KitchenSkillProvider.domain(count=3)}
    place = skills["PlacePitcherAtHome"]
    required = {(a.predicate.name, tuple(v.name for v in a.variables)) for a in place.preconditions}
    assert (PLACE_PLANNABLE, ("r", "p")) in required
    reset = skills[HUMAN_RESET]
    assert PLACE_PLANNABLE not in {a.predicate.name for a in reset.preconditions}
    added = {(a.predicate.name, tuple(v.name for v in a.variables)) for a in reset.add_effects}
    assert (PLACE_PLANNABLE, ("r", "p")) in added
    for skill in skills.values():
        assert PLACE_PLANNABLE not in {p.name for p in skill.ignore_effects}


class Planning:
    """A static-method container, never instantiated."""

    @staticmethod
    def atom(name: str, *objects: object) -> GroundAtom:  # noqa: PLR0917
        return GroundAtom(predicate=P.get(name=name), objects=objects)

    @staticmethod
    def plan(*, plannable: bool, offer_reset: bool) -> list[str]:
        from hitl_pmp.planning.fast_downward import FastDownwardPlanner

        provider = KitchenSkillProvider(env=KitchenEnvironment(requested_count=1))
        o = provider.env.objects()
        place = provider.ground(name="PlacePitcherAtHome")
        init = set(place.preconditions) | {
            *(Planning.atom("InPitcher", s, o.pitcher) for s in o.shrimp()),
            *(
                Planning.atom("PayloadOrder", a, b)
                for a, b in zip(o.shrimp(), o.shrimp()[1:], strict=False)
            ),
        }
        if not plannable:
            init = {a for a in init if a.predicate.name != PLACE_PLANNABLE}
        reset = provider.ground(name=HUMAN_RESET)
        skills = provider.skills() + ((reset.skill,) if offer_reset else ())
        plan = FastDownwardPlanner.plan(
            skills=skills,
            objects=provider.objects(),
            predicates=provider.predicates(),
            types=provider.types(),
            init_atoms=frozenset(init),
            goal=frozenset({Planning.atom("PitcherHome", o.pitcher)}),
            ground_skill_costs={reset: 100.0},
        )
        return [g.skill.name for g in plan]


def test_with_the_atom_false_the_human_pour_back_is_the_only_route():
    from hitl_pmp.planning.fast_downward import PlanningFailure

    assert Planning.plan(plannable=True, offer_reset=True) == ["PlacePitcherAtHome"]
    assert Planning.plan(plannable=False, offer_reset=True) == [HUMAN_RESET]
    with pytest.raises(PlanningFailure):
        Planning.plan(plannable=False, offer_reset=False)


class Stall:
    """The recorded stall: the jug welded in the hand, a dumpling under the base."""

    @staticmethod
    def restored() -> KitchenEnvironment:
        env = KitchenEnvironment(requested_count=RECORDED["requested"])
        env.reset_to_seed(seed=RECORDED["scene_seed"], requested=RECORDED["requested"])
        c = env.controller()
        c.data.qpos[:] = RECORDED["qpos"]
        c.data.qvel[:] = RECORDED["qvel"]
        c.data.time = RECORDED["time"]
        mujoco.mj_forward(c.model, c.data)
        c.scene.attach_pitcher()
        mujoco.mj_forward(c.model, c.data)
        c.held = "pitcher"
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        env.current_state = env.observe()
        return env

    @staticmethod
    def plannable(*, env: KitchenEnvironment) -> bool:
        return ATOM in env.observe().abstract_atoms

    @staticmethod
    def fake_place(*, monkeypatch, error: str, seconds: float = 0.0) -> None:
        """Replace the dispatched placement: optionally tick, then fail with `error`."""

        def place(self, **_):  # noqa: PLR0917 -- patched bound method
            if seconds:
                self.pause(phase="planned tuck jug for return", seconds=seconds, grip=1)
            raise ValueError(error)

        monkeypatch.setattr(KitchenController, "place_pitcher", place)


@pytest.fixture
def stall() -> KitchenEnvironment:
    env = Stall.restored()
    yield env
    env.close()


def test_an_untried_placement_is_plannable(*, stall):
    assert Stall.plannable(env=stall)
    assert ATOM in KitchenEnvironment(requested_count=1).observe().abstract_atoms


def test_a_real_planning_refusal_makes_the_atom_false(*, stall, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    # Without the floor-dumpling escape the real planner refuses, through the real path.
    monkeypatch.setattr(ps, "FLOOR_DUMPLING_MAX_Z_M", 0.0)
    stall.take_action(action=PLACE)
    assert stall.last_info["error"] == REFUSAL
    assert stall.last_info["controller_steps"] == 0
    assert not Stall.plannable(env=stall)
    assert ATOM not in stall.current_state.abstract_atoms


def test_a_failure_after_motion_is_an_attempt_not_a_refusal(*, stall, monkeypatch):
    Stall.fake_place(monkeypatch=monkeypatch, error="Pitcher slipped", seconds=0.5)
    stall.take_action(action=PLACE)
    assert stall.last_info["controller_steps"] > 0
    assert Stall.plannable(env=stall)


def test_moving_the_base_clears_the_refusal(*, stall, monkeypatch):
    Stall.fake_place(monkeypatch=monkeypatch, error=REFUSAL)
    stall.take_action(action=PLACE)
    assert not Stall.plannable(env=stall)
    c = stall.controller()
    c.data.qpos[c.base_q[0]] += 0.02
    mujoco.mj_forward(c.model, c.data)
    assert Stall.plannable(env=stall)
    # Cleared, not merely masked: the original pose no longer refuses either.
    c.data.qpos[c.base_q[0]] -= 0.02
    mujoco.mj_forward(c.model, c.data)
    assert Stall.plannable(env=stall)


def test_the_human_reset_and_a_held_out_task_forget_the_refusal(*, stall, monkeypatch):
    Stall.fake_place(monkeypatch=monkeypatch, error=REFUSAL)
    stall.take_action(action=PLACE)
    assert not Stall.plannable(env=stall)
    stall.forget_pick_refusals()
    assert Stall.plannable(env=stall)
    stall.take_action(action=PLACE)
    assert not Stall.plannable(env=stall)
    assert stall.reset_movables()
    assert Stall.plannable(env=stall)


def test_placement_from_the_recorded_stall_now_plans_and_places(*, stall):
    stall.take_action(action=PLACE)
    assert stall.last_info["error"] != REFUSAL, stall.last_info
    assert stall.last_info["controller_steps"] > 0
    assert stall.last_info["success"] is True, stall.last_info
    assert Stall.plannable(env=stall)


def test_a_simulation_checkpoint_keeps_the_refusal(*, stall, monkeypatch):
    """A resumed practice process must not re-dispatch a placement already refused."""
    import pickle

    Stall.fake_place(monkeypatch=monkeypatch, error=REFUSAL)
    stall.take_action(action=PLACE)
    assert not Stall.plannable(env=stall)
    checkpoint = pickle.loads(pickle.dumps(stall.simulation_checkpoint()))
    resumed = KitchenEnvironment(requested_count=RECORDED["requested"])
    try:
        resumed.restore_simulation_checkpoint(checkpoint=checkpoint)
        assert not Stall.plannable(env=resumed)
    finally:
        resumed.close()
