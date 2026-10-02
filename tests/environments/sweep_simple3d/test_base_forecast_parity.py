"""Unknown base location must not block or prematurely complete robot reset."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
from hitl_pmp.environments.sweep_simple3d.symbolic import SimpleSymbols
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects
from hitl_pmp.planning.pddl import PddlWriter


def test_reset_trace_matches_pddl_and_belief_forecasts() -> None:
    skills = {s.name: s for s in SimpleSymbols.skills()}
    classical = belief = SimpleSymbols.initial_atoms()
    names = ("PickFloorWiper", "SweepCubeToGoal", "SweepCubeToStart",
             "PlaceWiperAtStart", "ReturnRobotToStart")
    for name in names:
        skill = skills[name]
        objects = ((SimpleSymbols.SCENE, SimpleSymbols.CUBES[0])
                   if len(skill.parameters) == 2 else (SimpleSymbols.SCENE,))
        ground = GroundSkill(skill=skill, objects=objects)
        assert ground.preconditions <= classical
        assert ground.preconditions <= belief
        # Execute the universally quantified deletes emitted for classical EES.
        pddl = PddlWriter._action_str(skill=skill)
        for predicate in ground.ignore_effects:
            assert f"(not ({predicate.name} ?x0))" in pddl
        classical = frozenset(
            a for a in classical
            if a.predicate not in ground.ignore_effects and a not in ground.delete_effects
        ) | ground.add_effects
        belief = apply_success_effects(
            true_atoms=belief, ground_skill=ground,
            effects={ground: (ground.add_effects, ground.delete_effects, ground.ignore_effects)},
        )
        assert classical == belief
        base = {a.predicate.name for a in belief
                if a.predicate.name in ("RobotHome", "RobotAway")}
        assert base == ({"RobotHome"} if name == "ReturnRobotToStart" else set())
        assert (SimpleSymbols.initial_atoms() <= belief) == (name == "ReturnRobotToStart")
    assert belief == SimpleSymbols.initial_atoms()


def test_return_executes_when_already_home(*, monkeypatch: pytest.MonkeyPatch) -> None:
    facts = dict(HandEmpty=True, HoldingWiper=False, WiperAvailable=True,
                 WiperHome=True, RobotHome=True, RobotAway=False, ClosedEmpty=False, seed=0)
    state = State(data={SimpleSymbols.SCENE: np.array([
        float(facts[n]) for n in SimpleSymbols.SCENE_TYPE.feature_names
    ])})
    env = SweepSimpleEnvironment.model_construct(current_state=state)
    endings = []
    drives = []
    env._session = SimpleNamespace(ticks=0, begin=lambda **kwargs: None,
                                   end=lambda **kwargs: endings.append(kwargs))
    env._initial_state = SimpleNamespace(get_object_from_name=lambda name: "robot",
                                         get=lambda obj, name: 0.)
    env._primitive = SimpleNamespace(
        scene=SimpleNamespace(plan_arm=lambda **kwargs: [np.zeros(7)],
                              bodies=lambda: set(), wiper_body=1),
        motion=SimpleNamespace(follow=lambda **kwargs: True,
                               drive_to=lambda **kwargs: drives.append(kwargs)),
    )
    monkeypatch.setattr(SweepSimpleEnvironment, "observe", lambda self: state)
    monkeypatch.setattr(SweepSimpleEnvironment, "_write_event", lambda self, **kwargs: None)
    result = env.take_action(action=np.array([
        env.ACTION_NAMES.index("ReturnRobotToStart"), -1, .7, 0.
    ]))
    assert endings == [{"success": True, "note": ""}]
    assert drives == [{"target": (0., 0., 0.), "grip": 0.}]
    assert result is state
