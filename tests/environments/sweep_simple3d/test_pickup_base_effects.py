"""A physically held tool can be acquired while the base remains natively home."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
from hitl_pmp.environments.sweep_simple3d.symbolic import SimpleSymbols


@pytest.mark.parametrize("home", [True, False])
@pytest.mark.parametrize("held", [True, False])
def test_pickup_success_depends_on_holding_and_preserves_observed_base(
    *, monkeypatch: pytest.MonkeyPatch, home: bool, held: bool
) -> None:
    def observed(*, holding: bool, robot_home: bool) -> State:
        facts = dict(HandEmpty=not holding, HoldingWiper=holding, WiperAvailable=not holding,
                     WiperHome=not holding, RobotHome=robot_home, RobotAway=not robot_home,
                     ClosedEmpty=False, seed=2)
        return State(data={SimpleSymbols.SCENE: np.array([
            float(facts[n]) for n in SimpleSymbols.SCENE_TYPE.feature_names
        ])})

    initial = observed(holding=False, robot_home=True)
    final = observed(holding=held, robot_home=home)
    env = SweepSimpleEnvironment.model_construct(current_state=initial)
    endings = []
    session = SimpleNamespace(ticks=0, begin=lambda **kwargs: None,
                              end=lambda **kwargs: endings.append(kwargs))
    env._session = session
    env._primitive = SimpleNamespace(distance=.7, heading_offset=0., recover_wiper=lambda: None)
    monkeypatch.setattr(SweepSimpleEnvironment, "observe", lambda self: final)
    monkeypatch.setattr(SweepSimpleEnvironment, "_write_event", lambda self, **kwargs: None)
    result = env.take_action(action=np.array([env.ACTION_NAMES.index("PickFloorWiper"),
                                             -1, .7, 0.]))
    assert endings == [{"success": held, "note": ""}]
    assert result.get(obj=SimpleSymbols.SCENE, feature_name="RobotHome") == float(home)
    assert result.get(obj=SimpleSymbols.SCENE, feature_name="RobotAway") == float(not home)
    assert env._action_count == 1
    skill = next(s for s in SimpleSymbols.skills() if s.name == "PickFloorWiper")
    assert {p.name for p in skill.ignore_effects} == {"RobotHome", "RobotAway"}
    assert {a.predicate.name for a in skill.add_effects} == {"HoldingWiper"}
    assert {"RobotHome", "RobotAway"} <= {a.predicate.name for a in skill.delete_effects}


@pytest.mark.parametrize("name", ["SweepCubeToGoal", "SweepCubeToStart"])
@pytest.mark.parametrize("attained", [True, False])
def test_sweep_target_outcome_is_independent_of_observed_home_base(
    *, monkeypatch: pytest.MonkeyPatch, name: str, attained: bool
) -> None:
    reverse = name == "SweepCubeToStart"

    def observed(*, done: bool) -> State:
        scene = dict(HandEmpty=False, HoldingWiper=True, WiperAvailable=False,
                     WiperHome=False, RobotHome=True, RobotAway=False, ClosedEmpty=False, seed=2)
        in_goal = (not done) if reverse else done
        cube = dict(AtStart=not in_goal, NotAtStart=in_goal, InGoal=in_goal,
                    NotInGoal=not in_goal, OnFloor=True, x=0., y=0., z=.01)
        return State(data={
            SimpleSymbols.SCENE: np.array([float(scene[n])
                                           for n in SimpleSymbols.SCENE_TYPE.feature_names]),
            SimpleSymbols.CUBES[0]: np.array([float(cube[n])
                                              for n in SimpleSymbols.CUBE_TYPE.feature_names]),
        })

    env = SweepSimpleEnvironment.model_construct(current_state=observed(done=False))
    final = observed(done=attained)
    endings = []
    env._session = SimpleNamespace(ticks=0, begin=lambda **kwargs: None,
                                   end=lambda **kwargs: endings.append(kwargs))
    env._primitive = SimpleNamespace(sweep_cube=lambda **kwargs: None)
    monkeypatch.setattr(SweepSimpleEnvironment, "observe", lambda self: final)
    monkeypatch.setattr(SweepSimpleEnvironment, "_write_event", lambda self, **kwargs: None)
    result = env.take_action(action=np.array([env.ACTION_NAMES.index(name), 0, .55, 0.]))
    assert endings == [{"success": attained, "note": ""}]
    assert result.get(obj=SimpleSymbols.SCENE, feature_name="RobotHome") == 1.
    assert result.get(obj=SimpleSymbols.SCENE, feature_name="RobotAway") == 0.
    skill = next(s for s in SimpleSymbols.skills() if s.name == name)
    assert {p.name for p in skill.ignore_effects} == {"RobotHome", "RobotAway"}
    assert not {"RobotHome", "RobotAway"} & {a.predicate.name for a in skill.add_effects}
    assert {"RobotHome", "RobotAway"} <= {a.predicate.name for a in skill.delete_effects}
