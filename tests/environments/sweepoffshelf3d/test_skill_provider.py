"""SweepOffShelf3D's action encoding and its dispatch to the controllers, off-simulator."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweepoffshelf3d.controllers import SweepOffShelf3DControllers
from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import (
    ORACLE_SWEEP_PARAMS,
    SweepOffShelf3DActions,
    SweepOffShelf3DSkillProvider,
)
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S


def test_every_robot_skill_has_its_own_action_id():
    names = [skill.name for skill in S.robot_skills()]
    assert sorted(names) == sorted(SKILL_IDS)
    assert len(set(SKILL_IDS.values())) == len(SKILL_IDS)


@pytest.mark.parametrize(
    ("skill", "objects"),
    [
        (S.RETURN_CUBE_TO_SHELF_FROM_BOX, (O.robot, O.cubes[3], O.box)),
        (S.RETURN_CUBE_TO_SHELF_FROM_FLOOR, (O.robot, O.cubes[3])),
    ],
)
def test_a_return_runs_the_return_controller_on_its_cube(*, skill, objects, monkeypatch):
    calls = []
    monkeypatch.setattr(
        SweepOffShelf3DControllers,
        "return_cube_to_shelf",
        lambda self, *, cube: calls.append(cube),
    )
    action = SweepOffShelf3DActions.encode(
        ground_skill=GroundSkill(skill=skill, objects=objects), params=np.zeros(0)
    )
    assert action[1] == 3.0
    env = SweepOffShelf3DEnvironment()
    env._dispatch(skill_id=int(action[0]), values=action)  # noqa: SLF001
    assert calls == ["cube_3"]


def _sweep(*, cube):
    return GroundSkill(skill=S.SWEEP_SHELF, objects=(O.robot, O.wiper, O.box, O.cubes[cube]))


def test_a_sweep_encodes_its_target_cube_then_offset_and_speed():
    action = SweepOffShelf3DActions.encode(
        ground_skill=_sweep(cube=4), params=np.array([-0.06, 0.1])
    )
    assert action.tolist() == [float(SKILL_IDS["SweepShelf"]), 4.0, -0.06, 0.1]


def test_a_sweep_runs_the_sweep_controller_on_its_cube(*, monkeypatch):
    calls = []
    monkeypatch.setattr(
        SweepOffShelf3DControllers,
        "sweep_shelf",
        lambda self, *, cube, offset, speed: calls.append((cube, offset, speed)),
    )
    action = SweepOffShelf3DActions.encode(
        ground_skill=_sweep(cube=2), params=np.array([0.07, 0.2])
    )
    env = SweepOffShelf3DEnvironment()
    env._dispatch(skill_id=int(action[0]), values=action)  # noqa: SLF001
    assert calls == [("cube_2", 0.07, 0.2)]


def test_sampled_sweep_parameters_stay_inside_offset_and_speed_bounds():
    provider = SweepOffShelf3DSkillProvider(env=SweepOffShelf3DEnvironment())
    rng = np.random.default_rng(0)
    draws = np.array([
        provider.sample_params(ground_skill=_sweep(cube=0), rng=rng) for _ in range(500)
    ])
    assert draws.shape == (500, 2)
    (o_low, o_high), (v_low, v_high) = S.SWEEP_BOUNDS
    assert draws[:, 0].min() >= o_low and draws[:, 0].max() <= o_high
    assert draws[:, 1].min() >= v_low and draws[:, 1].max() <= v_high
    # Both signs of offset are drawn: the lane goes on either side of the cube.
    assert (draws[:, 0] < 0).any() and (draws[:, 0] > 0).any()


def test_the_oracle_sweeps_one_shelf_cube_inside_the_bounds():
    (o_low, o_high), (v_low, v_high) = S.SWEEP_BOUNDS
    offset, speed = ORACLE_SWEEP_PARAMS
    assert o_low <= offset <= o_high and v_low <= speed <= v_high
