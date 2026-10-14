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
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DActions
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
