"""No Tossing3D skill binds a side object; sides are robot-relative predicates."""

import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_ON_OPPOSITE_SIDE,
    BIN_ON_ROBOT_SIDE,
    CUBE_ON_OPPOSITE_SIDE,
    CUBE_ON_ROBOT_SIDE,
)
from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.practice_domain import TOSSING3D_DOMAIN
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    RESET_OPPOSITE_SIDE_SKILL,
    RESET_SAME_SIDE_SKILL,
)

ROBOT = Tossing3DEnvironment.robot_type
CUBE = Tossing3DEnvironment.cube_type
BIN = Tossing3DEnvironment.bin_type
BARRIER = Tossing3DEnvironment.barrier_type


def _types(*, skill: object) -> tuple[object, ...]:
    return tuple(variable.type for variable in skill.parameters)  # type: ignore[attr-defined]


def test_pick_and_pour_signatures_have_no_side() -> None:
    assert _types(skill=Tossing3DSkills.PICK_CUBE) == (ROBOT, CUBE, BARRIER, BIN)
    assert _types(skill=Tossing3DSkills.POUR_BIN) == (ROBOT, CUBE, BIN, BARRIER)
    assert _types(skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS) == (
        ROBOT,
        BIN,
        CUBE,
        BARRIER,
    )
    assert _types(skill=Tossing3DSkills.OPEN_GRIPPER) == (ROBOT, CUBE, BIN)


@pytest.mark.parametrize(
    "skill", [SameSideSkills.PICK_RIM, SameSideSkills.PICK_FLOOR, SameSideSkills.PICK_BIN]
)
def test_same_side_picks_have_no_side(*, skill: object) -> None:
    assert _types(skill=skill) == (ROBOT, CUBE, BIN, BARRIER)


def test_side_conditions_are_robot_relative_predicates() -> None:
    pick = {atom.predicate for atom in Tossing3DSkills.PICK_CUBE.preconditions}
    assert CUBE_ON_ROBOT_SIDE in pick
    pour = {atom.predicate for atom in Tossing3DSkills.POUR_BIN.preconditions}
    assert BIN_ON_ROBOT_SIDE in pour
    assert {p.name for p in (CUBE_ON_OPPOSITE_SIDE, BIN_ON_OPPOSITE_SIDE)} == {
        "CubeOnOppositeSide",
        "BinOnOppositeSide",
    }


def test_the_human_reset_is_two_skills_each_costing_five() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    resets = provider.human_cube_bin_reset_skills()
    assert [reset.skill.name for reset in resets] == [
        "ResetCubeBinSameSide",
        "ResetCubeBinOppositeSide",
    ]
    for reset in resets:
        assert _types(skill=reset.skill) == (ROBOT, CUBE, BIN, BARRIER)
    assert [provider.movables_reset_destination(ground_skill=reset) for reset in resets] == [
        "robot_side",
        "opposite_side",
    ]
    assert [provider.human_skill_name(ground_skill=reset) for reset in resets] == [
        "reset_cube_and_bin_near",
        "reset_cube_far",
    ]
    same, opposite = resets
    assert BIN_ON_ROBOT_SIDE in {atom.predicate for atom in same.skill.add_effects}
    assert BIN_ON_OPPOSITE_SIDE in {atom.predicate for atom in opposite.skill.add_effects}


def test_no_side_objects_remain_in_the_planning_vocabulary() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    assert all(obj.type in {ROBOT, CUBE, BIN, BARRIER} for obj in provider.objects())
    for skill in provider.skills():
        assert set(_types(skill=skill)) <= {ROBOT, CUBE, BIN, BARRIER}


def test_each_reset_skill_has_its_own_belief() -> None:
    names = [name for name, _ in TOSSING3D_DOMAIN.belief_sources]
    assert RESET_SAME_SIDE_SKILL == "ResetCubeBinSameSide"
    assert RESET_OPPOSITE_SIDE_SKILL == "ResetCubeBinOppositeSide"
    assert TOSSING3D_DOMAIN.belief_name(skill_name=RESET_SAME_SIDE_SKILL) == RESET_SAME_SIDE_SKILL
    assert (
        TOSSING3D_DOMAIN.belief_name(skill_name=RESET_OPPOSITE_SIDE_SKILL)
        == RESET_OPPOSITE_SIDE_SKILL
    )
    # The same-side reset keeps the old reset's prior-seed slot; the opposite-side one
    # is appended so every existing skill keeps its historical seed.
    assert names == [
        "PickCube",
        "MoveToTossLocationAndToss",
        "OpenGripper",
        RESET_SAME_SIDE_SKILL,
        "PourCubeOutOfBin",
        RESET_OPPOSITE_SIDE_SKILL,
    ]
