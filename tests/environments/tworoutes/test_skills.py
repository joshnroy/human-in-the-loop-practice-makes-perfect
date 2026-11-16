from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.skills import TwoRoutesSkills


def names(atoms: object) -> set[str]:  # noqa: PLR0917 -- a one-argument test helper
    return {atom.predicate.name for atom in atoms}  # type: ignore[attr-defined]


def ground(*, name: str) -> GroundSkill:
    skill = next(s for s in TwoRoutesSkills.all() if s.name == name)
    return GroundSkill(skill=skill, objects=(TwoRoutesEnvironment.robot,))


def test_both_routes_start_ready_and_reach_done_marked_with_their_route() -> None:
    for route, via, after in (("RouteA", "ViaA", "AfterA"), ("RouteB", "ViaB", "AfterB")):
        skill = ground(name=route)
        assert names(skill.preconditions) == {"Ready"}
        assert names(skill.add_effects) == {"Done", via, after}
        assert names(skill.delete_effects) == {"Ready"}
        assert skill.skill.param_dim == 0


def test_the_robot_reset_follows_route_a_and_the_human_reset_follows_route_b() -> None:
    robot = ground(name="ResetA")
    assert names(robot.preconditions) == {"AfterA"}
    assert names(robot.add_effects) == {"Ready"}
    assert names(robot.delete_effects) == {"AfterA", "Done", "ViaA"}

    human = GroundSkill(skill=TwoRoutesSkills.human_reset(), objects=(TwoRoutesEnvironment.robot,))
    assert human.skill.name == "human_reset_b"
    assert names(human.preconditions) == {"AfterB"}
    assert names(human.add_effects) == {"Ready"}
    assert names(human.delete_effects) == {"AfterB", "Done", "ViaB"}


def test_the_human_reset_is_not_a_robot_skill() -> None:
    assert [s.name for s in TwoRoutesSkills.all()] == ["RouteA", "RouteB", "ResetA"]
