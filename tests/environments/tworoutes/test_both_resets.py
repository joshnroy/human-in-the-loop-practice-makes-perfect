"""`both_resets`: after either route the robot waits in `After`, and either the robot's
reset (taking `robot_reset_steps` steps) or the human's puts it back."""

from hitl_pmp.environments.tworoutes.environment import (
    HUMAN_RESET_B,
    ROBOT_RESET,
    ROUTE_A,
    ROUTE_B,
    TwoRoutesEnvironment,
)
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider


def make(**kwargs: object) -> TwoRoutesEnvironment:
    env = TwoRoutesEnvironment(both_resets=True, **kwargs)
    env.hard_reset()
    return env


def test_either_route_leaves_the_robot_waiting_in_after() -> None:
    for route in (ROUTE_A, ROUTE_B):
        env = make()
        env.take_action(action=env.action_for(skill=route))
        assert env.flag(name="after") and not env.flag(name="ready")


def test_the_robot_reset_works_after_either_route_and_takes_its_steps() -> None:
    for route in (ROUTE_A, ROUTE_B):
        env = make(robot_reset_steps=3)
        steps: list[int] = []
        env.set_control_step_observers(before=lambda steps=steps: steps.append(1), after=None)
        env.take_action(action=env.action_for(skill=route))
        env.take_action(action=env.action_for(skill=ROBOT_RESET))
        assert env.flag(name="ready")
        assert len(steps) == 1 + 3


def test_the_human_reset_works_after_either_route() -> None:
    for route in (ROUTE_A, ROUTE_B):
        env = make()
        env.take_action(action=env.action_for(skill=route))
        assert env.reset_movables(destination=HUMAN_RESET_B)
        assert env.flag(name="ready")


def test_the_provider_offers_both_resets_from_after() -> None:
    env = make()
    provider = TwoRoutesSkillProvider(env=env, both_resets=True)
    names = {s.name for s in provider.skills()}
    assert ROBOT_RESET in names
    robot_reset = next(s for s in provider.skills() if s.name == ROBOT_RESET)
    human = provider.human_cube_bin_reset_skill()
    assert human is not None
    for skill in (robot_reset, human.skill):
        assert {a.predicate.name for a in skill.preconditions} == {"After"}
