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


def test_each_route_and_outcome_ends_in_its_own_state() -> None:
    """Josh, 2026-11-16: the reward is tied to the state, so the four ends differ --
    Done via A, Failed A, Done via B, Failed B -- and every one waits in After."""
    ends = {}
    for route in (ROUTE_A, ROUTE_B):
        for succeed in (True, False):
            env = make(
                route_a_start=float(succeed), route_a_rate=0.0,
                route_b_start=float(succeed), route_b_rate=0.0,
            )  # fmt: skip
            env.take_action(action=env.action_for(skill=route))
            names = ("ready", "done", "via_a", "via_b", "after_a", "after_b", "after")
            ends[route, succeed] = frozenset(n for n in names if env.flag(name=n))
    assert ends[ROUTE_A, True] == {"after", "after_a", "done", "via_a"}
    assert ends[ROUTE_A, False] == {"after", "after_a"}
    assert ends[ROUTE_B, True] == {"after", "after_b", "done", "via_b"}
    assert ends[ROUTE_B, False] == {"after", "after_b"}


def test_the_skills_declare_the_states_the_environment_reaches() -> None:
    provider = TwoRoutesSkillProvider(env=make(), both_resets=True)
    by_name = {s.name: s for s in (*provider.skills(), provider.human_skill())}
    for route, side in ((ROUTE_A, "AfterA"), (ROUTE_B, "AfterB")):
        added = {a.predicate.name for a in by_name[route].add_effects}
        assert {"After", side} <= added
    for reset in (ROBOT_RESET, HUMAN_RESET_B):
        deleted = {a.predicate.name for a in by_name[reset].delete_effects}
        assert {"After", "AfterA", "AfterB", "Done", "ViaA", "ViaB"} <= deleted


def test_without_the_robot_reset_only_the_human_returns_to_ready() -> None:
    """Josh, 2026-11-16: the toy as drawn has one reset, the human's."""
    env = make(robot_reset=False)
    names = {int(round(float(a[0]))) for a in env.get_valid_actions()}
    assert env.action_for(skill=ROBOT_RESET)[0] not in names
    provider = TwoRoutesSkillProvider(env=env, both_resets=True, robot_reset=False)
    assert ROBOT_RESET not in {s.name for s in provider.skills()}
    env.take_action(action=env.action_for(skill=ROUTE_A))
    env.take_action(action=env.action_for(skill=ROBOT_RESET))
    assert not env.flag(name="ready") and env.last_error is not None
    assert env.reset_movables(destination=HUMAN_RESET_B)
    assert env.flag(name="ready")
