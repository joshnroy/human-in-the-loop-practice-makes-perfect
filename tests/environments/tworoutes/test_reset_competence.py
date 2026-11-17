"""Either reset can fail (Josh, 2026-11-17): a failed robot reset or human reset leaves
the robot where it was, in `after`, so the method must try again. With competence 1, the
default, both always work, as before."""

import pytest

from hitl_pmp.environments.tworoutes.environment import (
    HUMAN_RESET_B,
    ROBOT_RESET,
    ROUTE_A,
    TwoRoutesEnvironment,
)


def _after_a_try(**kwargs: float) -> TwoRoutesEnvironment:
    env = TwoRoutesEnvironment(both_resets=True, chained_route_b=True, **kwargs)
    env.hard_reset()
    env.take_action(action=env.action_for(skill=ROUTE_A))
    assert env.flag(name="after")
    return env


@pytest.mark.parametrize("competence", [0.0, 1.0])
def test_a_robot_reset_works_with_its_competence(*, competence: float) -> None:
    env = _after_a_try(robot_reset_competence=competence)
    env.take_action(action=env.action_for(skill=ROBOT_RESET))
    assert env.flag(name="ready") is (competence == 1.0)
    assert env.flag(name="after") is (competence == 0.0)
    assert (env.last_error is None) is (competence == 1.0)


@pytest.mark.parametrize("competence", [0.0, 1.0])
def test_a_human_reset_works_with_its_competence(*, competence: float) -> None:
    env = _after_a_try(human_reset_competence=competence)
    # The human always takes the request; it is the reset that may not work.
    assert env.reset_movables(destination=HUMAN_RESET_B)
    assert env.flag(name="ready") is (competence == 1.0)
    assert env.flag(name="after") is (competence == 0.0)


def test_reset_competence_is_drawn_from_the_seeded_stream() -> None:
    outcomes = []
    for _ in range(2):
        env = _after_a_try(robot_reset_competence=0.5, seed=3)
        tried = []
        for _ in range(20):
            env.take_action(action=env.action_for(skill=ROBOT_RESET))
            tried.append(env.flag(name="ready"))
            if env.flag(name="ready"):
                env.take_action(action=env.action_for(skill=ROUTE_A))
        outcomes.append(tried)
    assert outcomes[0] == outcomes[1] and 0 < sum(outcomes[0]) < 20
