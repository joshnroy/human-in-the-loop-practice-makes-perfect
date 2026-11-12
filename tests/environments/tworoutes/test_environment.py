import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment


class Scripted:
    """Environments whose skills succeed always (`sure`) or never (`hopeless`)."""

    @staticmethod
    def sure(**kwargs: object) -> TwoRoutesEnvironment:
        env = TwoRoutesEnvironment(
            route_a_start=1.0, route_b_start=1.0, competence_cap=1.0, **kwargs
        )
        env.hard_reset()
        return env

    @staticmethod
    def hopeless(**kwargs: object) -> TwoRoutesEnvironment:
        env = TwoRoutesEnvironment(
            route_a_start=0.0, route_a_rate=0.0, route_b_start=0.0, route_b_rate=0.0, **kwargs
        )
        env.hard_reset()
        return env


def flags(*, env: TwoRoutesEnvironment) -> dict[str, bool]:
    state = env.get_current_state()
    return {
        name: bool(state.get(obj=env.robot, feature_name=name))
        for name in env.robot_type.feature_names
    }


def test_hard_reset_leaves_the_robot_ready_and_nothing_else() -> None:
    env = Scripted.sure()
    assert flags(env=env) == {
        "ready": True,
        "done": False,
        "via_a": False,
        "via_b": False,
        "after_a": False,
        "after_b": False,
    }


@pytest.mark.parametrize(
    ("route", "via", "after"), [("RouteA", "via_a", "after_a"), ("RouteB", "via_b", "after_b")]
)
def test_a_successful_route_finishes_and_records_which_route(
    *, route: str, via: str, after: str
) -> None:
    env = Scripted.sure()
    env.take_action(action=env.action_for(skill=route))
    result = flags(env=env)
    assert result["done"] and result[via] and result[after] and not result["ready"]
    assert env.last_error is None


@pytest.mark.parametrize(("route", "after"), [("RouteA", "after_a"), ("RouteB", "after_b")])
def test_a_failed_route_leaves_the_robot_needing_that_routes_reset(
    *, route: str, after: str
) -> None:
    env = Scripted.hopeless()
    env.take_action(action=env.action_for(skill=route))
    result = flags(env=env)
    assert result == {**dict.fromkeys(result, False), after: True}
    assert env.last_error == f"{route}: attempt failed"


def test_the_robot_resets_itself_after_route_a_but_not_after_route_b() -> None:
    env = Scripted.sure()
    env.take_action(action=env.action_for(skill="RouteA"))
    env.take_action(action=env.action_for(skill="ResetA"))
    assert flags(env=env)["ready"] and not flags(env=env)["done"]

    env.take_action(action=env.action_for(skill="RouteB"))
    before = flags(env=env)
    env.take_action(action=env.action_for(skill="ResetA"))
    assert flags(env=env) == before
    assert env.last_error == "ResetA: preconditions do not hold"


def test_only_the_human_reset_returns_to_ready_after_route_b() -> None:
    env = Scripted.sure()
    env.take_action(action=env.action_for(skill="RouteB"))
    assert env.reset_movables(destination="human_reset_b") is True
    assert flags(env=env)["ready"] and not flags(env=env)["after_b"]
    # Not after Route A: that reset is the robot's own.
    env.take_action(action=env.action_for(skill="RouteA"))
    assert env.reset_movables(destination="human_reset_b") is False
    assert env.reset_movables(destination="elsewhere") is False


def test_success_probability_follows_the_scripted_curve_and_its_cap() -> None:
    env = TwoRoutesEnvironment()
    assert env.success_probability(skill="RouteA") == pytest.approx(0.2)
    assert env.success_probability(skill="RouteB") == pytest.approx(0.1)
    env.attempts = {"RouteA": 3, "RouteB": 4}
    assert env.success_probability(skill="RouteA") == pytest.approx(0.5)
    assert env.success_probability(skill="RouteB") == pytest.approx(0.3)
    env.attempts = {"RouteA": 100, "RouteB": 100}
    assert env.success_probability(skill="RouteA") == pytest.approx(0.95)
    assert env.success_probability(skill="RouteB") == pytest.approx(0.95)


def test_practice_attempts_raise_competence_and_evaluation_attempts_do_not() -> None:
    practice = Scripted.hopeless()
    practice.take_action(action=practice.action_for(skill="RouteA"))
    assert practice.attempts == {"RouteA": 1, "RouteB": 0}

    evaluation = Scripted.hopeless(learns=False)
    evaluation.take_action(action=evaluation.action_for(skill="RouteA"))
    assert evaluation.attempts == {"RouteA": 0, "RouteB": 0}


def test_an_attempt_out_of_its_precondition_is_not_an_attempt() -> None:
    env = Scripted.hopeless()
    env.take_action(action=env.action_for(skill="RouteA"))
    env.take_action(action=env.action_for(skill="RouteA"))
    assert env.attempts["RouteA"] == 1
    assert env.last_error == "RouteA: preconditions do not hold"


def test_the_noop_changes_nothing_and_runs_no_control_step() -> None:
    env = Scripted.sure()
    steps: list[str] = []
    env.set_control_step_observers(
        before=lambda: steps.append("before"), after=lambda: steps.append("after")
    )
    before = flags(env=env)
    env.take_action(action=env.noop_action())
    assert flags(env=env) == before and steps == [] and env.attempts["RouteA"] == 0


def test_every_skill_is_one_observed_control_step() -> None:
    env = Scripted.sure()
    steps: list[str] = []
    env.set_control_step_observers(
        before=lambda: steps.append("before"), after=lambda: steps.append("after")
    )
    env.take_action(action=env.action_for(skill="RouteA"))
    env.take_action(action=env.action_for(skill="ResetA"))
    assert steps == ["before", "after", "before", "after"]


def test_a_step_refused_by_the_budget_changes_nothing() -> None:
    env = Scripted.sure()

    def refuse() -> None:
        raise ControlStepLimitReached

    env.set_control_step_observers(before=refuse, after=None)
    with pytest.raises(ControlStepLimitReached):
        env.take_action(action=env.action_for(skill="RouteA"))
    assert flags(env=env)["ready"] and env.attempts["RouteA"] == 0


def test_outcomes_are_determined_by_the_seed() -> None:
    def outcomes(*, seed: int) -> list[bool]:
        env = TwoRoutesEnvironment(seed=seed, route_a_rate=0.0, route_a_start=0.5)
        env.hard_reset()
        results = []
        for _ in range(30):
            env.take_action(action=env.action_for(skill="RouteA"))
            results.append(flags(env=env)["done"])
            env.take_action(action=env.action_for(skill="ResetA"))
        return results

    assert outcomes(seed=1) == outcomes(seed=1)
    assert outcomes(seed=1) != outcomes(seed=2)


def test_practice_logs_its_attempt_counts_per_robot_invocation(*, tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    env = Scripted.hopeless(attempts_log=log)
    env.take_action(action=env.action_for(skill="RouteA"))
    env.take_action(action=env.action_for(skill="ResetA"))
    env.take_action(action=env.action_for(skill="RouteB"))
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines == [
        {"invocation": 1, "attempts": {"RouteA": 1, "RouteB": 0}},
        {"invocation": 2, "attempts": {"RouteA": 1, "RouteB": 0}},
        {"invocation": 3, "attempts": {"RouteA": 1, "RouteB": 1}},
    ]


def test_an_evaluation_adopts_the_attempts_practice_had_made_by_a_snapshot(
    *, tmp_path: Path
) -> None:
    log = tmp_path / "attempts.jsonl"
    practice = Scripted.hopeless(attempts_log=log)
    for skill in ("RouteA", "ResetA", "RouteA", "ResetA", "RouteB"):
        practice.take_action(action=practice.action_for(skill=skill))
    # A half-written last line, as a reader beside a live writer can see.
    with log.open("a") as stream:
        stream.write('{"invocation": 6, "att')

    evaluation = TwoRoutesEnvironment(learns=False)
    evaluation.adopt_practice_attempts(log=log, robot_invocations=3)
    assert evaluation.attempts == {"RouteA": 2, "RouteB": 0}
    evaluation.adopt_practice_attempts(log=log, robot_invocations=99)
    assert evaluation.attempts == {"RouteA": 2, "RouteB": 1}
    evaluation.adopt_practice_attempts(log=log, robot_invocations=0)
    assert evaluation.attempts == {"RouteA": 0, "RouteB": 0}


def test_action_for_refuses_an_unknown_skill() -> None:
    with pytest.raises(ValueError, match="Unknown skill"):
        TwoRoutesEnvironment().action_for(skill="RouteC")


def test_the_noop_lies_inside_the_action_space_bounds() -> None:
    env = TwoRoutesEnvironment()
    noop = env.noop_action()
    assert np.all(noop >= env.action_space.low) and np.all(noop <= env.action_space.high)
