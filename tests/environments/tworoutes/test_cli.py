import argparse

import pytest

from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.environments.tworoutes.cli import TwoRoutesCli
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.step_protocol import TwoRoutesStepProtocol


def parse(*argv: str) -> argparse.Namespace:  # noqa: PLR0917 -- argv tokens
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    TwoRoutesCli.add_arguments(parser=parser)
    return parser.parse_args(list(argv))


def test_the_domain_is_registered_for_the_step_protocol() -> None:
    assert ENVIRONMENTS["tworoutes"] is TwoRoutesCli
    assert STEP_PROTOCOL_ENVIRONMENTS["tworoutes"] is TwoRoutesStepProtocol


def test_flag_defaults_are_the_environments_own() -> None:
    args = parse()
    fields = TwoRoutesEnvironment.model_fields
    for name in (
        "route_a_start",
        "route_a_rate",
        "route_b_start",
        "route_b_rate",
        "competence_cap",
        "route_a_reward",
        "route_b_reward",
    ):
        assert getattr(args, name) == fields[name].default
    assert args.human_reset is True


def test_practice_learns_and_evaluation_does_not_and_their_streams_differ() -> None:
    args = parse("--route-b-rate", "0.02", "--route-b-reward", "5", "--seed", "3")
    practice = TwoRoutesCli.build_problem(args=args, practice=True).env
    evaluation = TwoRoutesCli.build_problem(args=args, practice=False).env
    assert (practice.learns, evaluation.learns) == (True, False)
    assert (practice.seed, evaluation.seed) == (3, 10003)
    assert practice.route_b_rate == 0.02 and practice.route_b_reward == 5.0
    assert practice.attempts_log is None  # no --output-dir, so nowhere to log


def test_no_human_reset_withholds_the_human_skill() -> None:
    args = parse("--no-human-reset")
    problem = TwoRoutesCli.build_problem(args=args, practice=True)
    context = TwoRoutesCli.context(args=args, problem=problem)
    assert context.skill_provider.human_cube_bin_reset_skills() == ()
    assert [goal.describe() for goal in context.test_task_goals] == ["Done(robot)"]


def test_the_domain_runs_only_under_the_step_protocol() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "tworoutes", "--method", "ees"])
    with pytest.raises(ValueError, match="only under --practice-step-budget"):
        TwoRoutesCli.run_method(
            args=parse(), method_factory=lambda ctx: None, num_cycles=0, max_steps_per_interaction=0
        )
