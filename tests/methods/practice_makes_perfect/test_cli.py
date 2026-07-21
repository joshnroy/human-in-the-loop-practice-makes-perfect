import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.methods.practice_makes_perfect.cli import RandomSkillsCli


@pytest.fixture(autouse=True)
def _restore_lightswitch_config() -> Iterator[None]:
    """RandomSkillsCli.run() delegates to LightSwitchCli.run_method(), which still
    mutates LightSwitchEnvironment.light_on_tolerance/.same_position_tolerance as a
    ClassVar side effect via apply_config (see that method's own docstring for why
    those two specifically stay ClassVar rather than becoming constructor
    arguments). Everything else run_method touches -- env/tasks/problem/method -- is
    now a freshly constructed instance per call, with nothing left over to restore."""
    original_light_on_tolerance = LightSwitchEnvironment.light_on_tolerance
    original_same_position_tolerance = LightSwitchEnvironment.same_position_tolerance
    try:
        yield
    finally:
        LightSwitchEnvironment.light_on_tolerance = original_light_on_tolerance
        LightSwitchEnvironment.same_position_tolerance = original_same_position_tolerance


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global flags plus --env lightswitch's own, so
    RandomSkillsCli can be exercised the same way the real global CLI drives it."""
    from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    LightSwitchCli.add_arguments(parser=parser)
    RandomSkillsCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_adds_no_flags_of_its_own() -> None:
    """RandomSkillsMethod's own RNG reuses the global --seed, so this is a
    no-op, unlike a future method-CLI with real hyperparameters."""
    before = argparse.ArgumentParser()
    after = argparse.ArgumentParser()
    RandomSkillsCli.add_arguments(parser=after)
    assert [action.dest for action in after._actions] == [action.dest for action in before._actions]


def test_run_prints_a_parseable_success_rate(*, capsys: pytest.CaptureFixture[str]) -> None:
    """Unlike the oracle, this baseline has no guaranteed 100% solve rate -- just
    confirm run_method wiring actually completes and prints a well-formed
    success-rate line for the requested --num-test-tasks."""
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--grid-size", "5", "--seed", "0"])
    RandomSkillsCli.run(args=args)
    out = capsys.readouterr().out
    assert re.search(r"success rate: \d+/3 \(\d+%\)", out)


def test_run_applies_seed_deterministically(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--grid-size", "5", "--seed", "42"])
    RandomSkillsCli.run(args=args)
    first = capsys.readouterr().out

    args = _build_parser().parse_args(["--num-test-tasks", "3", "--grid-size", "5", "--seed", "42"])
    RandomSkillsCli.run(args=args)
    second = capsys.readouterr().out

    assert first == second
