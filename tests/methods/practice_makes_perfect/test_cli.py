import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.methods.practice_makes_perfect.cli import (
    EesCli,
    RandomSkillsCli,
    Tossing3DPomdpCli,
)


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


def test_add_arguments_registers_only_the_shared_practice_protocol_flags() -> None:
    """RandomSkillsMethod's own RNG reuses the global --seed, so it has no
    hyperparameters of its own -- but it still exposes the two protocol flags
    (shared with --method ees via PracticeCycleCli) so both can be run over the
    same transition budget and charted on one axis."""
    before = argparse.ArgumentParser()
    after = argparse.ArgumentParser()
    RandomSkillsCli.add_arguments(parser=after)
    added = [
        action.dest
        for action in after._actions
        if action.dest not in {a.dest for a in before._actions}
    ]
    assert added == ["num_cycles", "max_steps_per_interaction"]


def test_random_skills_defaults_to_no_practice_cycles() -> None:
    """This baseline never learns, so one evaluation sweep tells you everything --
    the flag exists purely to make an equal-budget comparison possible."""
    args = _build_parser().parse_args([])
    assert args.num_cycles == 0


def test_run_prints_a_parseable_success_rate(*, capsys: pytest.CaptureFixture[str]) -> None:
    """Unlike the oracle, this baseline has no guaranteed 100% solve rate -- just
    confirm run_method wiring actually completes and prints a well-formed
    success-rate line for the requested --num-test-tasks."""
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--grid-size", "5", "--seed", "0"])
    RandomSkillsCli.run(args=args, env_cli=LightSwitchCli)
    out = capsys.readouterr().out
    assert re.search(r"success rate: \d+/3 \(\d+%\)", out)


def test_run_applies_seed_deterministically(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--grid-size", "5", "--seed", "42"])
    RandomSkillsCli.run(args=args, env_cli=LightSwitchCli)
    first = capsys.readouterr().out

    args = _build_parser().parse_args(["--num-test-tasks", "3", "--grid-size", "5", "--seed", "42"])
    RandomSkillsCli.run(args=args, env_cli=LightSwitchCli)
    second = capsys.readouterr().out

    assert first == second


def _build_ees_parser() -> argparse.ArgumentParser:
    from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    LightSwitchCli.add_arguments(parser=parser)
    EesCli.add_arguments(parser=parser)
    return parser


def test_ees_defaults_match_the_papers_light_switch_protocol() -> None:
    """The paper states 150 steps per free period and epsilon-greedy 0.5 for
    Light Switch; 10 cycles is predicators' own num_online_learning_cycles
    default (the paper never states its free-period count)."""
    args = _build_ees_parser().parse_args([])
    assert args.num_cycles == 10
    assert args.max_steps_per_interaction == 150
    assert args.exploration_epsilon == 0.5


def test_ees_run_completes_end_to_end_through_the_cli(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI *wiring* only: that --method ees parses its flags, builds an EesMethod,
    and drives real practice cycles to completion. The actual learning claim is
    asserted in test_ees_method.py, which can see the returned Metrics; this can
    only see stdout. Small grid/cycle counts to stay fast -- the full protocol
    lives in the experiment log, not the test suite."""
    args = _build_ees_parser().parse_args([
        "--num-test-tasks",
        "5",
        "--grid-size",
        "5",
        "--seed",
        "0",
        "--num-cycles",
        "4",
        "--max-steps-per-interaction",
        "40",
        "--sampler-max-train-iters",
        "300",
    ])
    EesCli.run(args=args, env_cli=LightSwitchCli)
    assert re.search(r"success rate: \d+/5", capsys.readouterr().out)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], True), (["--ees-reset-gate"], True), (["--no-ees-reset-gate"], False)],
)
def test_the_ees_reset_gate_flag_parses_and_reaches_ees(
    *, argv: list[str], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parsed by --method ees and handed to the EesMethod the factory builds, so
    --no-ees-reset-gate is not silently dropped between argparse and plan_to."""
    from types import SimpleNamespace

    from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    args = _build_ees_parser().parse_args(argv)
    assert args.ees_reset_gate is expected
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(LightSwitchCli, "run_method", staticmethod(_capture))
    EesCli.run(args=args, env_cli=LightSwitchCli)
    env = LightSwitchEnvironment(grid_size=3)
    method = captured["method_factory"](  # type: ignore[operator]
        SimpleNamespace(env=env, skill_provider=LightSwitchSkillProvider(env=env))
    )
    assert isinstance(method, EesMethod)
    assert method.reset_cost_gate is expected


def test_ees_does_not_register_domain_owned_reset_cost() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert not hasattr(args, "human_reset_practice_cost")


def test_ees_rejects_the_retired_method_level_reset_cost_flag() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--ask-for-reset-cube-bin-cost", "0.134"])


def test_pomdp_does_not_register_domain_owned_reset_cost() -> None:
    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert not hasattr(args, "human_reset_practice_cost")
    assert args.pomdp_linear_cost_lambda is None


def test_pomdp_accepts_linear_cost_coefficient() -> None:
    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    args = parser.parse_args(["--pomdp-linear-cost-lambda", "0.03"])
    assert args.pomdp_linear_cost_lambda == pytest.approx(0.03)


def test_random_skills_does_not_register_the_reset_skill_cost_flag() -> None:
    """RandomSkillsMethod has no planner to offer the ground skill to -- offering
    the flag would be a lie in --help. Not in the experiment grid."""
    parser = argparse.ArgumentParser()
    RandomSkillsCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert not hasattr(args, "ask_for_reset_cube_bin_cost")


def test_non_tossing3d_cli_rejects_the_domain_owned_reset_cost() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=[
                "--env",
                "lightswitch",
                "--method",
                "ees",
                "--human-reset-practice-cost",
                "0.1",
            ]
        )


@pytest.mark.parametrize(
    ("flag", "field"),
    [
        ("reproduce-predicators-seen-task-order", "reproduce_predicators_seen_task_order"),
        ("reproduce-predicators-skip-perfect", "reproduce_predicators_skip_perfect"),
        (
            "reproduce-predicators-explore-target-only",
            "reproduce_predicators_explore_target_only",
        ),
        (
            "reproduce-predicators-random-when-stranded",
            "reproduce_predicators_random_when_stranded",
        ),
    ],
)
@pytest.mark.parametrize(("prefix", "expected"), [(None, False), ("", True), ("no-", False)])
def test_the_predicators_fidelity_flags_parse_and_reach_ees(
    *, flag: str, field: str, prefix: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each flag defaults off, so every earlier run keeps its behaviour, and is handed
    to the EesMethod the factory builds rather than dropped between argparse and the
    method."""
    from types import SimpleNamespace

    from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    argv = [] if prefix is None else [f"--{prefix}{flag}"]
    args = _build_ees_parser().parse_args(argv)
    assert getattr(args, field) is expected
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(LightSwitchCli, "run_method", staticmethod(_capture))
    EesCli.run(args=args, env_cli=LightSwitchCli)
    env = LightSwitchEnvironment(grid_size=3)
    method = captured["method_factory"](  # type: ignore[operator]
        SimpleNamespace(env=env, skill_provider=LightSwitchSkillProvider(env=env))
    )
    assert isinstance(method, EesMethod)
    assert getattr(method, field) is expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], False), (["--defer-rendering"], True), (["--no-defer-rendering"], False)],
)
def test_ees_on_tossing3d_honours_the_defer_rendering_flag(
    *, argv: list[str], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EES renders in-run by default, as it always has; --defer-rendering reaches
    Tossing3DCli.run_method, which then builds no renderer. POMDP forces it on
    regardless, and is not this flag's business."""
    from hitl_pmp.cli import Cli
    from hitl_pmp.environments.tossing3d.cli import Tossing3DCli

    seen: list[bool] = []

    def _run_method(*, args: argparse.Namespace, **_kwargs: object) -> None:
        seen.append(args.defer_rendering)

    monkeypatch.setattr(Tossing3DCli, "run_method", staticmethod(_run_method))
    Cli.main(argv=["--env", "tossing3d", "--method", "ees", *argv])
    assert seen == [expected]
