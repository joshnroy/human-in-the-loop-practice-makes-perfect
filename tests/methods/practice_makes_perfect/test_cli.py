import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.human_intervention import HumanResetTarget
from hitl_pmp.methods.help_seeking import HelpSeekingTrigger
from hitl_pmp.methods.practice_makes_perfect.cli import EesCli, HelpSeekingCli, RandomSkillsCli
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


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


def test_ees_sampler_classifier_defaults_to_mlp() -> None:
    args = _build_ees_parser().parse_args([])
    assert args.sampler_classifier == "mlp"


def _capture_ees_method(*, args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch) -> EesMethod:
    """Runs EesCli.run() far enough to build the real EesMethod through
    LightSwitchCli's composition root, then intercepts MethodRunner.run (called
    immediately after method_factory(context)) to capture it before any practice
    cycle actually executes -- cheaper and more direct than asserting through
    stdout, and it is the one place a fully-wired EesMethod instance is reachable
    from a CLI-level test."""
    from hitl_pmp.method_runner import MethodRunner

    captured: dict[str, EesMethod] = {}

    def _fake_run(*, method: EesMethod, **_kwargs: object) -> None:
        captured["method"] = method

    monkeypatch.setattr(MethodRunner, "run", staticmethod(_fake_run))
    EesCli.run(args=args, env_cli=LightSwitchCli)
    return captured["method"]


def test_ees_sampler_classifier_flag_reaches_the_built_method(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--sampler-classifier linear must reach EesMethod.sampler_classifier on the
    Method the CLI actually constructs, not just get parsed and dropped."""
    args = _build_ees_parser().parse_args([
        "--num-test-tasks",
        "5",
        "--grid-size",
        "5",
        "--seed",
        "0",
        "--sampler-classifier",
        "linear",
    ])
    method = _capture_ees_method(args=args, monkeypatch=monkeypatch)
    assert method.sampler_classifier == "linear"


def test_ees_sampler_classifier_omitted_flag_defaults_to_mlp_on_the_built_method(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _build_ees_parser().parse_args([
        "--num-test-tasks",
        "5",
        "--grid-size",
        "5",
        "--seed",
        "0",
    ])
    method = _capture_ees_method(args=args, monkeypatch=monkeypatch)
    assert method.sampler_classifier == "mlp"


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


def test_the_ask_for_help_default_builds_no_policy_at_all() -> None:
    """`None`, not "a policy configured off". That is the structural claim behind
    --ask-for-help never being byte-identical: an unconfigured run holds no detector and
    draws no randomness, and EesMethod returns its practice policy unwrapped."""
    args = argparse.Namespace(seed=0)
    assert HelpSeekingCli.build_policy(args=args) is None


def test_an_omitted_flag_falls_back_to_never() -> None:
    """A hand-built Namespace -- a test, a one-off script -- predates every one of these
    flags, so the absent case has to mean the incumbent behaviour."""
    assert HelpSeekingCli.build_policy(args=argparse.Namespace()) is None


def test_the_asking_flags_reach_the_policy() -> None:
    args = argparse.Namespace(
        ask_for_help=HelpSeekingTrigger.AT_RANDOM,
        stuck_patience=7,
        mean_steps_between_help_requests=40,
        seed=3,
    )
    policy = HelpSeekingCli.build_policy(args=args)
    assert policy is not None
    assert policy.trigger is HelpSeekingTrigger.AT_RANDOM
    assert policy.stuck_patience == 7
    assert policy.mean_steps_between_requests == 40
    assert policy.seed == 3


def test_the_policy_is_seeded_from_the_global_seed() -> None:
    """So a sweep's seeds do not all ask on identical steps -- the BallRing
    --noise-seed trap, where a constant default made every arm identical."""
    seeds = set()
    for seed in range(4):
        policy = HelpSeekingCli.build_policy(
            args=argparse.Namespace(ask_for_help=HelpSeekingTrigger.AT_RANDOM, seed=seed)
        )
        assert policy is not None
        seeds.add(policy.seed)
    assert seeds == {0, 1, 2, 3}


def test_ees_registers_the_asking_flags_with_never_as_the_default() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert args.ask_for_help is HelpSeekingTrigger.NEVER
    assert args.stuck_patience == 20
    assert args.mean_steps_between_help_requests == 150


def test_ees_accepts_every_asking_mode() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    args = parser.parse_args([
        "--ask-for-help",
        "on-stuck",
        "--stuck-patience",
        "5",
        "--mean-steps-between-help-requests",
        "30",
    ])
    assert args.ask_for_help is HelpSeekingTrigger.ON_STUCK
    assert args.stuck_patience == 5
    assert args.mean_steps_between_help_requests == 30


def test_ees_rejects_an_unknown_asking_mode() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--ask-for-help", "sometimes"])


def test_ees_rejects_a_non_positive_stuck_patience() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--stuck-patience", "0"])


def test_random_skills_does_not_register_the_asking_flags() -> None:
    """RandomSkillsMethod does not compose HelpSeekingMixin, so offering it a flag it
    cannot honour would be a lie in --help. It is not in the experiment grid."""
    parser = argparse.ArgumentParser()
    RandomSkillsCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert not hasattr(args, "ask_for_help")


def test_an_asking_ees_run_on_a_domain_with_no_human_fails_before_it_starts() -> None:
    """End to end through the real CLI: the flag builds a policy, EesMethod therefore
    declares it may ask, and PracticeLoop refuses up front because Light Switch wires no
    HumanOracle. Fails at construction rather than a cycle in, which is the whole point
    of validating on the Method's declaration instead of on a per-step poll."""
    args = Cli.parse_args(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "ees",
            "--ask-for-help",
            "on-stuck",
            "--stuck-patience",
            "1",
            "--num-cycles",
            "1",
            "--max-steps-per-interaction",
            "4",
            "--num-test-tasks",
            "1",
        ]
    )
    assert args.human_reset_target is HumanResetTarget.TASK_INITIAL
    with pytest.raises(ValueError, match="HumanOracle"):
        EesCli.run(args=args, env_cli=LightSwitchCli)
