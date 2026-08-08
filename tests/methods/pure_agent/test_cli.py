"""`--method pure-agent`'s own wiring: that it registers, that it builds two separate
backends, and that it refuses the one configuration that would silently mislabel an arm.

Nothing here queries an agent. What is under test is everything that happens before the
first call, which is exactly where a mistake costs a whole run's spend.

A `tempfile.TemporaryDirectory` rather than pytest's `tmp_path` fixture, because a fixture
arrives positionally and every parameter in this project is keyword-only (ruff PLR0917)."""

import argparse
import tempfile
from pathlib import Path

import pytest

from hitl_pmp.cli import METHODS
from hitl_pmp.methods.pure_agent.cli import PureAgentCli
from hitl_pmp.methods.pure_agent.prompts import PromptArm


def build_args(*, argv=(), root):
    parser = argparse.ArgumentParser()
    PureAgentCli.add_arguments(parser=parser)
    args = parser.parse_args(["--pure-agent-sandbox-dir", str(root / "sandbox"), *argv])
    args.output_dir = root / "out"
    return args


def test_the_method_is_registered_under_its_own_name():
    assert METHODS["pure-agent"] is PureAgentCli


def test_it_shares_the_ees_protocol_flags_so_the_arms_can_share_one_x_axis():
    with tempfile.TemporaryDirectory() as root:
        args = build_args(root=Path(root))
    assert args.num_cycles == 10
    assert args.max_steps_per_interaction == 150


def test_the_two_backends_get_separate_sandboxes_on_the_same_model():
    """The split between them is the firewall. One sandbox would mean one session store,
    and `--continue` would resume whichever conversation wrote last. The shared model is
    the other half: a practice agent on a different model from the evaluation agent would
    make the arm meaningless and would be invisible in the output."""
    with tempfile.TemporaryDirectory() as root:
        args = build_args(root=Path(root))
        practice = PureAgentCli.backend(args=args, sandbox=args.pure_agent_sandbox_dir / "practice")
        evaluation = PureAgentCli.backend(
            args=args, sandbox=args.pure_agent_sandbox_dir / "evaluation"
        )
    assert practice.sandbox_dir != evaluation.sandbox_dir
    assert practice.model == evaluation.model


def test_constructing_a_backend_touches_no_filesystem_and_needs_no_credentials():
    """`--help` and a config snapshot both construct one. Neither should create a sandbox
    or reach for a token, so the underlying agent is built lazily on first query."""
    with tempfile.TemporaryDirectory() as root:
        args = build_args(root=Path(root))
        backend = PureAgentCli.backend(args=args, sandbox=args.pure_agent_sandbox_dir / "practice")
        assert not backend.sandbox_dir.exists()


def test_the_described_arm_refuses_to_run_without_a_description():
    """Otherwise the run is labelled `described` in config_snapshot.json while its prompts
    are byte-identical to the minimal arm's -- and the two would then be compared as if
    they differed."""
    with tempfile.TemporaryDirectory() as root:
        args = build_args(argv=("--pure-agent-prompt-arm", "described"), root=Path(root))
        assert args.pure_agent_prompt_arm is PromptArm.DESCRIBED
        with pytest.raises(ValueError, match="needs a non-empty"):
            PureAgentCli.run(args=args, env_cli=None)


def test_the_default_arm_is_the_one_with_no_hint_in_it():
    with tempfile.TemporaryDirectory() as root:
        args = build_args(root=Path(root))
    assert args.pure_agent_prompt_arm is PromptArm.MINIMAL


def test_docker_is_off_by_default_and_says_so_in_the_backend_id():
    with tempfile.TemporaryDirectory() as root:
        args = build_args(root=Path(root))
        backend = PureAgentCli.backend(args=args, sandbox=args.pure_agent_sandbox_dir / "practice")
    assert backend.use_docker is False
    assert backend.describe().endswith("-host")
