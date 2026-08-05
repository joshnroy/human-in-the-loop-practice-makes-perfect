"""Offline tests for `Tossing3DCli`'s flags and for its registration in the global CLI.

`run_method` drives a real simulator, so it lives in `test_kinder_fidelity.py`.
"""

import argparse
from pathlib import Path

import pytest

from hitl_pmp.cli import ENVIRONMENTS, Cli
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment, Tossing3DTaskConfig
from hitl_pmp.environments.tossing3d.skill_oracle_policy import ORACLE_THROW_STANDOFF
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    Tossing3DCli.add_arguments(parser=parser)
    return parser


def test_the_environment_is_registered_under_tossing3d() -> None:
    assert ENVIRONMENTS["tossing3d"] is Tossing3DCli


def test_registering_it_does_not_pull_a_simulator_into_the_global_cli() -> None:
    """CI never installs the optional extra, and `hitl_pmp.cli` imports every registered
    environment's CLI. If this domain's import chain reached MuJoCo, `--env lightswitch`
    would stop working on a machine without it."""
    import sys

    Cli.parse_args(argv=["--env", "lightswitch", "--method", "skill-oracle"])
    assert "mujoco" not in sys.modules


def test_the_defaults_are_read_off_the_models_rather_than_re_literalled() -> None:
    args = _build_parser().parse_args([])
    fields = Tossing3DEnvironment.model_fields
    assert args.task_config == fields["task_config"].default.value
    assert args.variant == fields["variant"].default
    assert args.canonical_seed == fields["canonical_seed"].default
    assert args.scene_bg is True
    assert args.test_env_seed_offset == Tossing3DTasks.model_fields["test_env_seed_offset"].default
    assert args.oracle_throw_standoff == ORACLE_THROW_STANDOFF


def test_the_default_scene_is_the_coincident_one() -> None:
    """The default matters more here than defaults usually do: under stock, a cube that
    lands in the bin is a scored FAILURE, so a run that defaulted to stock would be
    rewarding the throw for missing."""
    args = _build_parser().parse_args([])
    assert Tossing3DTaskConfig(args.task_config) is Tossing3DTaskConfig.COINCIDENT


def test_stock_stays_selectable() -> None:
    args = _build_parser().parse_args(["--task-config", "stock"])
    assert Tossing3DTaskConfig(args.task_config) is Tossing3DTaskConfig.STOCK


def test_an_unknown_task_config_is_rejected_rather_than_falling_back_to_stock() -> None:
    """Silently falling back would produce a stock run labelled as the coincident one --
    the exact class of mistake this domain's default exists to remove."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--task-config", "made-up"])


def test_the_oracle_standoff_is_overridable_for_the_stock_scene() -> None:
    """1.35 solves the coincident scene and fails on stock, where the bin sits 23 cm
    further out; 1.55 is the measured value that solves stock."""
    args = _build_parser().parse_args(["--oracle-throw-standoff", "1.55"])
    assert args.oracle_throw_standoff == pytest.approx(1.55)


def test_the_global_cli_registers_this_domains_flags_when_env_is_tossing3d() -> None:
    args = Cli.parse_args(
        argv=["--env", "tossing3d", "--method", "skill-oracle", "--task-config", "stock"]
    )
    assert args.task_config == "stock"
    assert args.num_test_tasks == 10
