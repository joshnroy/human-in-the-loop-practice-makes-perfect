"""Resolve pure kitchen metadata before resume validation touches a simulator."""

import json
from types import SimpleNamespace

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.step_protocol import KitchenRecyclingStepProtocol
from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, StepProtocolEntry
from tests.practice_cost_fixtures import CostFixtures


def test_resume_resolves_identical_kitchen_json_and_rejects_changed_file_before_build(
    *, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    output.mkdir()
    config = tmp_path / "kitchen.json"
    config.write_text(KitchenConfig(scatter_objects=True).model_dump_json())
    argv = [
        "--env",
        "kitchen_recycling3d",
        "--method",
        "ees",
        "--practice-step-budget",
        "100",
        "--practice-cost-config",
        str(CostFixtures.UNIT),
        "--kitchen-config",
        str(config),
        "--output-dir",
        str(output),
        "--checkpoint-cycles",
        "--defer-rendering",
        "--practice-reset-policy",
        "never",
    ]
    first = Cli.parse_args(argv=argv)
    KitchenRecyclingCli.config(args=first)
    (output / "config_snapshot.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(first).items()}})
    )
    (output / CHECKPOINT_FILENAME).write_bytes(b"not opened until validation passes")
    received = []

    def dispatch(*, args, env_cli):
        received.append(args.kitchen_config_resolved)

    def forbidden(**kwargs):
        raise AssertionError(
            "resume configuration validation must not build a physical environment"
        )

    monkeypatch.setattr(KitchenRecyclingCli, "build_problem", staticmethod(forbidden))
    kwargs = {
        "method_cli": SimpleNamespace(run=dispatch),
        "env_cli": KitchenRecyclingCli,
        "environment": KitchenRecyclingStepProtocol,
    }
    resumed = Cli.parse_args(argv=[*argv, "--resume", str(output)])
    StepProtocolEntry.run(args=resumed, **kwargs)
    assert received == [first.kitchen_config_resolved]
    config.write_text(KitchenConfig(scatter_objects=False).model_dump_json())
    changed = Cli.parse_args(argv=[*argv, "--resume", str(output)])
    with pytest.raises(ValueError, match="kitchen_config_resolved"):
        StepProtocolEntry.run(args=changed, **kwargs)
    assert received == [first.kitchen_config_resolved]


class _WorldReached(Exception):
    """The method's CLI got as far as asking for a simulator."""


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(("--method", "ees"), id="ees"),
        pytest.param(("--method", "dp", "--dp-budget", "4"), id="dp"),
        pytest.param(
            ("--method", "pomdp", "--pomdp-solver", "determinized_astar")
            + ("--pomdp-determinized-variant", "curtis"),
            id="curtis",
        ),
        pytest.param(("--method", "pomdp", "--pomdp-solver", "expectimax"), id="expectimax"),
    ],
)
def test_a_run_resumes_under_exactly_the_flags_it_was_started_with(
    *, tmp_path, monkeypatch, method
):
    """A method's CLI settles some flags itself (EES always scores human skills, a
    planner run never renders, expectimax takes its domain's depth), and the snapshot
    records them settled. The resume check must settle them the same way before it
    compares, or the run's own command is refused for a flag nobody typed."""
    from hitl_pmp.cli import METHODS
    from hitl_pmp.config_snapshot import ConfigSnapshot

    output = tmp_path / "run"
    output.mkdir()
    argv = [
        "--env",
        "kitchen_recycling3d",
        *method,
        "--practice-step-budget",
        "100",
        "--practice-cost-config",
        str(CostFixtures.UNIT),
        "--output-dir",
        str(output),
        "--checkpoint-cycles",
        "--defer-rendering",
        "--practice-reset-policy",
        "never",
    ]

    def no_world(**kwargs):
        raise _WorldReached

    monkeypatch.setattr(KitchenRecyclingCli, "build_problem", staticmethod(no_world))

    def start(*, arguments):
        args = Cli.parse_args(argv=arguments)
        with pytest.raises(_WorldReached):
            StepProtocolEntry.run(
                args=args,
                method_cli=METHODS[args.method],
                env_cli=KitchenRecyclingCli,
                environment=KitchenRecyclingStepProtocol,
            )
        return args

    # What the run records once its method CLI has run, as `StepPracticeRunner` does.
    recorded = ConfigSnapshot.collect(args=start(arguments=argv), fd_exec_path=None)
    (output / "config_snapshot.json").write_text(recorded.model_dump_json())
    (output / CHECKPOINT_FILENAME).write_bytes(b"not opened until a world exists")

    start(arguments=[*argv, "--resume", str(output)])
