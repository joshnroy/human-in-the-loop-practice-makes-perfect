"""Parse the documented pending commands; never launch a simulator or planner."""

import re
import shlex
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli
from hitl_pmp.step_protocol import StepProtocolValidation

DOCUMENT = Path(__file__).resolve().parents[3] / "docs/experiments/rack-cli-preflight.md"


def _flags(*, name):
    match = re.search(rf"^{name}=\((.*?)^\)", DOCUMENT.read_text(), flags=re.MULTILINE | re.DOTALL)
    assert match is not None
    return shlex.split(match.group(1))


@pytest.mark.parametrize("solver", [None, "determinized_astar", "expectimax"])
@pytest.mark.parametrize("step_budget", [False, True])
def test_documented_preflight_arguments_parse_and_keep_intended_bounds(
    *, tmp_path, solver, step_budget
):
    argv = [*_flags(name="COMMON"), "--output-dir", str(tmp_path)]
    argv += _flags(name="STEP_FLAGS") if step_budget else ["--num-cycles", "1"]
    argv += (
        ["--method", "ees"]
        if solver is None
        else [*_flags(name="POMDP_FLAGS"), "--pomdp-solver", solver]
    )
    if solver == "expectimax":
        argv += ["--pomdp-search-depth", "1"]
    args = Cli.parse_args(argv=argv)
    assert args.env == "racktoserving3d"
    assert args.max_steps_per_interaction == 4
    assert args.goal_pursuit_horizon == 0
    assert args.skill_step_limit == 2000
    assert not hasattr(args, "cooling_seconds")
    assert not hasattr(args, "dumpling_width")
    assert args.defer_rendering
    assert args.num_test_tasks == 1
    if step_budget:
        StepProtocolValidation.check(args=args)
        assert args.practice_step_budget == args.evaluation_control_steps == 4000
        assert args.measurement_interval_steps == 4000
        assert not args.initial_measurement
        assert "--num-cycles" not in argv
    else:
        assert args.practice_step_budget is None
        assert args.num_cycles == 1
    if solver is not None:
        assert args.pomdp_solver == solver
        assert args.pomdp_search_depth == (1 if solver == "expectimax" else None)
        assert args.pomdp_max_search_iterations == 32
        assert args.pomdp_inference_engine == "grid"
        assert args.pomdp_competence_model == "local_trend"
        assert args.pomdp_num_particles == 1024
        assert args.pomdp_num_samples == 4


@pytest.mark.parametrize("solver", ["determinized_astar", "expectimax"])
def test_pomdp_commands_route_to_rack_without_launching(*, tmp_path, monkeypatch, solver):
    args = Cli.parse_args(
        argv=[
            *_flags(name="COMMON"),
            *_flags(name="STEP_FLAGS"),
            *_flags(name="POMDP_FLAGS"),
            "--pomdp-solver",
            solver,
            *(["--pomdp-search-depth", "1"] if solver == "expectimax" else []),
            "--output-dir",
            str(tmp_path),
        ]
    )
    runner = Mock()
    monkeypatch.setattr(RackServingCli, "run_method", runner)
    Tossing3DPomdpCli.run(args=args, env_cli=RackServingCli)
    runner.assert_called_once()
    assert runner.call_args.kwargs["args"] is args
    assert callable(runner.call_args.kwargs["method_factory"])
    assert not list(tmp_path.iterdir())


def test_documented_bash_has_valid_syntax_without_execution():
    blocks = re.findall(r"```bash\n(.*?)```", DOCUMENT.read_text(), flags=re.DOTALL)
    assert blocks
    result = subprocess.run(["bash", "-n"], input="\n".join(blocks), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
