import json
import sys
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from scripts.tossing3d_competence_2x2 import execute_experiments, experiment_commands


@pytest.mark.parametrize("search_iterations", [100, 1000])
@pytest.mark.parametrize("solver,depth", [("determinized_astar", 3), ("expectimax", 6)])
def test_four_commands_parse_with_only_model_and_engine_varying(
    *, tmp_path: Path, search_iterations: int, solver: str, depth: int
) -> None:
    commands = experiment_commands(
        results_root=tmp_path,
        num_seeds=1,
        num_cycles=10,
        python="python",
        max_search_iterations=search_iterations,
        solver=solver,
        search_depth=depth,
    )
    parsed = [Cli.parse_args(argv=command[3:]) for _, command in commands]
    assert {(args.pomdp_competence_model, args.pomdp_inference_engine) for args in parsed} == {
        ("global_curve", "particle"),
        ("global_curve", "grid"),
        ("local_trend", "particle"),
        ("local_trend", "grid"),
    }
    configurations = []
    for args in parsed:
        assert args.num_cycles == 10
        assert args.num_test_tasks == 10
        assert args.max_steps_per_interaction == 20
        assert args.seed == 0
        assert args.human_reset_practice_cost == 5
        assert args.pomdp_linear_cost_lambda == 0.0003
        assert args.pomdp_max_search_iterations == search_iterations
        assert args.pomdp_solver == solver
        assert args.pomdp_search_depth == depth
        assert args.record_episode_traces
        config = vars(args).copy()
        for key in ("pomdp_competence_model", "pomdp_inference_engine", "output_dir"):
            del config[key]
        configurations.append(config)
    assert all(config == configurations[0] for config in configurations)


def test_sweep_executor_records_all_arms_when_one_child_fails(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("OPENBLAS_NUM_THREADS", "PYTHONUNBUFFERED", "MPLCONFIGDIR"):
        monkeypatch.setenv(key, "test-placeholder")
    (tmp_path / "logs").mkdir()
    matrix = experiment_commands(
        results_root=tmp_path, num_seeds=1, num_cycles=10, python=sys.executable
    )
    runs = [
        (
            name,
            [
                sys.executable,
                "-c",
                f"import sys; print({name!r}); sys.exit({int(index == 1)})",
                "--seed",
                "0",
            ],
        )
        for index, (name, _) in enumerate(matrix)
    ]
    records = execute_experiments(runs=runs, results_root=tmp_path, max_workers=2)
    assert [record["returncode"] for record in records] == [0, 1, 0, 0]
    assert [record["name"] for record in records] == [name for name, _ in runs]
    for record in records:
        timing = json.loads(Path(str(record["timing"])).read_text())
        assert timing["returncode"] == record["returncode"]
        assert timing["spawn_attempts"] == 1
        assert timing["elapsed_seconds"] >= 0
        assert timing["max_workers"] == 2
        assert record["name"] in Path(str(record["log"])).read_text()
