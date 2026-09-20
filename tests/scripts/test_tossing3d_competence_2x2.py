from pathlib import Path

from hitl_pmp.cli import Cli
from scripts.tossing3d_competence_2x2 import experiment_commands


def test_four_commands_parse_with_only_model_and_engine_varying(*, tmp_path: Path) -> None:
    commands = experiment_commands(
        results_root=tmp_path, num_seeds=1, num_cycles=10, python="python"
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
        config = vars(args).copy()
        for key in ("pomdp_competence_model", "pomdp_inference_engine", "output_dir"):
            del config[key]
        configurations.append(config)
    assert all(config == configurations[0] for config in configurations)
