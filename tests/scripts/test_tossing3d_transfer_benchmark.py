"""Pin the established benchmark budget, separate from the short video demo."""

from pathlib import Path

from hitl_pmp.cli import Cli
from scripts.tossing3d_transfer_benchmark import TransferBenchmark


def test_standard_budget_and_far_side_tests(*, tmp_path: Path) -> None:
    runs = TransferBenchmark.plan(results_root=tmp_path)
    assert [run.seed for run in runs] == list(range(10))
    for run in runs:
        args = Cli.parse_args(argv=run.command[3:])
        assert (args.num_cycles, args.max_steps_per_interaction, args.num_test_tasks) == (
            100,
            20,
            10,
        )
        assert args.layout == "same-side"
        assert args.evaluation_layout == "barrier"
        assert args.practice_reset_policy == "never"
        assert args.practice_reset_interval is None
        assert args.ask_for_reset_cube_bin_cost is None
        assert args.sampler_max_train_iters == 10000
        assert args.goal_pursuit_horizon is None
        assert args.output_dir == tmp_path / "ees" / str(run.seed)
