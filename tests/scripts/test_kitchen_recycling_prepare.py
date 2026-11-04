import pytest

from hitl_pmp.cli import Cli
from scripts.kitchen_recycling.prepare import ARMS, argv


@pytest.mark.parametrize("arm", [arm for arm in ARMS if arm != "ees"])
def test_only_expectimax_is_given_a_search_depth(*, arm: str) -> None:
    """Curtis is bounded by its iterations; the CLI refuses a depth for it."""
    args = Cli.parse_args(
        argv=[
            *argv(arm=arm, seed=0, cost_file="costs.json", kitchen_config="config.json"),
            "--output-dir",
            "unused",
        ]  # fmt: skip
    )
    assert args.pomdp_solver == ("expectimax" if arm == "expectimax" else "determinized_astar")
    assert args.pomdp_search_depth == (6 if arm == "expectimax" else None)


@pytest.mark.parametrize("arm", ARMS)
def test_an_iteration_rows_budget_and_evaluation_workers_parse(*, arm: str) -> None:
    """The CLI takes a 65k / four-worker row as written; per-task RNG comes with it."""
    args = Cli.parse_args(
        argv=[
            *argv(
                arm=arm,
                seed=0,
                cost_file="costs.json",
                kitchen_config="config.json",
                practice_step_budget=65000,
                evaluation_workers=4,
            ),
            "--output-dir",
            "unused",
        ]
    )
    assert args.practice_step_budget == 65000
    assert (args.evaluation_workers, args.evaluation_task_rng) == (4, "per-task")
    assert args.defer_rendering
    assert not getattr(args, "ees_preview_test_goals", False)
