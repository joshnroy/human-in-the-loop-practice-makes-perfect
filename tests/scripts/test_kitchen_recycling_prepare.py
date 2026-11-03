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
