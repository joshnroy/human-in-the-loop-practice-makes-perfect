import json
import subprocess
import sys
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from scripts.kitchen_recycling import prepare
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


def test_every_swept_row_prices_its_resets_only_in_its_cost_file(*, tmp_path: Path) -> None:
    """A cost sweep carries each row's reset cost only in its own cost file: no flag
    prices a reset, so no reset cost can trip the POMDP's particle cost ceiling (the
    construction assert that killed kitchen v10's expectimax and Curtis rows at reset
    costs 100 and 10000 is gone with the per-skill price it checked)."""
    out = tmp_path / "launch"
    subprocess.run(
        [
            sys.executable,
            str(Path(prepare.__file__)),
            *("--out", str(out), "--tag", "kitchen-recycling-unit-test"),
            *("--human-reset-costs", "5", "100", "10000", "0", "--max-parallel", "36"),
        ],
        check=True,
        capture_output=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    for row in manifest["runs"]:
        assert "--human-reset-practice-cost" not in row["argv"]
        args = Cli.parse_args(argv=[*row["argv"], "--output-dir", "unused"])
        assert not hasattr(args, "human_reset_practice_cost")
        costs = json.loads(
            (out / "costs" / f"human-reset-cost-{row['human_reset_cost']}.json").read_text()
        )
        assert {skill["cost"]["value"] for skill in costs["human_skills"].values()} == {
            row["human_reset_cost"]
        }
