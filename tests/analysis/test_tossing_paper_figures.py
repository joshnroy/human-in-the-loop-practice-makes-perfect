import json
from pathlib import Path

import numpy as np
import pytest

from analysis.tossing_paper_figures import ARMS, COSTS, FrozenSweep


@pytest.fixture
def synthetic_inputs(*, tmp_path: Path) -> Path:
    """Invented fixture values; no experimental measurements are distributed."""
    costs, curves = {}, {}
    for cost in COSTS:
        for arm, _, _, n in ARMS:
            for seed in range(n):
                key = f"{cost}/{arm}/{seed}"
                costs[key] = {
                    "robot_cost": 17 + seed,
                    "human_cost": 345 + seed,
                    "human_invocations": 3 + seed,
                    "practice_steps": 200000,
                }
                if arm not in {"dp", "dpperiod"}:
                    points = [[0, 4 + seed % 3], [1, 10 + seed % 5]]
                    if cost == "10000" and arm in {"expectimax", "curtis"}:
                        costs[key]["practice_steps"] = 3200
                    else:
                        points.append([63, 45 + seed % 5])
                    curves[key] = points
    (tmp_path / "costs.json").write_text(json.dumps(costs))
    (tmp_path / "curves50.json").write_text(json.dumps(curves))
    return tmp_path


def test_stopped_learning_curves_end_at_last_measured_snapshot(*, synthetic_inputs: Path) -> None:
    sweep = FrozenSweep.load(input_dir=synthetic_inputs)
    steps, rows = sweep.shared_curve(cost="10000", arm="expectimax")
    assert steps.tolist() == [0, 3200]
    assert rows.shape == (10, 2)
    for seed in range(10):
        expected = sweep.curves[f"10000/expectimax/{seed}"]
        assert rows[seed].tolist() == [value for _, value in expected]


def test_summary_retains_initial_snapshot_for_stopped_runs_and_unequal_seed_counts(
    *, synthetic_inputs: Path
) -> None:
    sweep = FrozenSweep.load(input_dir=synthetic_inputs)
    solved = sweep.summary(cost="10000", arm="curtis", metric="solved")
    assert len(solved) == 10
    assert solved[0] == np.mean([n for _, n in sweep.curves["10000/curtis/0"]])
    assert len(sweep.summary(cost="20", arm="dp", metric="resets")) == 5
    assert len(sweep.summary(cost="20", arm="ees", metric="resets")) == 10
    with pytest.raises(ValueError, match="not re-scored"):
        sweep.summary(cost="20", arm="dp", metric="solved")


def test_charges_are_read_not_inferred(*, synthetic_inputs: Path) -> None:
    sweep = FrozenSweep.load(input_dir=synthetic_inputs)
    steps, rows = sweep.shared_curve(cost="20", arm="ees")
    assert steps[-1] == 200000
    assert rows.shape == (10, 3)
    values = sweep.summary(cost="100", arm="expectimax", metric="total_cost")
    assert values.tolist() == [362 + 2 * seed for seed in range(10)]


def test_missing_run_coverage_is_rejected(*, synthetic_inputs: Path) -> None:
    path = synthetic_inputs / "costs.json"
    costs = json.loads(path.read_text())
    del costs["0/ees/0"]
    path.write_text(json.dumps(costs))
    with pytest.raises(ValueError, match="240 run summaries"):
        FrozenSweep.load(input_dir=synthetic_inputs)
