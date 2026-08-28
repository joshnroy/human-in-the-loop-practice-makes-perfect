"""Aggregate over training seeds, not over tasks or mismatched transition grids."""

import numpy as np
import pytest

from analysis.practice_makes_perfect.tossing3d_transfer_curve import TransferCurve


def test_mean_and_sample_std_are_across_seed_success_rates() -> None:
    curve = TransferCurve.aggregate(
        evaluations=[[[0, 0, 10], [20, 2, 10]], [[0, 2, 10], [7, 6, 10]]]
    )
    np.testing.assert_allclose(curve["mean_success"], [0.1, 0.4])
    np.testing.assert_allclose(curve["std_success"], np.std([[0, 0.2], [0.2, 0.6]], axis=0, ddof=1))
    assert curve["cycle"] == [0, 1]
    assert curve["mean_actions"] == [0, 13.5]


@pytest.mark.parametrize(
    "runs",
    [
        [[[0, 0, 10]]],
        [[[0, 0, 10]], [[0, 0, 10], [20, 1, 10]]],
        [[[0, 0, 1]], [[0, 0, 10]]],
    ],
)
def test_reject_incomplete_or_wrong_denominator_inputs(*, runs) -> None:
    with pytest.raises(ValueError):
        TransferCurve.aggregate(evaluations=runs)


def test_loader_requires_all_seeds_and_the_far_side_protocol(*, tmp_path) -> None:
    import json

    config = {
        "args": {
            "layout": "same-side",
            "evaluation_layout": "barrier",
            "num_cycles": "100",
            "max_steps_per_interaction": "20",
            "num_test_tasks": "10",
            "sampler_max_train_iters": "10000",
            "practice_reset_policy": "never",
            "goal_pursuit_horizon": "None",
            "ask_for_reset_cube_bin_cost": "None",
        }
    }
    stats = {
        "evaluations": [[i * 20, 2, 10] for i in range(101)],
        "num_practice_resets": 0,
        "num_human_interventions_recorded": 0,
    }
    for seed in range(10):
        folder = tmp_path / "ees" / str(seed)
        folder.mkdir(parents=True)
        config["args"]["seed"] = str(seed)
        (folder / "config_snapshot.json").write_text(json.dumps(config))
        (folder / "stats.json").write_text(json.dumps(stats))
    assert TransferCurve.load(results_root=tmp_path)["num_seeds"] == 10
    config["args"]["evaluation_layout"] = "same-side"
    (folder / "config_snapshot.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="protocol"):
        TransferCurve.load(results_root=tmp_path)
    (folder / "stats.json").unlink()
    with pytest.raises(FileNotFoundError):
        TransferCurve.load(results_root=tmp_path)
