"""Tests for the Ball-Ring sampler-iteration analysis.

The load-bearing test here is `test_summary_table_matches_the_committed_arms`: it pins
the analysis to the *real* committed aggregate, so the five numbers quoted in the
experiment log and in `EesMethod.sampler_max_train_iters`' own comment cannot drift away
from the data without a test failing.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.ballring_sampler_iters import BallRingSamplerIters

_ARMS_JSON = Path(__file__).parents[3] / "docs/experiment-logs/2026-08-03-ballring-arms.json"

# (iters, mean %, sd %, worst-seed %) as reported in
# docs/experiment-logs/2026-08-03-ballring-iters.md.
_EXPECTED_TABLE = [
    (1000, 83.0, 22.1, 40.0),
    (3000, 90.0, 28.3, 10.0),
    (10000, 99.0, 3.2, 90.0),
    (30000, 91.0, 12.0, 60.0),
    (100000, 89.0, 16.0, 50.0),
]


def test_summary_table_matches_the_committed_arms():
    arms = BallRingSamplerIters.load_arms(json_path=_ARMS_JSON)
    table = BallRingSamplerIters.summary_table(arms=arms)
    assert len(table) == len(_EXPECTED_TABLE)
    for (iters, mean, sd, worst), (want_iters, want_mean, want_sd, want_worst) in zip(
        table, _EXPECTED_TABLE, strict=True
    ):
        assert iters == want_iters
        assert mean == pytest.approx(want_mean, abs=0.05)
        assert sd == pytest.approx(want_sd, abs=0.05)
        assert worst == pytest.approx(want_worst, abs=0.05)


def test_three_thousand_arm_is_bimodal_not_broadly_spread():
    """The 3000 arm's sd of 28.3 is ONE collapsed seed, not a wide spread.

    Pinned because the log leans on this distinction: reported as `90.0 +- 28.3` the arm
    reads as unreliable-on-average, when in fact 9 of 10 seeds land at 90-100% and a
    single seed sits at 10%. If a future re-aggregation smears that into a genuine
    spread, the prose in the log becomes wrong and this fails.
    """
    arms = BallRingSamplerIters.load_arms(json_path=_ARMS_JSON)
    percents = sorted(BallRingSamplerIters.endpoint_percents(seed_curves=arms["iters3k"]))
    assert percents[0] == pytest.approx(10.0)
    assert all(value >= 90.0 for value in percents[1:])


def test_ten_thousand_has_the_best_point_estimate_and_lowest_spread():
    """10000 is the argmax of this sweep on BOTH mean and sd.

    This is a point-estimate statement only, and it is NOT why 10000 was chosen as the
    default -- that rests on 1000 sitting below the `n_iter_no_change = 5000` floor and on
    10000 being predicators' own settings.py default. No pairwise difference in this sweep
    is significant at n=10 (paired, vs 10000: 1000 p=0.057, 30000 p=0.070, 100000 p=0.085,
    3000 p=0.350), so this test asserts an ordering, NOT that an optimum was established.
    """
    table = BallRingSamplerIters.summary_table(
        arms=BallRingSamplerIters.load_arms(json_path=_ARMS_JSON)
    )
    best_mean = max(table, key=lambda row: row[1])
    lowest_sd = min(table, key=lambda row: row[2])
    assert best_mean[0] == 10000
    assert lowest_sd[0] == 10000


def test_iters10k_arm_is_the_same_run_as_the_envfix_arm():
    """`iters10k` and `fix_envfix` in the aggregate are ONE measurement, not two.

    Both are current `main` at the (new) default of 10000, so they are byte-identical
    per seed. The log quotes 99.0 for both; without this note a reader counts two
    independent confirmations of 99.0 where there is one.
    """
    arms = BallRingSamplerIters.load_arms(json_path=_ARMS_JSON)
    assert BallRingSamplerIters.endpoint_percents(
        seed_curves=arms["iters10k"]
    ) == BallRingSamplerIters.endpoint_percents(seed_curves=arms["fix_envfix"])


def test_endpoint_uses_the_last_sweep_not_the_best_one():
    seed_curves = {"0": [[0, 0, 10], [100, 10, 10], [200, 4, 10]]}
    assert BallRingSamplerIters.endpoint_percents(seed_curves=seed_curves) == [40.0]


def test_endpoint_does_not_depend_on_row_order():
    """Rows are read out of JSON; nothing guarantees they arrive sorted by transitions."""
    shuffled = {"0": [[200, 4, 10], [0, 0, 10], [100, 10, 10]]}
    assert BallRingSamplerIters.endpoint_percents(seed_curves=shuffled) == [40.0]


def test_mean_curve_averages_across_seeds():
    seed_curves = {"0": [[0, 2, 10], [100, 6, 10]], "1": [[0, 4, 10], [100, 10, 10]]}
    curve = BallRingSamplerIters.mean_curve(seed_curves=seed_curves)
    assert curve[0][0] == pytest.approx(30.0)
    assert curve[100][0] == pytest.approx(80.0)


def test_load_arms_rejects_an_aggregate_missing_an_arm(*, tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text('{"iters1k": {"0": [[0, 1, 10]]}}')
    with pytest.raises(ValueError, match="missing sampler-iteration arms"):
        BallRingSamplerIters.load_arms(json_path=path)


def test_render_writes_a_figure(*, tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    BallRingSamplerIters.render(arms_json=_ARMS_JSON, output=output)
    assert output.exists()
    assert output.stat().st_size > 0
