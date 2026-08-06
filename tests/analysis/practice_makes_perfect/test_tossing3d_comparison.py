"""Tests for the Tossing3D EES analysis.

Two things are pinned here that this domain gets wrong more easily than Tossing Room
does. First, **seeds do not share a checkpoint grid**: a practice period on Tossing3D
ends when nothing is left worth practicing, so each seed's x-axis is data-driven and an
aggregation that assumes a common grid silently drops or misaligns points. Second, the
standoff of a run is read from its own `config_snapshot.json` rather than from the
directory someone named -- so a mislabelled directory cannot move a point on the plot.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_comparison import (
    Tossing3DComparison,
    _parse_arm,
)


def _write_run(
    *,
    root: Path,
    method: str,
    seed: str,
    evaluations: list[list[int]],
    standoff: float | None = None,
) -> None:
    run_dir = root / method / seed
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stats.json").write_text(
        json.dumps({"evaluations": evaluations, "breakdowns": [], "num_practice_resets": 0})
    )
    if standoff is not None:
        (run_dir / "config_snapshot.json").write_text(
            json.dumps({"args": {"oracle_throw_standoff": str(standoff)}})
        )


def test_per_seed_counts_reads_the_triples_verbatim(*, tmp_path: Path) -> None:
    _write_run(root=tmp_path, method="ees", seed="0", evaluations=[[0, 3, 10], [9, 7, 10]])
    counts = Tossing3DComparison.per_seed_counts(root=tmp_path, method="ees")
    assert counts == {"0": {0: (3, 10), 9: (7, 10)}}


def test_pooled_endpoints_uses_each_seeds_own_first_and_last_checkpoint(*, tmp_path: Path) -> None:
    """The load-bearing case: the two seeds stop at *different* transition counts, because
    a practice period here ends when there is nothing left worth practicing. Taking a
    fixed final checkpoint would drop seed 1 entirely."""
    _write_run(root=tmp_path, method="ees", seed="0", evaluations=[[0, 1, 10], [9, 6, 10]])
    _write_run(root=tmp_path, method="ees", seed="1", evaluations=[[0, 2, 10], [14, 8, 10]])
    endpoints = Tossing3DComparison.pooled_endpoints(root=tmp_path, method="ees")
    assert endpoints["first"] == (3, 20)
    assert endpoints["final"] == (14, 20)


def test_pooled_endpoints_is_empty_when_no_runs_exist(*, tmp_path: Path) -> None:
    assert Tossing3DComparison.pooled_endpoints(root=tmp_path, method="ees") == {}


def test_endpoint_rates_orders_seeds_numerically_so_two_arms_line_up(*, tmp_path: Path) -> None:
    """Seed directories are strings, so a lexicographic sort puts "10" before "2" and a
    paired test then compares seed 10 of one arm against seed 2 of the other."""
    for seed, solved in (("0", 1), ("2", 2), ("10", 3)):
        _write_run(
            root=tmp_path, method="ees", seed=seed, evaluations=[[0, 0, 10], [9, solved, 10]]
        )
    rates = Tossing3DComparison.endpoint_rates(root=tmp_path, method="ees")
    assert rates["seeds"] == [0.0, 2.0, 10.0]
    assert rates["final"] == [10.0, 20.0, 30.0]


def test_mean_curve_averages_only_the_seeds_that_reached_a_checkpoint(*, tmp_path: Path) -> None:
    _write_run(root=tmp_path, method="ees", seed="0", evaluations=[[0, 0, 10], [6, 4, 10]])
    _write_run(root=tmp_path, method="ees", seed="1", evaluations=[[0, 2, 10], [6, 8, 10]])
    _write_run(root=tmp_path, method="ees", seed="2", evaluations=[[0, 4, 10], [9, 9, 10]])
    curve = Tossing3DComparison.mean_curve(root=tmp_path, method="ees")
    assert curve[0][0] == pytest.approx(20.0)
    assert curve[6][0] == pytest.approx(60.0)
    # Only seed 2 reached 9 transitions, so its mean is that seed and its stderr is 0.
    assert curve[9] == (pytest.approx(90.0), 0.0)


def test_mean_curve_by_checkpoint_averages_the_same_seeds_at_every_point(
    *,
    tmp_path: Path,
) -> None:
    """The reason this exists: the two seeds' 2nd checkpoint sits at 6 and at 8
    transitions, so `mean_curve` computes that point over one seed each while this
    computes it over both. The x coordinate is the median of the two."""
    _write_run(root=tmp_path, method="ees", seed="0", evaluations=[[0, 1, 10], [6, 5, 10]])
    _write_run(root=tmp_path, method="ees", seed="1", evaluations=[[0, 3, 10], [8, 7, 10]])
    curve = Tossing3DComparison.mean_curve_by_checkpoint(root=tmp_path, method="ees")
    assert [point[0] for point in curve] == [0, 7]
    assert curve[0][1] == pytest.approx(20.0)
    assert curve[1][1] == pytest.approx(60.0)
    # Contrast: the transition-keyed view splits that same checkpoint into two n = 1 points.
    assert sorted(Tossing3DComparison.mean_curve(root=tmp_path, method="ees")) == [0, 6, 8]


def test_mean_curve_by_checkpoint_lets_a_short_seed_drop_out_of_the_tail(
    *,
    tmp_path: Path,
) -> None:
    """A seed whose run ended earlier contributes to the prefix only; the tail's n falls
    and its stderr goes to 0 at n = 1, rather than the seed being silently extrapolated."""
    _write_run(root=tmp_path, method="ees", seed="0", evaluations=[[0, 1, 10], [6, 5, 10]])
    _write_run(
        root=tmp_path, method="ees", seed="1", evaluations=[[0, 3, 10], [8, 7, 10], [12, 9, 10]]
    )
    curve = Tossing3DComparison.mean_curve_by_checkpoint(root=tmp_path, method="ees")
    assert len(curve) == 3
    assert curve[2] == (12, pytest.approx(90.0), 0.0)


def test_mean_curve_by_checkpoint_is_empty_with_no_runs(*, tmp_path: Path) -> None:
    assert Tossing3DComparison.mean_curve_by_checkpoint(root=tmp_path, method="ees") == []


def test_standoff_counts_reads_the_standoff_from_the_config_snapshot(*, tmp_path: Path) -> None:
    """The directory names here deliberately disagree with the recorded standoffs. The
    recorded value wins, because it is the one the run actually used."""
    _write_run(
        root=tmp_path / "mislabelled-a",
        method="skill-oracle",
        seed="0",
        evaluations=[[0, 10, 10]],
        standoff=1.35,
    )
    _write_run(
        root=tmp_path / "mislabelled-b",
        method="skill-oracle",
        seed="0",
        evaluations=[[0, 0, 10]],
        standoff=1.60,
    )
    assert Tossing3DComparison.standoff_counts(root=tmp_path) == {1.35: (10, 10), 1.6: (0, 10)}


def test_standoff_counts_pools_seeds_that_share_a_standoff(*, tmp_path: Path) -> None:
    for seed in ("0", "1"):
        _write_run(
            root=tmp_path / "s135",
            method="skill-oracle",
            seed=seed,
            evaluations=[[0, 4, 10]],
            standoff=1.35,
        )
    assert Tossing3DComparison.standoff_counts(root=tmp_path) == {1.35: (8, 20)}


def test_standoff_counts_raises_rather_than_dropping_a_run_with_no_snapshot(
    *,
    tmp_path: Path,
) -> None:
    _write_run(root=tmp_path / "s135", method="skill-oracle", seed="0", evaluations=[[0, 4, 10]])
    with pytest.raises(FileNotFoundError, match="config_snapshot.json"):
        Tossing3DComparison.standoff_counts(root=tmp_path)


def test_standoff_counts_raises_on_a_snapshot_from_some_other_domain(*, tmp_path: Path) -> None:
    """A `config_snapshot.json` exists but records no standoff -- a lightswitch run
    dropped into the tree, say. Reading it as a standoff sweep would be a fabrication."""
    run = tmp_path / "s135" / "skill-oracle" / "0"
    run.mkdir(parents=True)
    (run / "stats.json").write_text(json.dumps({"evaluations": [[0, 4, 10]]}))
    (run / "config_snapshot.json").write_text(json.dumps({"args": {"env": "lightswitch"}}))
    with pytest.raises(KeyError, match="oracle_throw_standoff"):
        Tossing3DComparison.standoff_counts(root=tmp_path)


def test_standoff_counts_raises_on_an_empty_root_rather_than_drawing_nothing(
    *,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="no standoff sweep"):
        Tossing3DComparison.standoff_counts(root=tmp_path)


def test_paired_tests_compare_every_arm_pair_not_only_consecutive_ones(
    *,
    tmp_path: Path,
    capsys,
) -> None:
    """Three arms means three comparisons. Zipping a label list against its own tail
    yields only two and silently omits first-vs-last, which is the shape this domain will
    be plotted in as soon as the skill-oracle ceiling is drawn as an arm."""
    for method in ("ees", "random-skills", "skill-oracle"):
        for seed in ("0", "1"):
            _write_run(
                root=tmp_path, method=method, seed=seed, evaluations=[[0, 1, 10], [9, 5, 10]]
            )
    arms = [(method, tmp_path) for method in ("ees", "random-skills", "skill-oracle")]
    Tossing3DComparison.print_paired_tests(
        arms=arms, method_of={method: method for method, _ in arms}
    )
    printed = capsys.readouterr().out
    assert "ees vs random-skills" in printed
    assert "random-skills vs skill-oracle" in printed
    assert "ees vs skill-oracle" in printed


def test_parse_arm_keeps_a_results_path_that_contains_an_equals_sign() -> None:
    """`--arm "ees=/results/standoff=1.35"` is one label and one path. A parse that
    counts "=" separators reads it as label/method/path and plots "1.35" as a directory."""
    assert _parse_arm(raw="ees=/results/standoff=1.35") == (
        "ees",
        "ees",
        Path("/results/standoff=1.35"),
    )


def test_parse_arm_takes_the_method_from_the_label_side_only() -> None:
    assert _parse_arm(raw="ees-20cyc:ees=/results/a") == ("ees-20cyc", "ees", Path("/results/a"))
    assert _parse_arm(raw="ees=/results/a") == ("ees", "ees", Path("/results/a"))


def test_parse_arm_rejects_a_malformed_spec() -> None:
    for raw in ("ees", "=path", "ees="):
        with pytest.raises(ValueError, match="expected"):
            _parse_arm(raw=raw)


def test_solving_band_reports_the_measured_endpoints_not_an_inferred_interval() -> None:
    counts = {1.20: (10, 10), 1.35: (10, 10), 1.38: (7, 10), 1.42: (0, 10), 1.65: (0, 10)}
    assert Tossing3DComparison.solving_band(counts=counts) == (1.20, 1.38)


def test_solving_band_is_none_when_nothing_solved() -> None:
    assert Tossing3DComparison.solving_band(counts={1.20: (0, 10), 1.65: (0, 10)}) is None


def test_a_single_checkpoint_arm_is_drawn_as_a_reference_level_not_a_point(
    *,
    tmp_path: Path,
) -> None:
    """`skill-oracle` never practices, so it evaluates once and its "curve" is one point.
    Plotted as a line it draws nothing and leaves a legend entry with no line, which reads
    as a plotting bug; it must become an `axhline` at its own mean instead."""
    import matplotlib.pyplot as plt

    _write_run(root=tmp_path, method="skill-oracle", seed="0", evaluations=[[0, 9, 10]])
    _write_run(root=tmp_path, method="skill-oracle", seed="1", evaluations=[[0, 10, 10]])
    figure, ax = plt.subplots()
    Tossing3DComparison._plot_curves(
        ax=ax, arms=[("oracle", tmp_path)], method_of={"oracle": "skill-oracle"}
    )
    horizontals = [line.get_ydata()[0] for line in ax.get_lines() if line.get_label() == "oracle"]
    assert horizontals == [pytest.approx(95.0)]
    plt.close(figure)


def test_render_writes_a_figure_with_and_without_the_standoff_panel(*, tmp_path: Path) -> None:
    ees_root = tmp_path / "arms"
    _write_run(root=ees_root, method="ees", seed="0", evaluations=[[0, 3, 10], [9, 7, 10]])
    _write_run(root=ees_root, method="random-skills", seed="0", evaluations=[[0, 1, 10]])
    standoff_root = tmp_path / "standoffs"
    _write_run(
        root=standoff_root / "s120",
        method="skill-oracle",
        seed="0",
        evaluations=[[0, 10, 10]],
        standoff=1.20,
    )
    arms = [("ees", ees_root), ("random-skills", ees_root)]
    method_of = {"ees": "ees", "random-skills": "random-skills"}

    two_panel = tmp_path / "two.png"
    Tossing3DComparison.render(
        arms=arms, method_of=method_of, standoff_root=None, output=two_panel, title="t"
    )
    assert two_panel.stat().st_size > 0

    three_panel = tmp_path / "nested" / "three.png"
    Tossing3DComparison.render(
        arms=arms, method_of=method_of, standoff_root=standoff_root, output=three_panel, title="t"
    )
    assert three_panel.stat().st_size > 0
