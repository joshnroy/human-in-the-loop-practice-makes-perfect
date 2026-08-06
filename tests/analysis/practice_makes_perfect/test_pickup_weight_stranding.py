"""The stranding reader, tested against hand-built `stats.json` trees rather than
against a real sweep: every number it reports is a count over
`practice_outcomes_per_cycle`, and a fixture makes each claim's input explicit.

The claims worth pinning are the ones a plausible implementation gets wrong:
stranding must be **terminal-from-here** rather than "the first period with no pile
access" (a robot that misses the pile for one period and comes back is not stranded),
a run that never strands must report `None` rather than the last index, and weight
draws must equal pickups **exactly**, since that identity is the whole reason a
stranded run's weight sample has size 1.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.pickup_weight_stranding import PickupWeightStranding


def _write_run(*, root: Path, arm: str, seed: int, periods: list[dict[str, int]]) -> None:
    """One run's stats.json, carrying only what this module reads: per-window attempt
    counts per lifted skill, plus the trailing empty window every real run writes."""
    run_dir = root / arm / "ees" / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    windows = [
        {name: {"num_attempts": count} for name, count in period.items()} for period in periods
    ]
    windows.append({})
    run_dir.joinpath("stats.json").write_text(
        json.dumps({
            "evaluations": [],
            "num_practice_resets": 0,
            "practice_outcomes_per_cycle": windows,
        })
    )


def test_pile_access_is_read_off_the_pickup_skills(*, tmp_path: Path) -> None:
    _write_run(
        root=tmp_path,
        arm="never",
        seed=0,
        periods=[{"PickupTrash": 3, "MoveRoom": 9}, {"MoveRoom": 12}, {"PickupRecycling": 1}],
    )
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0])
    assert record.pile_access == [True, False, True]


def test_a_run_that_comes_back_to_the_pile_is_not_stranded(*, tmp_path: Path) -> None:
    """A gap is not an onset. Stranding is terminal by definition here -- the ledge makes
    rooms 0-2 absorbing -- so a later pickup proves the earlier gap was not one."""
    _write_run(
        root=tmp_path,
        arm="never",
        seed=0,
        periods=[{"PickupTrash": 1}, {"MoveRoom": 12}, {"PickupTrash": 2}],
    )
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0])
    assert record.stranding_onset is None
    assert record.is_stranded is False


def test_the_onset_is_the_first_period_of_the_terminal_run_of_gaps(*, tmp_path: Path) -> None:
    _write_run(
        root=tmp_path,
        arm="never",
        seed=0,
        periods=[{"PickupTrash": 1}, {"MoveRoom": 12}, {"PickupTrash": 2}, {}, {"MoveRoom": 4}],
    )
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0])
    assert record.stranding_onset == 3
    assert record.is_stranded is True


def test_a_run_that_never_reaches_the_pile_strands_at_period_zero(*, tmp_path: Path) -> None:
    _write_run(root=tmp_path, arm="never", seed=0, periods=[{"MoveRoom": 5}, {"MoveRoom": 5}])
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0])
    assert record.stranding_onset == 0
    assert record.num_weight_draws == 0


def test_weight_draws_equal_pickups_of_either_kind(*, tmp_path: Path) -> None:
    """The identity the sharpened prediction rests on: under weight-at-pickup a run's
    weight sample size IS its pickup count, so a run with one pickup has n=1."""
    _write_run(
        root=tmp_path,
        arm="never",
        seed=0,
        periods=[{"PickupTrash": 4, "PickupRecycling": 1}, {"MoveRoom": 12}],
    )
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0])
    assert record.num_weight_draws == 5
    assert record.num_trash_pickups == 4
    assert record.num_recycling_pickups == 1


def test_post_onset_periods_report_which_skills_still_ran(*, tmp_path: Path) -> None:
    """Terminality is the claim that nothing recovers, and the evidence for it is what a
    stranded period is still able to execute."""
    _write_run(
        root=tmp_path,
        arm="never",
        seed=0,
        periods=[{"PickupTrash": 1}, {"MoveRoom": 8, "PressRecycling": 2}, {"MoveRoom": 9}],
    )
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0])
    assert record.post_onset_skills == {"MoveRoom", "PressRecycling"}
    assert record.num_post_onset_periods == 2


def test_a_missing_seed_is_an_error_rather_than_a_skip(*, tmp_path: Path) -> None:
    """A reader that silently skips a missing run reports a 9-seed result as a 10-seed
    one, which is the denominator going wrong invisibly."""
    _write_run(root=tmp_path, arm="never", seed=0, periods=[{"PickupTrash": 1}])
    with pytest.raises(FileNotFoundError, match="seed 1"):
        PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0, 1])


def test_the_trailing_empty_window_is_not_counted_as_a_period(*, tmp_path: Path) -> None:
    """`record_practice_outcomes` appends one entry per window plus a final one covering
    the last evaluation sweep alone, which contains no practice. Counting it would add a
    phantom stranded period to every run in both arms."""
    _write_run(root=tmp_path, arm="scheduled", seed=0, periods=[{"PickupTrash": 1}])
    (record,) = PickupWeightStranding.read_arm(root=tmp_path / "scheduled", seeds=[0])
    assert record.pile_access == [True]


def test_the_summary_reports_counts_with_denominators(*, tmp_path: Path) -> None:
    for seed, periods in enumerate([
        [{"PickupTrash": 2}, {"PickupTrash": 2}],
        [{"PickupRecycling": 1}, {"MoveRoom": 9}],
        [{"MoveRoom": 9}, {"MoveRoom": 9}],
    ]):
        _write_run(root=tmp_path, arm="never", seed=seed, periods=periods)
    summary = PickupWeightStranding.summarise(
        records=PickupWeightStranding.read_arm(root=tmp_path / "never", seeds=[0, 1, 2])
    )
    assert summary["num_seeds"] == 3
    assert summary["num_stranded"] == 2
    # Seed 1's onset is its LAST period, which has no later period to recover in, so it
    # is not evidence of stranding; seed 2's onset at period 0 is.
    assert summary["num_stranded_before_last_period"] == 1
    assert summary["num_seeds_with_one_weight_draw"] == 1
    assert summary["weight_draws_per_seed"] == [4, 1, 0]
    assert summary["stranding_onsets"] == [None, 1, 0]
