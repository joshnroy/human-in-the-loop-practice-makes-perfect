"""The bimodal-wall-clock reader, tested against hand-built run trees rather than the
committed sweeps: every claim it makes is a count or a partition over `stats.json` plus
`timing.json`, and a fixture makes each claim's input explicit.

The claims worth pinning are the ones a plausible implementation gets wrong: the two
partitions must be derived from *different* files (one from timing, one from practice
outcomes) or the agreement between them is circular; a missing `timing.json` must raise
rather than silently drop a seed; the two budgets must carry the same seed set or the
paired comparison is undefined; and a run that misses the pile for one cycle and comes
back is not late-stranded.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.reset_free_wallclock_modes import (
    ResetFreeWallclockModes,
    SeedRun,
)


def _write_run(
    *,
    root: Path,
    seed: int,
    periods: list[dict[str, int]],
    elapsed: float,
    solved: list[int] | None = None,
    omit_timing: bool = False,
) -> None:
    """One run's `stats.json` + `timing.json`, carrying only the fields this module
    reads."""
    run_dir = root / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    windows = [
        {name: {"num_attempts": count} for name, count in period.items()} for period in periods
    ]
    # Every real run writes one trailing window covering the final evaluation sweep
    # alone, which contains no practice; the reader must drop it, not count it.
    windows.append({})
    sweeps = solved if solved is not None else [0] * (len(periods) + 1)
    run_dir.joinpath("stats.json").write_text(
        json.dumps({
            "evaluations": [[100 * i, count, 30] for i, count in enumerate(sweeps)],
            "practice_outcomes_per_cycle": windows,
        })
    )
    if omit_timing:
        return
    run_dir.joinpath("timing.json").write_text(
        json.dumps({
            "elapsed_seconds": elapsed,
            "sweep_runs_in_flight_at_start": 3,
            "sweep_runs_in_flight_at_end": 3,
            "machine_at_start": {"cli_processes": 4, "load_average_1min": 7.5},
            "machine_at_end": {"cli_processes": 11, "load_average_1min": 13.6},
        })
    )


def _two_mode_budget(*, root: Path, elapsed_fast: float, elapsed_slow: float) -> None:
    """Six seeds stranded in cycle 1 and fast, four stranded later and slow -- the shape
    the 10x one-way reset-free cell actually has."""
    for seed in (2, 3, 4, 5, 8, 9):
        _write_run(
            root=root,
            seed=seed,
            periods=[{"PickupTrash": 1, "ThrowTrash": 1}, {"MoveRoom": 12}, {"MoveRoom": 12}],
            elapsed=elapsed_fast + seed,
            solved=[1, 2, 2, 2],
        )
    for seed in (0, 1, 6, 7):
        _write_run(
            root=root,
            seed=seed,
            periods=[{"PickupTrash": 19, "ThrowTrash": 18}, {"PickupTrash": 3}, {"MoveRoom": 12}],
            elapsed=elapsed_slow + seed,
            solved=[1, 18, 19, 19],
        )


def test_effective_attempts_count_pile_skills_only(*, tmp_path: Path) -> None:
    """A stranded robot walks and presses buttons for a whole cycle. Counting those
    would report a starved arm as busy, which is the failure this measure exists to
    avoid."""
    _write_run(
        root=tmp_path,
        seed=0,
        periods=[{"MoveRoom": 120, "PressTrash": 30, "PickupTrash": 1, "ThrowTrash": 1}],
        elapsed=250.0,
    )
    (run,) = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    assert run.effective_attempts == 2


def test_a_cycle_that_misses_the_pile_and_comes_back_is_not_late_stranded(
    *, tmp_path: Path
) -> None:
    """Stranding is terminal-from-here, so a gap is not an onset: a run that misses the
    pile in cycle 2 and returns in cycle 3 was never stranded at all, and `None` -- not
    cycle 2 -- is the honest answer. Treating that gap as an onset would file the run in
    the fast mode and turn ordinary exploration noise into the effect being claimed."""
    _write_run(
        root=tmp_path,
        seed=0,
        periods=[{"PickupTrash": 2}, {"MoveRoom": 12}, {"PickupTrash": 2}],
        elapsed=250.0,
    )
    (run,) = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    assert run.stranding_onset is None
    assert ResetFreeWallclockModes.is_late_stranded(run=run)


def test_a_run_stranded_in_cycle_one_is_the_early_mode(*, tmp_path: Path) -> None:
    _write_run(root=tmp_path, seed=0, periods=[{"PickupTrash": 2}, {"MoveRoom": 12}], elapsed=250.0)
    (run,) = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    assert run.stranding_onset == 1
    assert not ResetFreeWallclockModes.is_late_stranded(run=run)


def test_the_wall_clock_split_is_taken_at_the_largest_gap(*, tmp_path: Path) -> None:
    """The threshold is data-derived, never a hardcoded number of seconds: the same
    partition has to survive a change of budget that scales every run by ~10x."""
    _two_mode_budget(root=tmp_path, elapsed_fast=254.0, elapsed_slow=595.0)
    runs = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    threshold, gap = ResetFreeWallclockModes.largest_gap(runs=runs)
    assert ResetFreeWallclockModes.slow_seeds(runs=runs) == {0, 1, 6, 7}
    assert gap == pytest.approx(595.0 - (254.0 + 9))
    assert 263.0 < threshold < 595.0


def test_the_two_partitions_are_read_from_different_files(*, tmp_path: Path) -> None:
    """The whole result is that a partition taken off `timing.json` coincides with one
    taken off `stats.json`. If wall clock leaked into the stranding side the agreement
    would be an identity rather than a finding, so this pins that scrambling the
    elapsed times leaves the stranding partition untouched."""
    _two_mode_budget(root=tmp_path, elapsed_fast=254.0, elapsed_slow=595.0)
    before = ResetFreeWallclockModes.late_stranded_seeds(
        runs=ResetFreeWallclockModes.read_budget(directory=tmp_path)
    )
    for seed in range(10):
        timing = json.loads((tmp_path / str(seed) / "timing.json").read_text())
        timing["elapsed_seconds"] = 400.0 - seed
        (tmp_path / str(seed) / "timing.json").write_text(json.dumps(timing))
    after = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    assert ResetFreeWallclockModes.late_stranded_seeds(runs=after) == before == {0, 1, 6, 7}
    assert ResetFreeWallclockModes.slow_seeds(runs=after) != {0, 1, 6, 7}


def test_perfect_agreement_between_the_partitions_hits_the_exact_floor(*, tmp_path: Path) -> None:
    """With a 6/4 split there are C(10, 4) = 210 labellings, and under the two-sided
    total-probability convention only the single most extreme table clears the observed
    probability -- so a perfect match is p = 1/210, the smallest p this design can
    produce. (2/210 is the corresponding floor for a symmetric 5/5 margin; the asymmetry
    matters and is easy to quote wrong.) Naming the floor is what stops "all four slow
    seeds are the four late-stranded ones" reading as stronger evidence than ten seeds
    can carry."""
    _two_mode_budget(root=tmp_path, elapsed_fast=254.0, elapsed_slow=595.0)
    runs = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    table = ResetFreeWallclockModes.agreement_table(runs=runs)
    assert table == ((4, 0), (0, 6))
    assert ResetFreeWallclockModes.agreement_p_value(runs=runs) == pytest.approx(1 / 210)


def test_a_missing_timing_json_raises_rather_than_dropping_the_seed(*, tmp_path: Path) -> None:
    """A reader that skips one silently reports a 9-seed result as a 10-seed one."""
    _write_run(root=tmp_path, seed=0, periods=[{"PickupTrash": 2}], elapsed=250.0)
    _write_run(root=tmp_path, seed=1, periods=[{"PickupTrash": 2}], elapsed=250.0, omit_timing=True)
    with pytest.raises(FileNotFoundError, match="seed 1"):
        ResetFreeWallclockModes.read_budget(directory=tmp_path)


def test_budgets_that_do_not_share_a_seed_set_raise(*, tmp_path: Path) -> None:
    """Every cross-budget statement here is per-seed. With the seed sets unequal the
    pairing is undefined, and a reader that zipped them would silently compare seed 3's
    1x run against seed 4's 10x one."""
    one, ten = tmp_path / "1x", tmp_path / "10x"
    _write_run(root=one, seed=0, periods=[{"PickupTrash": 2}], elapsed=25.0)
    _write_run(root=one, seed=1, periods=[{"PickupTrash": 2}], elapsed=25.0)
    _write_run(root=ten, seed=0, periods=[{"PickupTrash": 2}], elapsed=250.0)
    with pytest.raises(ValueError, match="same seeds"):
        ResetFreeWallclockModes.load_budgets(directories={"1x": one, "10x": ten})


def test_identical_effective_attempts_across_budgets_is_a_reported_null_result(
    *, tmp_path: Path
) -> None:
    """Every per-seed difference being exactly zero is the headline, so it must come out
    as p = 1 on a real test rather than as an unquantified assertion."""
    one, ten = tmp_path / "1x", tmp_path / "10x"
    _two_mode_budget(root=one, elapsed_fast=25.0, elapsed_slow=57.0)
    _two_mode_budget(root=ten, elapsed_fast=254.0, elapsed_slow=595.0)
    budgets = ResetFreeWallclockModes.load_budgets(directories={"1x": one, "10x": ten})
    change = ResetFreeWallclockModes.effective_attempt_change(
        budgets=budgets, earlier="1x", later="10x"
    )
    assert change.differences == [0] * 10
    assert change.p_value == 1.0


def test_a_covariate_that_does_not_separate_the_modes_gets_a_p_value(*, tmp_path: Path) -> None:
    """ "Concurrency is ruled out" is a claim, so it needs a test rather than an
    eyeballed overlap. Identical covariate values across every seed give the largest p
    the enumeration can return."""
    _two_mode_budget(root=tmp_path, elapsed_fast=254.0, elapsed_slow=595.0)
    runs = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    assert ResetFreeWallclockModes.covariate_p_value(runs=runs, name="load_at_start") == 1.0


def test_a_covariate_that_perfectly_tracks_the_modes_hits_the_same_floor(*, tmp_path: Path) -> None:
    """The permutation test over differences in means and the hypergeometric Fisher test
    are separate implementations, so agreeing on 1/210 here is a cross-check on both."""
    _two_mode_budget(root=tmp_path, elapsed_fast=254.0, elapsed_slow=595.0)
    for seed in (0, 1, 6, 7):
        path = tmp_path / str(seed) / "timing.json"
        timing = json.loads(path.read_text())
        timing["machine_at_start"]["load_average_1min"] = 40.0
        path.write_text(json.dumps(timing))
    runs = ResetFreeWallclockModes.read_budget(directory=tmp_path)
    assert ResetFreeWallclockModes.covariate_p_value(
        runs=runs, name="load_at_start"
    ) == pytest.approx(1 / 210)


def test_render_writes_a_figure(*, tmp_path: Path) -> None:
    one, ten = tmp_path / "1x", tmp_path / "10x"
    _two_mode_budget(root=one, elapsed_fast=25.0, elapsed_slow=57.0)
    _two_mode_budget(root=ten, elapsed_fast=254.0, elapsed_slow=595.0)
    budgets = ResetFreeWallclockModes.load_budgets(directories={"1x": one, "10x": ten})
    output = tmp_path / "modes.png"
    ResetFreeWallclockModes.render(budgets=budgets, output=output)
    assert output.stat().st_size > 0


def test_report_carries_counts_with_their_denominators(*, tmp_path: Path) -> None:
    """`x/y` everywhere, never a bare percentage -- the denominators on this domain are
    small and uneven."""
    one, ten = tmp_path / "1x", tmp_path / "10x"
    _two_mode_budget(root=one, elapsed_fast=25.0, elapsed_slow=57.0)
    _two_mode_budget(root=ten, elapsed_fast=254.0, elapsed_slow=595.0)
    budgets = ResetFreeWallclockModes.load_budgets(directories={"1x": one, "10x": ten})
    text = ResetFreeWallclockModes.report(budgets=budgets)
    assert "4/10" in text
    assert "%" not in text


def test_seed_run_is_frozen() -> None:
    run = SeedRun(
        seed=0,
        elapsed_seconds=250.0,
        effective_attempts=2,
        stranding_onset=1,
        final_solved=7,
        num_test_tasks=30,
        runs_in_flight_at_start=3,
        runs_in_flight_at_end=3,
        cli_processes_at_start=4,
        cli_processes_at_end=11,
        load_at_start=7.5,
        load_at_end=13.6,
    )
    with pytest.raises(Exception, match="frozen"):
        run.seed = 1
