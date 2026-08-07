"""`ResetFreeCycleBudget` reads the reset-policy A/B at **two cycle budgets** and asks
whether the reset-free arm's deficit is starvation -- so what is pinned here is the
arithmetic that turns eight sweeps into the gap-versus-budget comparison, not matplotlib.

The cube is budget x ledge x policy. Five things can go wrong silently, and each has a
test:

- **a cell goes missing.** "Did the gap close?" is a difference of differences; with a
  cell absent it is not defined, and a report that drew the comparisons it still could
  would read as a result;
- **a directory is pointed at the wrong budget.** The two budgets differ only in
  `--num-cycles`, so nothing in a path prevents the 1x sweep being read as the 10x one.
  Cycles are counted out of the run's own `breakdowns` and checked against the budget;
- **the manipulation check stops scaling.** `num_practice_resets` is 10 for a 1x
  `scheduled` run and 100 for a 10x one, so a hard-coded 10 would silently reject every
  correct 10x run -- or, worse, a hard-coded pass would accept a mislabelled arm;
- **a family's denominator drifts.** TRASH and RECYCLING are 14 each per seed and EMPTY
  is 2, so a goal misfiled between families moves tasks between denominators invisibly;
- **the pairing breaks.** All eight cells ran the same fixed seeds, so the gap is taken
  within a seed, and the change in gap is taken within a seed too.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.reset_free_cycle_budget import ResetFreeCycleBudget

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"


def _sweep(*, trash_solved: int, recycling_solved: int, empty_solved: int) -> list[dict]:
    """One evaluation sweep with the domain's real 14 / 14 / 2 composition."""
    outcomes = []
    for index in range(14):
        outcomes.append({"task_index": index, "goal": _TRASH, "solved": index < trash_solved})
    for index in range(14):
        outcomes.append({
            "task_index": 14 + index,
            "goal": _RECYCLING,
            "solved": index < recycling_solved,
        })
    for index in range(2):
        outcomes.append({"task_index": 28 + index, "goal": _EMPTY, "solved": index < empty_solved})
    return outcomes


def _write_run(
    *,
    directory: Path,
    seed: int,
    level: tuple[int, int, int],
    num_cycles: int,
    num_practice_resets: int,
) -> None:
    """One run flat at `level`, with `num_cycles` cycles -- so `num_cycles + 1`
    evaluation checkpoints, which is the shape the harness really writes."""
    run = directory / str(seed)
    run.mkdir(parents=True)
    breakdowns = [
        {
            "num_online_transitions": index * 150,
            "outcomes": _sweep(
                trash_solved=level[0], recycling_solved=level[1], empty_solved=level[2]
            ),
        }
        for index in range(num_cycles + 1)
    ]
    windows = [
        {
            "MoveRoom": {"num_attempts": 100, "num_successes": 100},
            "PickupTrash": {"num_attempts": 2, "num_successes": 2},
            "ThrowTrash": {"num_attempts": 3, "num_successes": 3},
        }
        for _ in range(num_cycles)
    ]
    windows.append({"MoveRoom": {"num_attempts": 0, "num_successes": 0}})
    run.joinpath("stats.json").write_text(
        json.dumps({
            "breakdowns": breakdowns,
            "num_practice_resets": num_practice_resets,
            "practice_outcomes_per_cycle": windows,
            "task_name": "default",
        })
    )


def _write_arm(
    *,
    root: Path,
    budget: str,
    ledge: str,
    policy: str,
    levels: list[tuple[int, int, int]],
) -> Path:
    directory = root / f"{budget}-{ledge}-{policy}"
    num_cycles = ResetFreeCycleBudget.cycles_for(budget=budget)
    for seed, level in enumerate(levels):
        _write_run(
            directory=directory,
            seed=seed,
            level=level,
            num_cycles=num_cycles,
            num_practice_resets=num_cycles if policy == "scheduled" else 0,
        )
    return directory


def _cube(*, root: Path) -> dict[tuple[str, str, str], Path]:
    """All eight cells, two seeds each. The one-way gap is 8 TRASH per seed at 1x and
    2 TRASH per seed at 10x, so the gap closes by 6 per seed -- the shape this
    experiment is looking for. The two-way cells are level at both budgets."""
    levels = {
        ("1x", "one-way", "scheduled"): [(12, 3, 2), (12, 3, 2)],
        ("1x", "one-way", "never"): [(4, 2, 2), (4, 2, 2)],
        ("10x", "one-way", "scheduled"): [(14, 5, 2), (14, 5, 2)],
        ("10x", "one-way", "never"): [(12, 4, 2), (12, 4, 2)],
        ("1x", "two-way", "scheduled"): [(14, 14, 2), (14, 14, 2)],
        ("1x", "two-way", "never"): [(14, 14, 2), (14, 14, 2)],
        ("10x", "two-way", "scheduled"): [(14, 14, 2), (14, 14, 2)],
        ("10x", "two-way", "never"): [(14, 14, 2), (14, 14, 2)],
    }
    return {
        key: _write_arm(root=root, budget=key[0], ledge=key[1], policy=key[2], levels=value)
        for key, value in levels.items()
    }


# ------------------------------------------------------------------ the cube


def test_a_missing_cell_raises_rather_than_drawing_the_comparisons_it_still_can(
    *,
    tmp_path: Path,
) -> None:
    """ "Did the gap close?" is a difference of differences. With a cell absent it is
    undefined, and drawing the halves that survive would read as a result."""
    cells = _cube(root=tmp_path)
    del cells[("10x", "one-way", "never")]
    with pytest.raises(ValueError, match="10x.*one-way.*never"):
        ResetFreeCycleBudget.load_cells(directories=cells)


def test_all_eight_cells_load_and_keep_their_budget_ledge_and_policy_identity(
    *,
    tmp_path: Path,
) -> None:
    loaded = ResetFreeCycleBudget.load_cells(directories=_cube(root=tmp_path))
    assert set(loaded) == set(ResetFreeCycleBudget.cells())
    assert len(loaded) == 8


# -------------------------------------------------------------- budget integrity


def test_a_sweep_at_the_wrong_cycle_budget_raises(*, tmp_path: Path) -> None:
    """The two budgets differ only in `--num-cycles`, so nothing in a directory name
    prevents the 1x sweep being read as the 10x one. Cycles are counted out of the run's
    own breakdowns."""
    cells = _cube(root=tmp_path)
    cells[("10x", "two-way", "scheduled")] = cells[("1x", "two-way", "scheduled")]
    with pytest.raises(ValueError, match="cycles"):
        ResetFreeCycleBudget.load_cells(directories=cells)


def test_the_manipulation_check_scales_with_the_budget(*, tmp_path: Path) -> None:
    """A 10x `scheduled` run records 100 practice resets, not 10. A check hard-coded to
    10 would reject every correct 10x run; one hard-coded to pass would accept a
    mislabelled arm."""
    cells = _cube(root=tmp_path)
    stats_path = cells[("10x", "one-way", "scheduled")] / "0" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["num_practice_resets"] = 10
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="num_practice_resets"):
        ResetFreeCycleBudget.load_cells(directories=cells)


def test_a_never_run_that_recorded_practice_resets_raises_at_either_budget(
    *,
    tmp_path: Path,
) -> None:
    cells = _cube(root=tmp_path)
    stats_path = cells[("10x", "two-way", "never")] / "1" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["num_practice_resets"] = 100
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="num_practice_resets"):
        ResetFreeCycleBudget.load_cells(directories=cells)


# ------------------------------------------------------------------ denominators


def test_family_denominators_are_the_domain_composition_pooled_over_seeds(
    *,
    tmp_path: Path,
) -> None:
    """14 TRASH / 14 RECYCLING / 2 EMPTY per seed, so two seeds pool to 28 / 28 / 4."""
    loaded = ResetFreeCycleBudget.load_cells(directories=_cube(root=tmp_path))
    cell = loaded[("10x", "one-way", "never")]
    assert ResetFreeCycleBudget.pooled_final(cell=cell, family="TRASH") == (24, 28)
    assert ResetFreeCycleBudget.pooled_final(cell=cell, family="RECYCLING") == (8, 28)
    assert ResetFreeCycleBudget.pooled_final(cell=cell, family=None) == (36, 60)


def test_a_composition_that_is_not_fourteen_fourteen_two_raises(*, tmp_path: Path) -> None:
    cells = _cube(root=tmp_path)
    stats_path = cells[("1x", "one-way", "never")] / "0" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for breakdown in stats["breakdowns"]:
        breakdown["outcomes"][0]["goal"] = _RECYCLING
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="composition"):
        ResetFreeCycleBudget.load_cells(directories=cells)


# ------------------------------------------------------------------ the pairing


def test_the_gap_is_taken_within_a_seed(*, tmp_path: Path) -> None:
    """`scheduled` minus `never` at the final checkpoint, seed by seed. Here both seeds
    sit at 12 against 4 TRASH, so both gaps are 8."""
    loaded = ResetFreeCycleBudget.load_cells(directories=_cube(root=tmp_path))
    gaps = ResetFreeCycleBudget.paired_gaps(
        cells=loaded, budget="1x", ledge="one-way", family="TRASH"
    )
    assert gaps == [8.0, 8.0]


def test_the_change_in_gap_is_itself_taken_within_a_seed(*, tmp_path: Path) -> None:
    """The starvation question is whether the gap shrinks, so the quantity tested is
    `gap_at_1x - gap_at_10x` **per seed**. Here 8 - 2 = 6 for both seeds."""
    loaded = ResetFreeCycleBudget.load_cells(directories=_cube(root=tmp_path))
    changes = ResetFreeCycleBudget.paired_gap_changes(cells=loaded, ledge="one-way", family="TRASH")
    assert changes == [6.0, 6.0]


def test_a_gap_that_does_not_move_survives_as_zeros(*, tmp_path: Path) -> None:
    """The two-way cells are level at both budgets. "Every seed moved by exactly zero"
    is a finding, and it is only visible if ties are kept rather than dropped."""
    loaded = ResetFreeCycleBudget.load_cells(directories=_cube(root=tmp_path))
    assert ResetFreeCycleBudget.paired_gap_changes(cells=loaded, ledge="two-way", family=None) == [
        0.0,
        0.0,
    ]


# ------------------------------------------------------------------ presentation


def test_effective_attempts_count_only_pile_reaching_skills() -> None:
    """A stranded robot walks and presses buttons all period. Counting `MoveRoom` would
    report it as busy, which is the opposite of the measurement's purpose."""
    windows = [
        {
            "MoveRoom": {"num_attempts": 140, "num_successes": 140},
            "PressTrash": {"num_attempts": 8, "num_successes": 8},
            "PickupTrash": {"num_attempts": 1, "num_successes": 1},
            "ThrowTrash": {"num_attempts": 1, "num_successes": 1},
        },
        {"MoveRoom": {"num_attempts": 150, "num_successes": 150}},
        {"MoveRoom": {"num_attempts": 0, "num_successes": 0}},
    ]
    assert ResetFreeCycleBudget.cumulative_effective_attempts(
        windows=windows, num_checkpoints=3
    ) == [0, 2, 2]


def test_starved_cycles_are_counted_as_x_over_y_not_as_a_rate(*, tmp_path: Path) -> None:
    """A cycle in which nothing pile-reaching was attempted. The fixture attempts 5 every
    cycle, so no cycle is starved -- and the denominator is every cycle of every seed."""
    loaded = ResetFreeCycleBudget.load_cells(directories=_cube(root=tmp_path))
    starved, total = ResetFreeCycleBudget.starved_cycles(cell=loaded[("10x", "one-way", "never")])
    assert (starved, total) == (0, 200)


def test_counts_are_formatted_as_x_over_y_never_a_bare_percentage() -> None:
    assert ResetFreeCycleBudget.format_count(solved=127, total=140) == "127/140"
    assert "%" not in ResetFreeCycleBudget.format_count(solved=2, total=2)


def test_the_two_budgets_are_distinguished_by_more_than_hue() -> None:
    """Budget is carried by linestyle and policy by colour, so neither identity rests on
    hue alone in a figure that has four lines per panel."""
    styles = {
        (budget, policy): ResetFreeCycleBudget.style(budget=budget, policy=policy)
        for budget in ("1x", "10x")
        for policy in ("scheduled", "never")
    }
    assert len({style for style in styles.values()}) == 4
    assert styles[("1x", "scheduled")][1] != styles[("10x", "scheduled")][1]
    assert styles[("1x", "scheduled")][0] != styles[("1x", "never")][0]


def test_cycles_for_pins_the_two_budgets_this_experiment_ran() -> None:
    """10 cycles is the merged 1x protocol; 100 is the 10x arm. Both are asserted against
    each run's own breakdowns, so a wrong number here fails loudly rather than silently
    reading the wrong sweep."""
    assert ResetFreeCycleBudget.cycles_for(budget="1x") == 10
    assert ResetFreeCycleBudget.cycles_for(budget="10x") == 100
