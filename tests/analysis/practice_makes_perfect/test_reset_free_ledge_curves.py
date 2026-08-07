"""`ResetFreeLedgeCurves` draws the reset-policy A/B under both ledge conditions on one
axes per goal family, so what is pinned here is the arithmetic that turns four sweeps
into those three figures -- not matplotlib.

The domain is `tossingroom` throughout; the manipulation is the ledge
and the arms are the reset policy. Four things can go wrong silently and each has a test:

- **the four arms stop being a square.** Three arms cannot express "the gap within a
  ledge condition", which is the only comparison this experiment makes, so a missing arm
  must raise rather than draw the comparisons it still can;
- **a family's denominator drifts.** TRASH and RECYCLING are 14 each per seed and EMPTY
  is 2, so a goal misfiled between families moves tasks between denominators invisibly.
  The composition is asserted, not assumed;
- **the manipulation check stops checking.** `num_practice_resets` is a measurement of
  what happened rather than a restatement of the flag, so a `never` run reporting resets
  means the arms are not what their names say;
- **the pairing breaks.** All four arms ran the same fixed seeds, so per-seed
  differences must stay aligned seed-to-seed; an unpaired reading throws that away.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.reset_free_ledge_curves import ResetFreeLedgeCurves

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
    sweeps: list[tuple[int, int, int]],
    num_practice_resets: int,
) -> None:
    run = directory / str(seed)
    run.mkdir(parents=True)
    breakdowns = [
        {
            "num_online_transitions": index * 150,
            "outcomes": _sweep(trash_solved=trash, recycling_solved=recycling, empty_solved=empty),
        }
        for index, (trash, recycling, empty) in enumerate(sweeps)
    ]
    # One practice window per cycle plus the trailing empty one the harness records.
    # `MoveRoom` is present throughout so the "effective" filter has something to reject.
    windows = [
        {
            "MoveRoom": {"num_attempts": 100, "num_successes": 100},
            "PickupTrash": {"num_attempts": 2, "num_successes": 2},
            "ThrowTrash": {"num_attempts": 3, "num_successes": 3},
        }
        for _ in range(len(sweeps) - 1)
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
    ledge: str,
    policy: str,
    sweeps: list[tuple[int, int, int]],
    num_seeds: int = 2,
) -> Path:
    directory = root / f"{ledge}-{policy}"
    for seed in range(num_seeds):
        _write_run(
            directory=directory,
            seed=seed,
            sweeps=sweeps,
            num_practice_resets=10 if policy == "scheduled" else 0,
        )
    return directory


def _square(*, root: Path) -> dict[tuple[str, str], Path]:
    """All four arms present, each flat at a distinct per-family level."""
    levels = {
        ("one-way", "scheduled"): (14, 3, 2),
        ("one-way", "never"): (7, 2, 2),
        ("two-way", "scheduled"): (14, 14, 2),
        ("two-way", "never"): (13, 14, 2),
    }
    return {
        key: _write_arm(root=root, ledge=key[0], policy=key[1], sweeps=[value, value])
        for key, value in levels.items()
    }


# ------------------------------------------------------------------ the square


def test_a_missing_arm_raises_rather_than_drawing_the_comparisons_it_still_can(
    *,
    tmp_path: Path,
) -> None:
    """Three arms cannot express a within-ledge gap under both ledge conditions. A
    report that silently drew the one comparison it could would read as a result."""
    arms = _square(root=tmp_path)
    del arms[("two-way", "never")]
    with pytest.raises(ValueError, match="two-way.*never"):
        ResetFreeLedgeCurves.load_arms(directories=arms)


def test_all_four_arms_load_and_keep_their_ledge_and_policy_identity(*, tmp_path: Path) -> None:
    loaded = ResetFreeLedgeCurves.load_arms(directories=_square(root=tmp_path))
    assert set(loaded) == {
        ("one-way", "scheduled"),
        ("one-way", "never"),
        ("two-way", "scheduled"),
        ("two-way", "never"),
    }


# ------------------------------------------------------------------ denominators


def test_family_denominators_are_the_domain_composition_pooled_over_seeds(
    *,
    tmp_path: Path,
) -> None:
    """14 TRASH / 14 RECYCLING / 2 EMPTY per seed, so two seeds pool to 28 / 28 / 4 --
    and the figures' `x/140` axis is that same arithmetic at ten seeds."""
    loaded = ResetFreeLedgeCurves.load_arms(directories=_square(root=tmp_path))
    pooled = ResetFreeLedgeCurves.pooled_curve(arm=loaded[("one-way", "scheduled")], family="TRASH")
    assert pooled[0][1] == 28
    recycling = ResetFreeLedgeCurves.pooled_curve(
        arm=loaded[("one-way", "scheduled")], family="RECYCLING"
    )
    assert recycling[0][1] == 28
    overall = ResetFreeLedgeCurves.pooled_curve(arm=loaded[("one-way", "scheduled")], family=None)
    assert overall[0][1] == 60


def test_a_composition_that_is_not_fourteen_fourteen_two_raises(*, tmp_path: Path) -> None:
    """A goal misfiled between families shows up here as a wrong denominator rather than
    as a plausible wrong rate."""
    arms = _square(root=tmp_path)
    stats_path = arms[("one-way", "never")] / "0" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for breakdown in stats["breakdowns"]:
        breakdown["outcomes"][0]["goal"] = _RECYCLING
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="composition"):
        ResetFreeLedgeCurves.load_arms(directories=arms)


def test_pooled_solved_counts_are_summed_across_seeds_not_averaged(*, tmp_path: Path) -> None:
    """Both seeds solve 7/14 TRASH, so the arm pools to 14/28 -- not 7/14."""
    loaded = ResetFreeLedgeCurves.load_arms(directories=_square(root=tmp_path))
    pooled = ResetFreeLedgeCurves.pooled_curve(arm=loaded[("one-way", "never")], family="TRASH")
    assert pooled[0] == (14, 28)


# ------------------------------------------------------------ manipulation check


def test_a_never_run_that_recorded_practice_resets_raises(*, tmp_path: Path) -> None:
    """`num_practice_resets` is measured, so it is the check that the arms are what
    their names say. A `never` arm with resets in it is a mislabelled sweep."""
    arms = _square(root=tmp_path)
    stats_path = arms[("two-way", "never")] / "1" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["num_practice_resets"] = 10
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="num_practice_resets"):
        ResetFreeLedgeCurves.load_arms(directories=arms)


# ------------------------------------------------------------------ the pairing


def test_per_seed_differences_stay_aligned_seed_to_seed(*, tmp_path: Path) -> None:
    """The arms share seeds, so the difference is taken within a seed. Here seed 0 of
    `scheduled` differs from seed 0 of `never` by 14-7=7 TRASH, and likewise seed 1;
    a sorted-then-zipped reading would coincide here, so the test makes the arms
    differ per seed to catch it."""
    arms = _square(root=tmp_path)
    # Rewrite the one-way arms so each seed sits at its own level.
    for policy, levels in (
        ("scheduled", [(14, 3, 2), (10, 3, 2)]),
        ("never", [(4, 2, 2), (9, 2, 2)]),
    ):
        for seed, level in enumerate(levels):
            stats_path = arms[("one-way", policy)] / str(seed) / "stats.json"
            stats = json.loads(stats_path.read_text())
            for breakdown in stats["breakdowns"]:
                breakdown["outcomes"] = _sweep(
                    trash_solved=level[0], recycling_solved=level[1], empty_solved=level[2]
                )
            stats_path.write_text(json.dumps(stats))
    loaded = ResetFreeLedgeCurves.load_arms(directories=arms)
    differences = ResetFreeLedgeCurves.paired_final_differences(
        arms=loaded, ledge="one-way", family="TRASH"
    )
    # seed 0: 14 - 4 = 10;  seed 1: 10 - 9 = 1.
    assert differences == [10.0, 1.0]


def test_paired_differences_are_empty_of_meaning_when_arms_are_identical(
    *,
    tmp_path: Path,
) -> None:
    """All-zero differences must survive as zeros -- the two-way cell's headline is
    "9/10 seeds differ by exactly zero", which is only visible if ties are kept."""
    arms = _square(root=tmp_path)
    loaded = ResetFreeLedgeCurves.load_arms(directories=arms)
    differences = ResetFreeLedgeCurves.paired_final_differences(
        arms=loaded, ledge="two-way", family="RECYCLING"
    )
    assert differences == [0.0, 0.0]


# ------------------------------------------------------------------ presentation


def test_effective_attempts_count_only_pile_reaching_skills() -> None:
    """A stranded robot walks and presses buttons all period. Counting `MoveRoom` would
    report it as busy, which is the exact opposite of the measurement's purpose."""
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
    # Checkpoint 0 is before any practice; checkpoint i accumulates windows 0..i-1. The
    # second cycle attempted nothing effective, so the total must not move.
    assert ResetFreeLedgeCurves.cumulative_effective_attempts(
        windows=windows, num_checkpoints=3
    ) == [0, 2, 2]


def test_the_trailing_practice_window_is_not_attributed_to_a_checkpoint() -> None:
    """`practice_outcomes_per_cycle` carries one window past the last cycle. Folding it
    in would credit an arm with practice no checkpoint ever saw."""
    windows = [
        {"PickupTrash": {"num_attempts": 5, "num_successes": 5}},
        {"PickupTrash": {"num_attempts": 7, "num_successes": 7}},
        {"PickupTrash": {"num_attempts": 999, "num_successes": 999}},
    ]
    assert ResetFreeLedgeCurves.cumulative_effective_attempts(
        windows=windows, num_checkpoints=3
    ) == [0, 5, 12]


def test_mean_transitions_per_cycle_is_measured_not_assumed(*, tmp_path: Path) -> None:
    """This is the number a cycles axis would rest on. The fixture charges 150 per
    cycle over 1 cycle, matching the real sweeps -- and it is computed rather than
    asserted, because a period that ends early would legitimately lower it."""
    loaded = ResetFreeLedgeCurves.load_arms(directories=_square(root=tmp_path))
    assert ResetFreeLedgeCurves.mean_transitions_per_cycle(
        arm=loaded[("one-way", "never")]
    ) == pytest.approx(150.0)


def test_policy_display_names_never_leak_the_flag_value_into_a_label() -> None:
    """The `--practice-reset-policy` values stay `scheduled`/`never` in keys, paths and
    committed `config_snapshot.json`; only the display label changes."""
    scheduled = ResetFreeLedgeCurves.label(ledge="one-way", policy="scheduled")
    never = ResetFreeLedgeCurves.label(ledge="two-way", policy="never")
    assert scheduled == "one-way ledge, practice-session env resets"
    assert never == "two-way ledge, never env reset"
    assert "scheduled" not in scheduled
    # "never env reset" legitimately contains the word, so check the flag-ish forms.
    assert "`never`" not in never


def test_counts_are_formatted_as_x_over_y_never_a_bare_percentage() -> None:
    """A percentage hides the denominator, and EMPTY is 2 tasks per seed."""
    assert ResetFreeLedgeCurves.format_count(solved=127, total=140) == "127/140"
    assert "%" not in ResetFreeLedgeCurves.format_count(solved=2, total=2)


def test_every_arm_has_a_distinct_colour_and_the_policy_is_reinforced_by_linestyle() -> None:
    """Four colours, and the reset policy also carried by linestyle so identity never
    rests on hue alone."""
    styles = [
        ResetFreeLedgeCurves.style(ledge=ledge, policy=policy)
        for ledge, policy in ResetFreeLedgeCurves.arms()
    ]
    assert len({style[0] for style in styles}) == 4
    scheduled = {
        ResetFreeLedgeCurves.style(ledge=ledge, policy="scheduled")[1]
        for ledge in ("one-way", "two-way")
    }
    never = {
        ResetFreeLedgeCurves.style(ledge=ledge, policy="never")[1]
        for ledge in ("one-way", "two-way")
    }
    assert len(scheduled) == 1 and len(never) == 1 and scheduled != never
