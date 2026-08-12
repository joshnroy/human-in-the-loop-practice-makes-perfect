"""`ResetPolicyControl` is the reset-policy A/B drawn per goal family, across both of
Tossing Room's skill configurations. What is pinned here is the arithmetic that turns a
set of sweeps into paired per-family comparisons, not matplotlib.

Four things can go wrong silently, and each has a test:

- **a group is half-present.** This module reports a paired difference, so a
  `(config, ledge, budget)` with only one policy has nothing to report and must be
  dropped rather than half-drawn;
- **the pairing breaks.** The difference is `never - scheduled` *within a seed*, so a
  partially-complete sweep must pair the seeds both arms actually ran instead of zipping
  two lists of different length and silently comparing seed 7 against seed 9;
- **the unsplit rendering stops being read.** `--unsplit-skills` is a flag on the same
  domain with the same fixed 14/14/2 test set, so its goal strings have to classify into
  the same three families -- a regression there would move tasks between denominators;
- **the aggregate stops matching the figures.** The committed JSON exists so the log is
  re-derivable without the raw sweeps, which is only true while it carries exactly the
  per-family counts the figures and the report consume.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.reset_policy_control import ResetPolicyControl

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"

# The `--unsplit-skills` rendering of the same three families: one shared `ItemInBin`
# throw predicate and one shared `BinEmpty`, so only the bound object names the family.
_UNSPLIT_TRASH = "ItemInBin(trash, trash_bin)"
_UNSPLIT_RECYCLING = "ItemInBin(recycling, recycling_bin)"
_UNSPLIT_EMPTY = "BinEmpty(recycling_bin) & BinEmpty(trash_bin)"


def _sweep(*, trash: int, recycling: int, empty: int, unsplit: bool) -> list[dict]:
    """One evaluation sweep with the domain's real 14 / 14 / 2 composition."""
    trash_goal = _UNSPLIT_TRASH if unsplit else _TRASH
    recycling_goal = _UNSPLIT_RECYCLING if unsplit else _RECYCLING
    empty_goal = _UNSPLIT_EMPTY if unsplit else _EMPTY
    outcomes = []
    for index in range(14):
        outcomes.append({"task_index": index, "goal": trash_goal, "solved": index < trash})
    for index in range(14):
        outcomes.append({
            "task_index": 14 + index,
            "goal": recycling_goal,
            "solved": index < recycling,
        })
    for index in range(2):
        outcomes.append({"task_index": 28 + index, "goal": empty_goal, "solved": index < empty})
    return outcomes


def _write_cell(
    *,
    directory: Path,
    seeds: dict[int, tuple[int, int, int]],
    policy: str,
    num_cycles: int = 10,
    unsplit: bool = False,
) -> Path:
    """One cell, every seed flat at its own `(trash, recycling, empty)` level.

    `num_practice_resets` is derived from the policy and the cycle count rather than
    passed, because that is exactly the consistency `load_cell` checks: a `scheduled` run
    resets once per cycle and a `never` run never does.
    """
    for seed, level in seeds.items():
        run = directory / str(seed)
        run.mkdir(parents=True)
        breakdowns = [
            {
                "num_online_transitions": index * 150,
                "outcomes": _sweep(
                    trash=level[0], recycling=level[1], empty=level[2], unsplit=unsplit
                ),
            }
            for index in range(num_cycles + 1)
        ]
        # One practice window per cycle plus the trailing one the harness really writes.
        # `PickupTrash` is a pile-reaching skill, so these count as effective attempts and
        # every seed here reads as non-stuck -- the subgroup split is exercised by
        # `stuck_split`'s own tests, not re-litigated here.
        practice = [{"PickupTrash": {"num_attempts": 5, "num_successes": 5}}] * (num_cycles + 1)
        (run / "stats.json").write_text(
            json.dumps({
                "breakdowns": breakdowns,
                "num_practice_resets": num_cycles if policy == "scheduled" else 0,
                "practice_outcomes_per_cycle": practice,
            })
        )
    return directory


def _two_arm_cells(*, tmp_path: Path, unsplit: bool = False) -> dict:
    """A `split`/`one-way`/`1x` group where `never` is uniformly two TRASH worse."""
    scheduled = _write_cell(
        directory=tmp_path / "scheduled",
        seeds={seed: (12, 5, 2) for seed in range(4)},
        policy="scheduled",
        unsplit=unsplit,
    )
    never = _write_cell(
        directory=tmp_path / "never",
        seeds={seed: (10, 5, 2) for seed in range(4)},
        policy="never",
        unsplit=unsplit,
    )
    config = "unsplit" if unsplit else "split"
    return ResetPolicyControl.load(
        directories={
            (config, "one-way", "1x", "scheduled"): scheduled,
            (config, "one-way", "1x", "never"): never,
        }
    )


def test_a_group_missing_one_policy_is_dropped_rather_than_half_drawn(*, tmp_path: Path) -> None:
    """A lone arm has no paired difference to report, so its group must not appear at
    all -- drawing it would put a single-arm panel beside genuine comparisons."""
    scheduled = _write_cell(
        directory=tmp_path / "scheduled",
        seeds={seed: (12, 5, 2) for seed in range(4)},
        policy="scheduled",
    )
    paired_never = _write_cell(
        directory=tmp_path / "paired-never",
        seeds={seed: (10, 5, 2) for seed in range(4)},
        policy="never",
    )
    lonely = _write_cell(
        directory=tmp_path / "lonely",
        seeds={seed: (9, 5, 2) for seed in range(4)},
        policy="scheduled",
    )
    cells = ResetPolicyControl.load(
        directories={
            ("split", "one-way", "1x", "scheduled"): scheduled,
            ("split", "one-way", "1x", "never"): paired_never,
            ("split", "two-way", "1x", "scheduled"): lonely,
        }
    )
    assert ResetPolicyControl.groups(cells=cells) == [("split", "one-way", "1x")]


def test_the_paired_difference_is_taken_within_a_seed(*, tmp_path: Path) -> None:
    """`never - scheduled`, per seed, on TRASH: every seed is 10 against 12."""
    cells = _two_arm_cells(tmp_path=tmp_path)
    seeds, differences = ResetPolicyControl.paired_differences(
        cells=cells, group=("split", "one-way", "1x"), family="TRASH"
    )
    assert seeds == [0, 1, 2, 3]
    assert differences == [-2.0, -2.0, -2.0, -2.0]


def test_pairing_uses_only_the_seeds_both_arms_ran(*, tmp_path: Path) -> None:
    """A partially-complete sweep must not align seed 3 of one arm against seed 9 of the
    other. The intersection is what makes a mid-sweep partial figure honest rather than
    quietly wrong."""
    scheduled = _write_cell(
        directory=tmp_path / "scheduled",
        seeds={seed: (12, 5, 2) for seed in (0, 1, 2, 3)},
        policy="scheduled",
    )
    never = _write_cell(
        directory=tmp_path / "never",
        seeds={seed: (10, 5, 2) for seed in (0, 2)},
        policy="never",
    )
    cells = ResetPolicyControl.load(
        directories={
            ("split", "one-way", "1x", "scheduled"): scheduled,
            ("split", "one-way", "1x", "never"): never,
        }
    )
    seeds, differences = ResetPolicyControl.paired_differences(
        cells=cells, group=("split", "one-way", "1x"), family="TRASH"
    )
    assert seeds == [0, 2]
    assert differences == [-2.0, -2.0]


def test_the_unsplit_skills_rendering_reads_into_the_same_families(*, tmp_path: Path) -> None:
    """`--unsplit-skills` renames the predicates but not the test set, so the same
    14/14/2 composition must come back and the per-family counts must be unchanged."""
    cells = _two_arm_cells(tmp_path=tmp_path, unsplit=True)
    scheduled = cells[("unsplit", "one-way", "1x", "scheduled")]
    never = cells[("unsplit", "one-way", "1x", "never")]
    assert ResetPolicyControl.final_counts(cell=scheduled, family="TRASH") == (48, 56)
    assert ResetPolicyControl.final_counts(cell=never, family="TRASH") == (40, 56)
    assert ResetPolicyControl.final_counts(cell=scheduled, family="RECYCLING") == (20, 56)
    assert ResetPolicyControl.final_counts(cell=scheduled, family="EMPTY") == (8, 8)


def test_a_family_whose_arms_are_identical_reports_every_difference_as_zero(
    *,
    tmp_path: Path,
) -> None:
    """EMPTY is 2 tasks a seed and both arms solve both, every seed. That has to surface
    as "no inference", never as a null result that sounds like evidence of no effect."""
    cells = _two_arm_cells(tmp_path=tmp_path)
    _, differences = ResetPolicyControl.paired_differences(
        cells=cells, group=("split", "one-way", "1x"), family="EMPTY"
    )
    assert differences == [0.0, 0.0, 0.0, 0.0]


def test_the_aggregate_carries_the_counts_the_figures_are_built_from(*, tmp_path: Path) -> None:
    """The committed JSON is what keeps the log re-derivable without the raw sweeps, so
    it has to hold the per-seed per-checkpoint family counts, keyed by cell."""
    cells = _two_arm_cells(tmp_path=tmp_path)
    # Round-tripped through JSON deliberately: what has to stay re-derivable is the
    # committed *file*, and the in-memory form holds tuples that only become the lists a
    # reader will parse once serialised.
    aggregate = json.loads(json.dumps(ResetPolicyControl.aggregate(cells=cells)))
    assert set(aggregate) == {
        "split:one-way:1x:scheduled",
        "split:one-way:1x:never",
    }
    never = aggregate["split:one-way:1x:never"]["0"]
    assert never["transitions"][-1] == 1500
    assert never["families"]["TRASH"][-1] == [10, 14]
    assert never["overall"][-1] == [17, 30]


def test_parse_cells_rejects_a_key_that_is_not_four_parts() -> None:
    with pytest.raises(ValueError, match="four colon-separated parts"):
        ResetPolicyControl.parse_cells(raw=["split:one-way:scheduled=/tmp/x"])


def test_parse_cells_rejects_an_unknown_policy() -> None:
    """A typo here would otherwise reach `load_cell`, which would check
    `num_practice_resets` against a policy that does not exist."""
    with pytest.raises(ValueError, match="policy must be"):
        ResetPolicyControl.parse_cells(raw=["split:one-way:1x:sometimes=/tmp/x"])
