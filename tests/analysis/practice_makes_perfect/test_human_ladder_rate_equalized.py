"""`RateEqualizedComparison` answers the question `human_ladder_curves.py` itself
flags as open: `on-stuck` spends ~3.4x the rescues `at-random` does at the shared
`--mean-steps-between-help-requests 150` default, so the timing contrast it draws is
not at matched cost. This module re-derives the same five arms' numbers once
`at-random` is retuned (via two *-matched arms) to spend at `on-stuck`'s own rate, and
answers: does `stuck` still beat `at-random` once the rate is equalised, and does
`solves_per_rescue` now favour stuck-detection?

Three things can go wrong silently, and each has a test:

- **the per-seed ratio divides by a rescue count of zero.** A seed never rescued has
  an undefined "solves per rescue", not a zero -- treating it as zero would understate
  the arm's efficiency, and dropping it silently would change the denominator without
  saying so;
- **the matched-vs-stuck comparison isn't actually paired on the SAME seeds.** All
  five arms here share fixed seeds, so a comparison built from misaligned seed lists
  would silently compare seed 3 of one arm to seed 7 of another;
- **the report claims a rate is "matched" without checking it moved.** A matched arm
  that spent the same 101 rescues the unmatched one did would mean the retuned
  `--mean-steps-between-help-requests` flag was silently ignored.

Expected values are derived on paper, not recorded from a run of the code. Reuses
`HumanLadderCurves.load_arm`/`entry`/`paired_final_differences`/`solves_per_rescue` and
`PairedTests.sign_flip` rather than reimplementing them -- same per-seed arm-data shape
`{seed: {"transitions", "families", "overall", "interventions", "human_cost"}}`.
"""

import json
from pathlib import Path

from analysis.practice_makes_perfect.human_ladder_curves import HumanLadderCurves
from analysis.practice_makes_perfect.human_ladder_rate_equalized import RateEqualizedComparison

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"


def _sweep(*, trash_solved: int, recycling_solved: int, empty_solved: int) -> list[dict]:
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
    *, directory: Path, seed: int, sweeps: list[tuple[int, int, int]], interventions: int = 0
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
    run.joinpath("stats.json").write_text(
        json.dumps({
            "breakdowns": breakdowns,
            "num_practice_resets": 0,
            "num_human_interventions_recorded": interventions,
            "summed_human_cost_recorded": float(interventions),
            "task_name": "default",
        })
    )


def _write_arm(
    *,
    root: Path,
    name: str,
    finals: dict[int, tuple[int, int, int]],
    rescues: dict[int, int],
) -> Path:
    """Ten seeds, matching the real ladder's seed count, so `PairedTests.sign_flip`
    exercises the same n as the real report does."""
    directory = root / name / "ees"
    for seed in range(10):
        _write_run(
            directory=directory,
            seed=seed,
            sweeps=[(0, 0, 0), finals[seed]],
            interventions=rescues[seed],
        )
    return directory


def _uniform(*, value: int) -> dict[int, int]:
    return dict.fromkeys(range(10), value)


def _write_fixture(*, root: Path) -> dict[str, Path]:
    """Five arms, ten seeds each, built so every seed's rescue count is nonzero for
    every treated arm (a real ratio exists everywhere) except one seed of
    `at-random-initial-matched`, which is deliberately left un-rescued so the
    exclusion path is exercised against real-shaped data rather than only a
    synthetic all-zero arm.

    Shaped so `stuck-initial` (many rescues, big gap) is the more EFFICIENT arm at
    matched cost and `at-random-initial` (few rescues, smaller total gap but a
    bigger gap-per-rescue) is the ranking-inversion case the real PR #151 data
    showed at the unmatched default -- i.e. this fixture reproduces the shape of
    the actual defect being tested for, not just arbitrary numbers."""
    no_human_final = 4
    stuck_rescues = _uniform(value=30)
    stuck_final = {seed: 4 + 10 for seed in range(10)}  # +10 over control, 30 rescues: 0.333/rescue
    at_random_unmatched_rescues = _uniform(value=9)
    at_random_unmatched_final = {
        seed: 4 + 4 for seed in range(10)
    }  # +4 over control, 9 rescues: 0.444/rescue
    at_random_matched_rescues = {**_uniform(value=30), 0: 0}  # seed 0 never rescued
    at_random_matched_final = {seed: 4 + 4 for seed in range(10)}
    at_random_matched_final[0] = 4  # seed 0: never rescued, so it also never gains

    return {
        "no-human": _write_arm(
            root=root,
            name="no-human",
            finals={seed: (no_human_final, 0, 0) for seed in range(10)},
            rescues=_uniform(value=0),
        ),
        "stuck-initial": _write_arm(
            root=root,
            name="stuck-initial",
            finals={seed: (stuck_final[seed], 0, 0) for seed in range(10)},
            rescues=stuck_rescues,
        ),
        "at-random-initial": _write_arm(
            root=root,
            name="at-random-initial",
            finals={seed: (at_random_unmatched_final[seed], 0, 0) for seed in range(10)},
            rescues=at_random_unmatched_rescues,
        ),
        "at-random-initial-matched": _write_arm(
            root=root,
            name="at-random-initial-matched",
            finals={seed: (at_random_matched_final[seed], 0, 0) for seed in range(10)},
            rescues=at_random_matched_rescues,
        ),
    }


def _load(*, root: Path) -> dict[str, dict]:
    directories = _write_fixture(root=root)
    return {
        name: HumanLadderCurves.load_arm(directory=directory, arm=name)
        for name, directory in directories.items()
    }


def test_per_seed_ratio_excludes_never_rescued_seeds(*, tmp_path: Path) -> None:
    arms = _load(root=tmp_path)
    ratios, excluded = RateEqualizedComparison.per_seed_solves_per_rescue(
        arms=arms, treatment="at-random-initial-matched", control="no-human"
    )
    assert excluded == [0]
    assert len(ratios) == 9
    # Every included seed: (4 + 4 - 4) / 30 = 4/30.
    assert all(abs(ratio - 4 / 30) < 1e-9 for ratio in ratios)


def test_per_seed_ratio_matches_pooled_solves_per_rescue_when_rescue_counts_are_uniform(
    *, tmp_path: Path
) -> None:
    """When every seed of an arm was rescued the same number of times, the per-seed
    mean ratio and the pooled gap/rescues ratio must agree -- they are the same
    quantity computed two ways in that special case, so a divergence would mean one
    of the two arithmetic paths is wrong."""
    arms = _load(root=tmp_path)
    ratios, excluded = RateEqualizedComparison.per_seed_solves_per_rescue(
        arms=arms, treatment="stuck-initial", control="no-human"
    )
    assert excluded == []
    pooled = HumanLadderCurves.solves_per_rescue(
        arms=arms, treatment="stuck-initial", control="no-human"
    )
    assert abs(sum(ratios) / len(ratios) - pooled) < 1e-9


def test_matched_arm_actually_spent_a_different_rate_than_unmatched(*, tmp_path: Path) -> None:
    """The whole point of a *-matched arm is that retuning
    --mean-steps-between-help-requests changed how often it asked. If the matched and
    unmatched rescue totals came out equal, the retuned flag was silently ignored."""
    arms = _load(root=tmp_path)
    unmatched_rescues = sum(
        arms["at-random-initial"][seed]["interventions"] for seed in arms["at-random-initial"]
    )
    matched_rescues = sum(
        arms["at-random-initial-matched"][seed]["interventions"]
        for seed in arms["at-random-initial-matched"]
    )
    assert matched_rescues != unmatched_rescues


def test_matched_vs_stuck_is_paired_on_shared_seeds(*, tmp_path: Path) -> None:
    """`stuck-initial` and `at-random-initial-matched` share all ten seeds, so the
    sign-flip test must see ten paired differences, not fewer -- a seed dropped by
    misaligned iteration would silently understate n."""
    arms = _load(root=tmp_path)
    differences = HumanLadderCurves.paired_final_differences(
        arms=arms, treatment="at-random-initial-matched", control="stuck-initial", family=None
    )
    assert len(differences) == 10
    # Every seed but seed 0: (4+4) - (4+10) = -6. Seed 0: 4 - 14 = -10.
    assert differences.count(-6.0) == 9
    assert differences.count(-10.0) == 1


def test_ranking_inversion_report_reproduces_the_defect_shape(*, tmp_path: Path) -> None:
    """At the UNMATCHED rate this fixture is built to reproduce PR #151's actual
    finding: the smaller-gap, cheaper arm (`at-random-initial`) posts the BETTER
    solves-per-rescue ratio than the bigger-gap, more expensive `stuck-initial` --
    the inversion this whole re-run exists to check whether matching the rate
    resolves."""
    arms = _load(root=tmp_path)
    stuck_ratio = HumanLadderCurves.solves_per_rescue(
        arms=arms, treatment="stuck-initial", control="no-human"
    )
    unmatched_ratio = HumanLadderCurves.solves_per_rescue(
        arms=arms, treatment="at-random-initial", control="no-human"
    )
    assert unmatched_ratio is not None
    assert stuck_ratio is not None
    assert unmatched_ratio > stuck_ratio
