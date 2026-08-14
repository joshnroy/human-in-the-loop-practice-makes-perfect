"""Covers the reference-arm comparison rule: three arms in one tree, keyed by
`(policy, method)` rather than by seed alone.

Built from synthetic curves with a *known* verdict, for the same reason the sibling
`test_tossing3d_reset_free_arms.py` is: a test that read the real sweep would assert
whatever the run happened to do, which is not a check on the rule.

The one that matters most is
`test_two_methods_under_one_policy_are_not_collapsed_into_one_arm`. This experiment is
the first on this domain to put two *methods* under a single policy directory, and the
sibling loader keys purely on the containing directory's name -- so pointed at
`scheduled/` it would read `random-skills/3` and `skill-oracle/3` into the same slot and
report whichever it globbed last. That collision is invisible for as long as the two arms
agree, and the whole point of a ceiling arm is that it does not.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_reference_arms import (
    WINDOW,
    Tossing3DReferenceArms,
)

NUM_TASKS = 10


def _write_run(*, run_dir: Path, solved_per_sweep: list[int], steps_per_cycle: int = 4) -> None:
    """One run's `stats.json`, in the real schema's shape.

    `steps_per_cycle` defaults to a robot that keeps practising; pass `0` for one that
    strands immediately.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluations = [
        [index * steps_per_cycle, solved, NUM_TASKS]
        for index, solved in enumerate(solved_per_sweep)
    ]
    (run_dir / "stats.json").write_text(
        json.dumps({"evaluations": evaluations, "practice_outcomes_per_cycle": []})
    )


def _flat(*, level: int, num_sweeps: int = 101) -> list[int]:
    return [level] * num_sweeps


def test_load_runs_keys_by_seed(*, tmp_path: Path) -> None:
    for seed in (0, 1, 2):
        _write_run(
            run_dir=tmp_path / "scheduled" / "random-skills" / str(seed),
            solved_per_sweep=_flat(level=seed),
        )
    runs = Tossing3DReferenceArms.load_runs(
        results_root=tmp_path, policy="scheduled", method="random-skills"
    )
    assert sorted(runs) == [0, 1, 2]
    assert runs[2][-1][1] == 2


def test_load_runs_ignores_directories_that_are_not_seeds(*, tmp_path: Path) -> None:
    _write_run(
        run_dir=tmp_path / "scheduled" / "random-skills" / "0", solved_per_sweep=_flat(level=3)
    )
    stray = tmp_path / "scheduled" / "random-skills" / "notaseed"
    _write_run(run_dir=stray, solved_per_sweep=_flat(level=9))
    runs = Tossing3DReferenceArms.load_runs(
        results_root=tmp_path, policy="scheduled", method="random-skills"
    )
    assert sorted(runs) == [0]


def test_two_methods_under_one_policy_are_not_collapsed_into_one_arm(*, tmp_path: Path) -> None:
    """The collision this loader exists to avoid -- see the module docstring."""
    for seed in range(3):
        _write_run(
            run_dir=tmp_path / "scheduled" / "random-skills" / str(seed),
            solved_per_sweep=_flat(level=2),
        )
        _write_run(
            run_dir=tmp_path / "scheduled" / "skill-oracle" / str(seed),
            solved_per_sweep=_flat(level=10),
        )
    random_skills = Tossing3DReferenceArms.load_runs(
        results_root=tmp_path, policy="scheduled", method="random-skills"
    )
    oracle = Tossing3DReferenceArms.load_runs(
        results_root=tmp_path, policy="scheduled", method="skill-oracle"
    )
    assert sorted(random_skills) == sorted(oracle) == [0, 1, 2]
    assert Tossing3DReferenceArms.late_scores(curves=random_skills) == {0: 2.0, 1: 2.0, 2: 2.0}
    assert Tossing3DReferenceArms.late_scores(curves=oracle) == {0: 10.0, 1: 10.0, 2: 10.0}


def test_a_single_sweep_reference_arm_scores_that_sweep(*, tmp_path: Path) -> None:
    """`skill-oracle` runs `num_cycles=0`, so it has exactly one evaluation sweep. The
    late window must not require `WINDOW` of them and must not pad."""
    _write_run(
        run_dir=tmp_path / "scheduled" / "skill-oracle" / "0",
        solved_per_sweep=[9],
        steps_per_cycle=0,
    )
    runs = Tossing3DReferenceArms.load_runs(
        results_root=tmp_path, policy="scheduled", method="skill-oracle"
    )
    assert len(runs[0]) == 1 < WINDOW
    assert Tossing3DReferenceArms.late_scores(curves=runs) == {0: 9.0}


def test_a_real_difference_is_not_reported_as_a_null_result() -> None:
    left = {seed: 7.0 for seed in range(10)}
    right = {seed: 1.0 + 0.1 * seed for seed in range(10)}
    result = Tossing3DReferenceArms.compare(left=left, right=right)
    assert result.num_seeds == 10
    assert result.num_right_lower == 10
    assert result.num_right_higher == 0
    assert result.mean_difference < 0
    assert result.p_value < 0.01
    assert result.is_null is False


def test_no_difference_is_reported_as_a_null_result_with_an_mde() -> None:
    left = {seed: 2.0 + 0.3 * (seed % 3) for seed in range(10)}
    right = {seed: left[seed] + (0.1 if seed % 2 else -0.1) for seed in range(10)}
    result = Tossing3DReferenceArms.compare(left=left, right=right)
    assert result.is_null is True
    assert result.p_value > 0.05
    # The MDE is what separates "no effect" from "no power", so a null result must
    # always carry a finite one.
    assert 0.0 < result.minimum_detectable_effect < float("inf")


def test_comparison_pairs_only_on_shared_seeds() -> None:
    left = {seed: 5.0 for seed in range(10)}
    right = {seed: 3.0 for seed in range(7)}
    result = Tossing3DReferenceArms.compare(left=left, right=right)
    assert result.seeds == list(range(7))
    assert result.num_seeds == 7


def test_comparison_refuses_when_no_seeds_are_shared() -> None:
    with pytest.raises(ValueError, match="no shared seeds"):
        Tossing3DReferenceArms.compare(left={0: 1.0}, right={5: 1.0})


def test_pooled_reports_a_count_against_the_seed_total() -> None:
    scores = {0: 1.2, 1: 2.4, 2: 0.6}
    numerator, denominator = Tossing3DReferenceArms.pooled(scores=scores, num_total=NUM_TASKS)
    # approx, not ==: each seed's contribution is a window mean, so the numerator is a
    # sum of floats and exact equality would be asserting IEEE-754 rounding.
    assert numerator == pytest.approx(4.2)
    assert denominator == 30
