"""Covers the two-skill reader, and specifically its guard against the silent zero.

The reader exists because `tossing3d_ees_arms.py` and its siblings key on the *old*
three-skill names and read their per-skill tallies with `.get(name)`, so a two-skill
`stats.json` yields `0/0` everywhere and they report "never practiced" rather than
raising. That failure is confident and empty, which is worse than a crash.

Two tests carry the whole argument and the rest is plumbing:

- `test_the_old_three_skill_names_raise_rather_than_reporting_zero` -- the bug this
  module exists to prevent.
- `test_the_unparameterized_pick_is_not_reported_as_a_name_mismatch` -- the way a naive
  guard reintroduces it. `PickCube` declares `param_dim == 0`, so it has no sampler and
  *correctly* contributes nothing to any competence plot. A guard that fires on "this
  skill shows no learnable activity" would fire on the healthy case, get muted, and stop
  protecting anything.

Built from synthetic tallies with a known shape, because the property under test is
"does the guard reach the right verdict"; a test reading the real sweep would assert
whatever that sweep happened to record.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_two_skill_curves import (
    EXPECTED_SKILLS,
    LEARNING_ARM,
    SkillNameMismatchError,
    Tossing3DTwoSkillCurves,
)
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

NUM_TASKS = 10

PICK = "PickCube"
TOSS = "MoveToTossLocationAndToss"


def _tally(
    *,
    attempts: int,
    unparameterized: int = 0,
    unparameterized_successes: int = 0,
    informed: int = 0,
    informed_successes: int = 0,
    random_attempts: int = 0,
    random_successes: int = 0,
) -> dict[str, int]:
    """One `SkillPracticeTally` as `stats.json` stores it.

    `num_successes` is derived from the per-pool successes rather than passed, because
    `SkillPracticeTally` validates that the uninformative remainder (attempts and
    successes alike) is non-negative -- a hand-written total that does not decompose is
    rejected, which is the validator doing its job.
    """
    return {
        "num_attempts": attempts,
        "num_successes": unparameterized_successes + informed_successes + random_successes,
        "num_random_attempts": random_attempts,
        "num_random_successes": random_successes,
        "num_informed_attempts": informed,
        "num_informed_successes": informed_successes,
        "num_unparameterized_attempts": unparameterized,
        "num_unparameterized_successes": unparameterized_successes,
    }


def _write_run(
    *,
    run_dir: Path,
    solved_per_sweep: list[int],
    practice_outcomes: list[dict[str, dict[str, int]]] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluations": [[i * 4, solved, NUM_TASKS] for i, solved in enumerate(solved_per_sweep)],
        "practice_outcomes_per_cycle": practice_outcomes or [],
    }
    (run_dir / "stats.json").write_text(json.dumps(payload))


def _healthy_two_skill_cycle() -> dict[str, dict[str, int]]:
    """What a real two-skill cycle records: the pick executed with no sampler ever
    consultable, the composed toss executed with an informed draw behind it."""
    return {
        PICK: _tally(attempts=4, unparameterized=4, unparameterized_successes=4),
        TOSS: _tally(
            attempts=4, informed=3, informed_successes=2, random_attempts=1, random_successes=0
        ),
    }


def _three_skill_cycle() -> dict[str, dict[str, int]]:
    """The pre-migration domain, which is exactly what the guard has to catch."""
    return {
        "Pick": _tally(attempts=4, unparameterized=4, unparameterized_successes=4),
        "MoveToThrowPose": _tally(attempts=4, informed=4, informed_successes=3),
        "Toss": _tally(attempts=4, unparameterized=4, unparameterized_successes=1),
    }


def test_the_expected_names_are_the_domains_own_skill_names() -> None:
    """The constant is a copy of two strings that live in `src/`, so it can drift. This
    is what stops a future rename from recreating the silent zero in the new reader."""
    assert set(EXPECTED_SKILLS) == {
        Tossing3DSkills.PICK_CUBE.name,
        Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS.name,
    }


def test_the_expected_param_dims_are_the_domains_own_param_dims() -> None:
    """The guard's whole distinction rests on `PickCube` declaring no parameters and the
    composed toss declaring some. If either drifts, the guard classifies wrongly."""
    assert EXPECTED_SKILLS[Tossing3DSkills.PICK_CUBE.name] == Tossing3DSkills.PICK_CUBE.param_dim
    assert EXPECTED_SKILLS[Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS.name] == (
        Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS.param_dim
    )
    assert Tossing3DSkills.PICK_CUBE.param_dim == 0


def test_the_old_three_skill_names_raise_rather_than_reporting_zero(*, tmp_path: Path) -> None:
    """The defect this module exists to prevent: a three-skill results tree read by a
    two-skill reader must fail loudly, not return an empty plot."""
    _write_run(
        run_dir=tmp_path / LEARNING_ARM / "0",
        solved_per_sweep=[1, 5, 9],
        practice_outcomes=[_three_skill_cycle(), _three_skill_cycle()],
    )
    coverage = Tossing3DTwoSkillCurves.skill_coverage(results_root=tmp_path, method=LEARNING_ARM)
    with pytest.raises(SkillNameMismatchError) as excinfo:
        Tossing3DTwoSkillCurves.check_skill_names(coverage=coverage)
    message = str(excinfo.value)
    # The message has to name both sides, or the reader still has to go digging.
    assert PICK in message
    assert TOSS in message
    assert "MoveToThrowPose" in message


def test_the_unparameterized_pick_is_not_reported_as_a_name_mismatch(*, tmp_path: Path) -> None:
    """The critical subtlety. `PickCube` has `param_dim == 0`, so it contributes no
    learnable activity by construction. Conflating that with a name mismatch would make
    the healthy case fire the guard."""
    _write_run(
        run_dir=tmp_path / LEARNING_ARM / "0",
        solved_per_sweep=[1, 5, 9],
        practice_outcomes=[_healthy_two_skill_cycle(), _healthy_two_skill_cycle()],
    )
    coverage = Tossing3DTwoSkillCurves.skill_coverage(results_root=tmp_path, method=LEARNING_ARM)
    Tossing3DTwoSkillCurves.check_skill_names(coverage=coverage)  # must not raise

    by_name = {entry.skill_name: entry for entry in coverage}
    assert by_name[PICK].status == "unlearnable-by-construction"
    assert by_name[PICK].is_correctly_empty is True
    assert by_name[TOSS].status == "learnable"
    assert by_name[TOSS].is_correctly_empty is False


def test_a_skill_present_but_never_practiced_is_neither_of_those(*, tmp_path: Path) -> None:
    """A third state, kept separate from both: the names match, so nothing is broken,
    but there is genuinely no data. Folding it into either neighbour would either raise
    on a correct-but-idle run or silently call an empty run healthy."""
    _write_run(
        run_dir=tmp_path / LEARNING_ARM / "0",
        solved_per_sweep=[1, 5, 9],
        practice_outcomes=[{PICK: _tally(attempts=0), TOSS: _tally(attempts=0)}],
    )
    coverage = Tossing3DTwoSkillCurves.skill_coverage(results_root=tmp_path, method=LEARNING_ARM)
    Tossing3DTwoSkillCurves.check_skill_names(coverage=coverage)  # names are fine
    assert {entry.status for entry in coverage} == {"present-but-unpracticed"}


def test_a_declared_parameterized_skill_recording_only_unparameterized_draws_is_flagged(
    *, tmp_path: Path
) -> None:
    """Source says the composed toss has four parameters; data saying every attempt was
    unparameterized means the results predate the migration even though the name matches.
    A name-only guard would pass this."""
    _write_run(
        run_dir=tmp_path / LEARNING_ARM / "0",
        solved_per_sweep=[1, 5, 9],
        practice_outcomes=[
            {
                PICK: _tally(attempts=4, unparameterized=4),
                TOSS: _tally(attempts=4, unparameterized=4),
            }
        ],
    )
    coverage = Tossing3DTwoSkillCurves.skill_coverage(results_root=tmp_path, method=LEARNING_ARM)
    by_name = {entry.skill_name: entry for entry in coverage}
    assert by_name[TOSS].status == "contradicts-declared-param-dim"
    with pytest.raises(SkillNameMismatchError):
        Tossing3DTwoSkillCurves.check_skill_names(coverage=coverage)


def test_an_empty_results_tree_raises_rather_than_plotting_nothing(*, tmp_path: Path) -> None:
    """No runs at all is the other route to a confident empty plot."""
    coverage = Tossing3DTwoSkillCurves.skill_coverage(results_root=tmp_path, method=LEARNING_ARM)
    with pytest.raises(SkillNameMismatchError):
        Tossing3DTwoSkillCurves.check_skill_names(coverage=coverage)


def test_curves_are_loaded_per_seed(*, tmp_path: Path) -> None:
    _write_run(run_dir=tmp_path / LEARNING_ARM / "0", solved_per_sweep=[1, 5, 9])
    _write_run(run_dir=tmp_path / LEARNING_ARM / "3", solved_per_sweep=[0, 2, 4])
    curves = Tossing3DTwoSkillCurves.load_curves(results_root=tmp_path, method=LEARNING_ARM)
    assert sorted(curves) == [0, 3]
    assert [solved for _t, solved, _n in curves[0]] == [1, 5, 9]


def test_final_and_best_are_pooled_as_counts(*, tmp_path: Path) -> None:
    """`x/y`, never a percentage -- and best-ever pooled per seed, since a maximum taken
    after pooling would be a different (smaller) quantity."""
    _write_run(run_dir=tmp_path / LEARNING_ARM / "0", solved_per_sweep=[1, 10, 6])
    _write_run(run_dir=tmp_path / LEARNING_ARM / "1", solved_per_sweep=[2, 4, 8])
    curves = Tossing3DTwoSkillCurves.load_curves(results_root=tmp_path, method=LEARNING_ARM)
    assert Tossing3DTwoSkillCurves.pooled_final(curves=curves) == (14, 20)
    assert Tossing3DTwoSkillCurves.pooled_best(curves=curves) == (18, 20)


def test_regressed_seeds_report_the_drop_and_where_it_peaked(*, tmp_path: Path) -> None:
    """The headline shape: a seed that reached its best and ended below it."""
    _write_run(run_dir=tmp_path / LEARNING_ARM / "0", solved_per_sweep=[1, 10, 6])
    _write_run(run_dir=tmp_path / LEARNING_ARM / "1", solved_per_sweep=[2, 4, 8])
    curves = Tossing3DTwoSkillCurves.load_curves(results_root=tmp_path, method=LEARNING_ARM)
    regressed = Tossing3DTwoSkillCurves.regressed_seeds(curves=curves)
    assert [entry.seed for entry in regressed] == [0]
    assert regressed[0].best == 10
    assert regressed[0].final == 6
    assert regressed[0].peak_index == 1


def test_a_flat_reference_arm_is_summarised_as_one_level(*, tmp_path: Path) -> None:
    """`skill-oracle` writes a single evaluation, and is drawn as a horizontal line
    rather than a curve. The reader has to hand the plotter one number, not a series."""
    for seed in range(3):
        _write_run(run_dir=tmp_path / "skill-oracle" / str(seed), solved_per_sweep=[10])
    curves = Tossing3DTwoSkillCurves.load_curves(results_root=tmp_path, method="skill-oracle")
    assert Tossing3DTwoSkillCurves.pooled_final(curves=curves) == (30, 30)
    assert Tossing3DTwoSkillCurves.mean_final_per_seed(curves=curves) == 10.0


def test_best_versus_final_is_reported_for_every_arm(*, tmp_path: Path) -> None:
    """The non-learning arm's gap is the calibration for the learning arm's, so the
    reader has to return both rather than only the arm under test."""
    _write_run(run_dir=tmp_path / LEARNING_ARM / "0", solved_per_sweep=[1, 10, 6])
    _write_run(run_dir=tmp_path / "random-skills" / "0", solved_per_sweep=[1, 7, 3])
    curves_by_arm = {
        arm: Tossing3DTwoSkillCurves.load_curves(results_root=tmp_path, method=arm)
        for arm in (LEARNING_ARM, "random-skills")
    }
    gaps = Tossing3DTwoSkillCurves.best_versus_final(curves_by_arm=curves_by_arm)
    assert gaps == {LEARNING_ARM: [4], "random-skills": [4]}


def test_both_figures_are_written(*, tmp_path: Path) -> None:
    """A smoke test only -- it asserts the files exist, not what they look like."""
    for seed in range(3):
        _write_run(run_dir=tmp_path / LEARNING_ARM / str(seed), solved_per_sweep=[1, 10, 6])
        _write_run(run_dir=tmp_path / "random-skills" / str(seed), solved_per_sweep=[1, 4, 3])
        _write_run(run_dir=tmp_path / "skill-oracle" / str(seed), solved_per_sweep=[10])
    curves_by_arm = {
        arm: Tossing3DTwoSkillCurves.load_curves(results_root=tmp_path, method=arm)
        for arm in (LEARNING_ARM, "skill-oracle", "random-skills")
    }
    curve_path = tmp_path / "curves.png"
    bias_path = tmp_path / "bias.png"
    Tossing3DTwoSkillCurves.plot(curves_by_arm=curves_by_arm, output_path=curve_path)
    Tossing3DTwoSkillCurves.plot_selection_bias(curves_by_arm=curves_by_arm, output_path=bias_path)
    assert curve_path.exists()
    assert bias_path.exists()
