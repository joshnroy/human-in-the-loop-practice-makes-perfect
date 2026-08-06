"""Tests for the reset-free-practice A/B analysis, outcome side and practice side.

Four things on the outcome side can be wrong in ways that are invisible on inspection,
so all four are pinned against values worked out by hand rather than by running the
code:

1. **Goal-family classification.** The EMPTY goal names *both* bins
   (`RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)`), so a naive
   "does it mention recycling?" test swallows it and silently reports 16 RECYCLING /
   0 EMPTY -- plausible-looking numbers, not an error. The EMPTY-first rule order is
   the whole defence, and a test is the only thing that keeps it from being "tidied"
   into a dict lookup that happens to iterate the other way.
2. **The manipulation and composition checks**, which must RAISE. A check that reports
   a violation and then prints the report anyway is not a check.
3. **The exact paired p-value**, computed by enumeration.
4. **The MDE's degenerate case.** At 20/20 against 20/20 the normal-approximation
   formula returns 0.0pp, which reads as infinite sensitivity and means the opposite.
   `None` there is a deliberate behaviour, so it is pinned.

The practice side (`PracticeSideReport`) is pinned for a different reason. Its raw input
-- the per-period skill traces -- is ~550 KB and lives outside the repo, so the condensed
`2026-08-06-reset-free-practice-traces.json` is the only surviving record of what
practice actually *did*. Every number the experiment log quotes from it is asserted here
against the committed file, including the equal-budget check that makes the two arms'
compositions comparable at all (14900 practice attempts in each) and the attribution of
the `never` arm's residual late-run throws to the individual seeds that produced them.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroom_reset_interval import PairedTests
from analysis.practice_makes_perfect.tossingroomsplit_reset_policy import (
    PracticeSideReport,
    ResetPolicyReport,
    TwoProportionSensitivity,
    expected_denominators,
)

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"


def _arm(*, resets: int, finals: list[int]) -> dict:
    """One arm with `len(finals)` seeds and two checkpoints, where `finals[i]` is seed
    i's TRASH count at the last sweep. RECYCLING and EMPTY are held flat so a test that
    moves TRASH moves nothing else."""
    return {
        str(seed): {
            "resets": resets,
            "families": {
                "TRASH": [[0, 0, 14], [1500, final, 14]],
                "RECYCLING": [[0, 1, 14], [1500, 1, 14]],
                "EMPTY": [[0, 2, 2], [1500, 2, 2]],
            },
        }
        for seed, final in enumerate(finals)
    }


def _arms(*, scheduled: list[int], never: list[int]) -> dict:
    return {
        "scheduled": _arm(resets=10, finals=scheduled),
        "never": _arm(resets=0, finals=never),
    }


def test_empty_is_classified_before_recycling() -> None:
    """The EMPTY goal names both bins, so rule ORDER decides whether its 2 tasks per
    seed land in EMPTY or are silently added to RECYCLING's denominator."""
    assert ResetPolicyReport.classify(goal=_EMPTY) == "EMPTY"
    assert ResetPolicyReport.classify(goal=_TRASH) == "TRASH"
    assert ResetPolicyReport.classify(goal=_RECYCLING) == "RECYCLING"


def test_an_unrecognised_goal_raises_rather_than_bucketing_as_other() -> None:
    with pytest.raises(ValueError, match="unrecognised goal"):
        ResetPolicyReport.classify(goal="SomethingElse(a, b)")


def test_the_domain_supplies_the_designed_composition() -> None:
    assert expected_denominators() == {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}


def test_a_wrong_family_denominator_is_reported_as_a_violation() -> None:
    """A composition violation has to be *detected*, since every rate below it would be
    computed on the wrong denominator."""
    arms = _arms(scheduled=[14, 14], never=[0, 0])
    assert ResetPolicyReport.composition_violations(arms=arms) == []
    arms["never"]["0"]["families"]["EMPTY"][0][2] = 3
    assert ResetPolicyReport.composition_violations(arms=arms) == ["never seed 0 EMPTY: 3 != 2"]


def test_a_reset_count_that_did_not_move_is_reported_as_a_violation() -> None:
    """`never` taking resets anyway would make the two arms the same experiment."""
    arms = _arms(scheduled=[14, 14], never=[0, 0])
    assert ResetPolicyReport.reset_violations(arms=arms) == []
    arms["never"]["1"]["resets"] = 4
    assert ResetPolicyReport.reset_violations(arms=arms) == ["never seed 1: 4 resets != 0"]


def test_paired_differences_are_never_minus_scheduled_in_tasks() -> None:
    """The sign convention the whole report rests on: negative means removing the reset
    cost the agent tasks."""
    arms = _arms(scheduled=[14, 10, 3], never=[4, 10, 8])
    assert ResetPolicyReport.paired_differences(arms=arms, family="TRASH") == [-10.0, 0.0, 5.0]
    assert ResetPolicyReport.direction_counts(differences=[-10.0, 0.0, 5.0]) == (1, 1, 1)


def test_overall_counts_sum_the_three_families() -> None:
    arms = _arms(scheduled=[9], never=[0])
    # 9 TRASH + 1 RECYCLING + 2 EMPTY out of 14 + 14 + 2.
    assert ResetPolicyReport.overall_counts_at(arms=arms, arm="scheduled") == [(12, 30)]
    assert ResetPolicyReport.final_solved(arms=arms, arm="never") == [3]


def test_the_minimum_detectable_effect_matches_a_hand_computation() -> None:
    """2.801585 * sqrt(0.8071*0.1929/140 + 0.2786*0.7214/140) * 100 = 14.14pp, the
    TRASH comparison of the committed aggregate."""
    mde = TwoProportionSensitivity.minimum_detectable_effect(
        counts_a=(113, 140), counts_b=(39, 140)
    )
    assert mde is not None
    assert mde == pytest.approx(14.14, abs=0.01)
    assert TwoProportionSensitivity.observed_difference(
        counts_a=(113, 140), counts_b=(39, 140)
    ) == pytest.approx(-52.86, abs=0.01)


def test_a_ceiling_versus_ceiling_comparison_reports_no_sensitivity_rather_than_zero() -> None:
    """20/20 against 20/20 makes every variance term exactly zero, so the formula would
    return 0.0pp -- which reads as "this design detects arbitrarily small effects" and
    means the exact opposite. `None` is the only honest answer."""
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(20, 20), counts_b=(20, 20))
        is None
    )
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(0, 20), counts_b=(0, 20))
        is None
    )
    # One arm off the ceiling is a real comparison again, however few tasks it holds.
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(20, 20), counts_b=(15, 20))
        is not None
    )


def test_the_committed_aggregate_reproduces_the_reported_numbers() -> None:
    """The aggregate in `docs/experiment-logs/` is the record that survives -- the raw
    sweep directories live outside the repo. If it stops reproducing the experiment
    log's numbers, the log is no longer verifiable from anything in this repo."""
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "experiment-logs"
        / "2026-08-06-reset-free-practice-ab.json"
    )
    arms = ResetPolicyReport.load_arms(json_path=path)
    assert ResetPolicyReport.seeds(arms=arms) == [str(seed) for seed in range(10)]
    assert ResetPolicyReport.reset_counts(arms=arms, arm="scheduled") == [10] * 10
    assert ResetPolicyReport.reset_counts(arms=arms, arm="never") == [0] * 10
    assert ResetPolicyReport.achieved_transitions(arms=arms, arm="scheduled") == [1500] * 10
    assert ResetPolicyReport.achieved_transitions(arms=arms, arm="never") == [1500] * 10
    assert ResetPolicyReport.composition_violations(arms=arms) == []
    assert ResetPolicyReport.reset_violations(arms=arms) == []

    assert ResetPolicyReport.final_solved(arms=arms, arm="scheduled") == [
        8, 16, 18, 18, 16, 16, 18, 17, 6, 18,
    ]  # fmt: skip
    assert ResetPolicyReport.final_solved(arms=arms, arm="never") == [
        8, 10, 6, 5, 7, 11, 8, 12, 6, 12,
    ]  # fmt: skip
    for arm, expected in (
        ("scheduled", {"TRASH": (113, 140), "RECYCLING": (18, 140), "EMPTY": (20, 20)}),
        ("never", {"TRASH": (39, 140), "RECYCLING": (26, 140), "EMPTY": (20, 20)}),
    ):
        for family, counts in expected.items():
            assert ResetPolicyReport.pooled_counts(arms=arms, arm=arm, family=family) == counts
    assert ResetPolicyReport.pooled_counts(arms=arms, arm="scheduled") == (151, 300)
    assert ResetPolicyReport.pooled_counts(arms=arms, arm="never") == (85, 300)


def test_the_committed_aggregate_reproduces_the_exact_paired_p_value() -> None:
    """Imported from `tossingroom_reset_interval` rather than reimplemented, so this
    pins the wiring and the fixed sign convention, not the enumeration itself.

    8 of 10 seeds are nonzero and every one of them is negative, so the only sign
    assignments at least as extreme as the observed one are all-negative and
    all-positive: p = 2/2**8 = 0.0078125.
    """
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "experiment-logs"
        / "2026-08-06-reset-free-practice-ab.json"
    )
    arms = json.loads(path.read_text())
    differences = ResetPolicyReport.paired_differences(arms=arms)
    assert differences == [0.0, -6.0, -12.0, -13.0, -9.0, -5.0, -10.0, -5.0, 0.0, -6.0]
    assert ResetPolicyReport.direction_counts(differences=differences) == (8, 2, 0)
    assert PairedTests.sign_flip(differences=differences).p_value == pytest.approx(2 / 256)


# --------------------------------------------------------------------- practice side


def _raw_trace(*, policy: str, periods_per_seed: list[list[dict[str, list[int]]]]) -> dict:
    """A raw per-period trace file of the shape the practice sweeps emit, with the
    sampler-parameter arrays present -- those are the bulk of the 550 KB the condensed
    record has to drop, so a fixture without them could not show that it does."""
    return {
        "label": policy,
        "env": "tossingroomsplit",
        "practice_reset_policy": policy,
        "num_cycles": len(periods_per_seed[0]),
        "seeds": [
            {
                "seed": seed,
                "periods": [
                    {
                        "skills": {
                            skill: {
                                "attempts": attempts,
                                "successes": successes,
                                "greedy_forces": [0.5] * attempts,
                                "greedy_targets": [0.5] * attempts,
                            }
                            for skill, (attempts, successes) in period.items()
                        }
                    }
                    for period in periods
                ],
            }
            for seed, periods in enumerate(periods_per_seed)
        ],
    }


def _write_traces(*, tmp_path: Path, scheduled: dict, never: dict) -> dict[str, Path]:
    paths = {}
    for policy, raw in (("scheduled", scheduled), ("never", never)):
        path = tmp_path / f"traces-{policy}.json"
        path.write_text(json.dumps(raw))
        paths[policy] = path
    return paths


def _two_period_traces(*, tmp_path: Path) -> dict[str, Path]:
    """One seed per arm, two periods each. `scheduled` throws in both periods;
    `never` throws in the first and is stranded in the second, spending the same
    budget on MoveRoom instead."""
    scheduled = _raw_trace(
        policy="scheduled",
        periods_per_seed=[
            [
                {"ThrowTrash": [3, 2], "MoveRoom": [7, 7]},
                {"ThrowRecycling": [2, 1], "MoveRoom": [8, 8]},
            ]
        ],
    )
    never = _raw_trace(
        policy="never",
        periods_per_seed=[
            [{"ThrowTrash": [1, 1], "MoveRoom": [9, 9]}, {"MoveRoom": [10, 10]}],
        ],
    )
    return _write_traces(tmp_path=tmp_path, scheduled=scheduled, never=never)


def test_condensing_keeps_the_per_period_counts_and_drops_the_sampler_arrays(
    *, tmp_path: Path
) -> None:
    """The condensed record is what gets committed, so it has to carry every count the
    figures and the report are computed from -- and none of the per-attempt sampler
    arrays, which are the 550 KB that make the raw traces uncommittable."""
    practice = PracticeSideReport.condense(trace_paths=_two_period_traces(tmp_path=tmp_path))
    assert PracticeSideReport.seeds(practice=practice) == ["0"]
    assert PracticeSideReport.num_cycles(practice=practice) == 2
    assert practice["scheduled"]["0"]["periods"] == [
        {"ThrowTrash": [3, 2], "MoveRoom": [7, 7]},
        {"ThrowRecycling": [2, 1], "MoveRoom": [8, 8]},
    ]
    assert "greedy_forces" not in json.dumps(practice)


def test_condensing_rejects_a_trace_file_whose_policy_is_not_the_arm_it_was_given_as(
    *, tmp_path: Path
) -> None:
    """Pointing `--trace never=...` at the scheduled arm's file produces perfectly
    plausible numbers with the two arms swapped, so the file's own recorded
    `practice_reset_policy` is checked against the name it was passed under."""
    paths = _two_period_traces(tmp_path=tmp_path)
    with pytest.raises(ValueError, match="records practice_reset_policy"):
        PracticeSideReport.condense(
            trace_paths={"scheduled": paths["never"], "never": paths["scheduled"]}
        )


def test_pooled_attempts_sum_over_seeds_and_cycles_and_default_to_every_skill(
    *, tmp_path: Path
) -> None:
    practice = PracticeSideReport.condense(trace_paths=_two_period_traces(tmp_path=tmp_path))
    assert PracticeSideReport.pooled_attempts(practice=practice, arm="scheduled") == 20
    assert PracticeSideReport.pooled_attempts(practice=practice, arm="never") == 20
    assert (
        PracticeSideReport.pooled_attempts(practice=practice, arm="scheduled", skill="ThrowTrash")
        == 3
    )
    # A skill an arm never attempted is 0, not a KeyError: `never` here has no
    # ThrowRecycling period at all.
    assert (
        PracticeSideReport.pooled_attempts(practice=practice, arm="never", skill="ThrowRecycling")
        == 0
    )
    assert (
        PracticeSideReport.pooled_successes(practice=practice, arm="scheduled", skill="ThrowTrash")
        == 2
    )


def test_throw_attempts_pool_the_two_throw_skills_per_cycle(*, tmp_path: Path) -> None:
    practice = PracticeSideReport.condense(trace_paths=_two_period_traces(tmp_path=tmp_path))
    assert PracticeSideReport.throws_per_seed_per_cycle(practice=practice, arm="scheduled") == [
        [3, 2]
    ]
    assert PracticeSideReport.throws_per_cycle(practice=practice, arm="never") == [1, 0]


def test_a_live_throw_cycle_is_one_with_any_throw_attempt_at_all(*, tmp_path: Path) -> None:
    """The count the stranding claim rests on: a period with a single throw attempt is
    live, a period with none is not. One attempt is very little practice, which is why
    the figure shows the attempt counts beside this."""
    practice = PracticeSideReport.condense(trace_paths=_two_period_traces(tmp_path=tmp_path))
    assert PracticeSideReport.live_throw_cycles(practice=practice, arm="scheduled") == [2]
    assert PracticeSideReport.live_throw_cycles(practice=practice, arm="never") == [1]


def test_contributing_seeds_names_which_seeds_carry_a_cycle(*, tmp_path: Path) -> None:
    """A pooled per-cycle count cannot distinguish "every seed still throws a little"
    from "one seed throws and the rest are stranded", and on this arm it is the
    second."""
    practice = PracticeSideReport.condense(trace_paths=_two_period_traces(tmp_path=tmp_path))
    assert PracticeSideReport.contributing_seeds(practice=practice, arm="never", cycle=0) == ["0"]
    assert PracticeSideReport.contributing_seeds(practice=practice, arm="never", cycle=1) == []


def test_an_unequal_practice_budget_is_reported_as_a_violation(*, tmp_path: Path) -> None:
    """Composition is only a fair comparison if both arms bought the same number of
    skill executions. If they did not, an arm attempting a skill more often could just
    be an arm that acted more often."""
    practice = PracticeSideReport.condense(trace_paths=_two_period_traces(tmp_path=tmp_path))
    assert PracticeSideReport.budget_violations(practice=practice) == []
    practice["never"]["0"]["periods"][1]["MoveRoom"] = [11, 11]
    assert PracticeSideReport.budget_violations(practice=practice) == [
        "seed 0: never 21 attempts != scheduled 20"
    ]


def _committed_practice() -> dict:
    return PracticeSideReport.load_practice(
        json_path=Path(__file__).resolve().parents[3]
        / "docs"
        / "experiment-logs"
        / "2026-08-06-reset-free-practice-traces.json"
    )


def test_the_committed_practice_record_reproduces_the_reported_skill_composition() -> None:
    """Both arms spend exactly 14900 practice attempts, so the composition IS the
    comparison. The `PressTrash` down / `PressRecycling` up asymmetry is the positional
    part: the trash button sits at room 6, unreachable once the robot is past the ledge,
    and the recycling button at room 1 is where a stranded robot ends up."""
    practice = _committed_practice()
    assert PracticeSideReport.seeds(practice=practice) == [str(seed) for seed in range(10)]
    assert PracticeSideReport.num_cycles(practice=practice) == 10
    assert PracticeSideReport.budget_violations(practice=practice) == []
    assert PracticeSideReport.pooled_attempts(practice=practice, arm="scheduled") == 14900
    assert PracticeSideReport.pooled_attempts(practice=practice, arm="never") == 14900
    expected = {
        "PickupTrash": (693, 298),
        "PickupRecycling": (44, 9),
        "MoveRoom": (11356, 11693),
        "ThrowTrash": (666, 300),
        "ThrowRecycling": (44, 9),
        "PressTrash": (264, 156),
        "PressRecycling": (1833, 2435),
    }
    for skill, (scheduled, never) in expected.items():
        assert (
            PracticeSideReport.pooled_attempts(practice=practice, arm="scheduled", skill=skill)
            == scheduled
        )
        assert (
            PracticeSideReport.pooled_attempts(practice=practice, arm="never", skill=skill) == never
        )
    assert sum(scheduled for scheduled, _ in expected.values()) == 14900
    assert sum(never for _, never in expected.values()) == 14900


def test_the_committed_practice_record_reproduces_the_throw_timeline() -> None:
    """The shape the whole practice-side claim is: `never` starts ahead (78 against 59
    throw attempts in cycle 0, since it is not paying for a reset) and collapses by
    cycle 3, while `scheduled` never does."""
    practice = _committed_practice()
    assert PracticeSideReport.throws_per_cycle(practice=practice, arm="scheduled") == [
        59, 46, 75, 87, 67, 77, 83, 43, 74, 99,
    ]  # fmt: skip
    assert PracticeSideReport.throws_per_cycle(practice=practice, arm="never") == [
        78, 72, 36, 20, 17, 18, 17, 17, 17, 17,
    ]  # fmt: skip
    assert PracticeSideReport.live_throw_cycles(practice=practice, arm="scheduled") == [
        7, 9, 8, 8, 7, 7, 9, 9, 8, 7,
    ]  # fmt: skip
    assert PracticeSideReport.live_throw_cycles(practice=practice, arm="never") == [
        4, 10, 1, 1, 1, 1, 3, 3, 1, 1,
    ]  # fmt: skip


def test_the_never_arms_residual_late_throws_are_one_seed() -> None:
    """The flat ~17 per cycle from cycle 4 on is not "every seed still practising a
    little" -- it is seed 1 alone, the one seed that never strands itself (10/10 live
    cycles, and the only `never` seed that never presses the recycling button). Reading
    the pooled curve without this would badly overstate what the arm still practises."""
    practice = _committed_practice()
    for cycle in range(4, 10):
        contributing = PracticeSideReport.contributing_seeds(
            practice=practice, arm="never", cycle=cycle
        )
        assert contributing == ["1"]
    assert PracticeSideReport.contributing_seeds(practice=practice, arm="never", cycle=3) == [
        "0",
        "1",
    ]
    assert (
        PracticeSideReport.pooled_attempts(practice=practice, arm="never", skill="PressRecycling")
        == 2435
    )
