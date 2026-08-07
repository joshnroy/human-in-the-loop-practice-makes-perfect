"""Tests for the 2x2 two-way-ledge analysis: world x reset policy on Tossing Room
(split throws).

Everything pinned here is something that can be wrong while the printout still looks
entirely reasonable:

1. **Goal-family classification.** The EMPTY goal names *both* bins
   (`RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)`), so the naive
   "does it mention recycling?" test swallows it and reports 16 RECYCLING / 0 EMPTY --
   plausible numbers, not an error. The rule order is the whole defence, so the naive
   rule is asserted to be wrong here rather than merely described as wrong.
2. **The 2x2 being complete.** A missing cell turns the interaction into an
   unfalsifiable comparison of three arms, so `load_arms` raises rather than reporting
   whatever it found.
3. **The penalty and interaction arithmetic**, including the sign convention: a
   positive penalty means removing the reset COST tasks.
4. **The MDE's degenerate case.** At the ceiling in both arms the normal-approximation
   formula returns 0.0pp, which reads as infinite sensitivity and means the opposite.
5. **The stranding detector**, which is the mechanism claim. Two things it must get
   right: a period spent entirely on `MoveRoom`/`Press*` is stranded while a period
   with a single `ThrowTrash` attempt is not, and the trailing bookkeeping window that
   `Metrics.practice_outcomes_per_cycle` always ends with is not a cycle. That second
   one is the expensive mistake: a 10-cycle run records **11** windows, the eleventh is
   empty, and an empty window is stranded by definition -- so counting it turns a true
   0/100 into 10/110 for every arm, including the ones that never strand.
"""

import json
import math
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroomsplit_two_way_ledge import (
    _MDE_CONSTANT,
    TwoProportionSensitivity,
    TwoWayLedgeReport,
    arm_name,
    expected_denominators,
)
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import EvaluationBreakdown, TaskOutcome

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"

_ONE_WAY_SCHEDULED = "one-way-scheduled"
_ONE_WAY_NEVER = "one-way-never"
_TWO_WAY_SCHEDULED = "two-way-scheduled"
_TWO_WAY_NEVER = "two-way-never"

# A practice period that picked an item up and threw it: live, by any reading.
_LIVE_CYCLE = {"MoveRoom": [10, 10], "PickupTrash": [3, 3], "ThrowTrash": [3, 2]}
# A practice period that only walked and pressed a button: stranded.
_STRANDED_CYCLE = {"MoveRoom": [12, 12], "PressRecycling": [5, 0]}


def _seed_record(
    *,
    resets: int,
    solved: int,
    cycles: list[dict[str, list[int]]],
    transitions: int = 1500,
) -> dict:
    """One seed's aggregate entry: `len(cycles) + 1` evaluation sweeps, every solved
    task in TRASH so that a seed's overall count is exactly `solved`, and `len(cycles)`
    practice periods plus the trailing no-practice window every
    `Metrics.practice_outcomes_per_cycle` ends with."""
    num_sweeps = len(cycles) + 1

    def curve(*, final: int, total: int) -> list[list[int]]:
        return [[index * 100, 0, total] for index in range(num_sweeps - 1)] + [
            [transitions, final, total]
        ]

    return {
        "resets": resets,
        "transitions": transitions,
        "families": {
            "TRASH": curve(final=solved, total=14),
            "RECYCLING": curve(final=0, total=14),
            "EMPTY": curve(final=0, total=2),
        },
        "practice": [*cycles, {}],
    }


def _arm(
    *,
    policy: str,
    finals: list[int],
    cycles_per_seed: list[list[dict[str, list[int]]]] | None = None,
) -> dict:
    per_seed = cycles_per_seed or [[dict(_LIVE_CYCLE)] for _ in finals]
    return {
        str(seed): _seed_record(
            resets=10 if policy == "scheduled" else 0, solved=solved, cycles=cycles
        )
        for seed, (solved, cycles) in enumerate(zip(finals, per_seed, strict=True))
    }


def _arms(
    *,
    finals: dict[str, list[int]],
    cycles: dict[str, list[list[dict[str, list[int]]]]] | None = None,
) -> dict:
    """A full 2x2 from each arm's per-seed final TRASH count."""
    per_arm_cycles = cycles or {}
    return {
        arm: _arm(
            policy=arm.rsplit("-", maxsplit=1)[1],
            finals=values,
            cycles_per_seed=per_arm_cycles.get(arm),
        )
        for arm, values in finals.items()
    }


def _square(*, one_way: tuple[list[int], list[int]], two_way: tuple[list[int], list[int]]) -> dict:
    return _arms(
        finals={
            _ONE_WAY_SCHEDULED: one_way[0],
            _ONE_WAY_NEVER: one_way[1],
            _TWO_WAY_SCHEDULED: two_way[0],
            _TWO_WAY_NEVER: two_way[1],
        }
    )


# ------------------------------------------------------------------- the 2x2 itself


def test_arm_names_are_derived_from_the_world_and_policy_pair() -> None:
    """Four unrelated strings would let a typo create a fifth arm and leave a cell of
    the square silently empty; deriving the name makes the square the thing that
    exists."""
    assert arm_name(world="one-way", policy="scheduled") == _ONE_WAY_SCHEDULED
    assert arm_name(world="two-way", policy="never") == _TWO_WAY_NEVER
    assert TwoWayLedgeReport.arms() == (
        _ONE_WAY_SCHEDULED,
        _ONE_WAY_NEVER,
        _TWO_WAY_SCHEDULED,
        _TWO_WAY_NEVER,
    )
    with pytest.raises(ValueError, match="not a world"):
        arm_name(world="three-way", policy="never")
    with pytest.raises(ValueError, match="not a reset policy"):
        arm_name(world="one-way", policy="sometimes")


def test_load_arms_raises_when_a_cell_of_the_square_is_missing(*, tmp_path: Path) -> None:
    """Three arms cannot answer the interaction question at all, so a missing cell is an
    error rather than a report with one comparison quietly dropped."""
    arms = _square(one_way=([18], [8]), two_way=([18], [17]))
    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps(arms))
    assert sorted(TwoWayLedgeReport.load_arms(json_path=complete)) == sorted(
        TwoWayLedgeReport.arms()
    )

    del arms[_TWO_WAY_NEVER]
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(arms))
    with pytest.raises(ValueError, match="missing arms.*two-way-never"):
        TwoWayLedgeReport.load_arms(json_path=incomplete)


# ------------------------------------------------------------------- classification


def test_empty_is_classified_before_recycling_and_the_naive_rule_would_be_wrong() -> None:
    """The EMPTY goal string contains both "Recycling" and "Trash", so a rule list that
    tested either throw family first would bucket EMPTY's 2 tasks per seed into a throw
    family's denominator and report 16 RECYCLING / 0 EMPTY."""
    assert "Recycling" in _EMPTY
    assert "Trash" in _EMPTY
    assert TwoWayLedgeReport.classify(goal=_EMPTY) == "EMPTY"
    assert TwoWayLedgeReport.classify(goal=_TRASH) == "TRASH"
    assert TwoWayLedgeReport.classify(goal=_RECYCLING) == "RECYCLING"


def test_an_unrecognised_goal_raises_rather_than_bucketing_as_other() -> None:
    with pytest.raises(ValueError, match="unrecognised goal"):
        TwoWayLedgeReport.classify(goal="SomethingElse(a, b)")


def test_the_domain_supplies_the_designed_composition() -> None:
    assert expected_denominators() == {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}


# -------------------------------------------------------------------- the hard checks


def test_a_reset_count_that_did_not_move_is_reported_as_a_violation() -> None:
    arms = _square(one_way=([18, 16], [8, 10]), two_way=([18, 16], [17, 16]))
    assert TwoWayLedgeReport.reset_violations(arms=arms) == []
    arms[_TWO_WAY_NEVER]["1"]["resets"] = 4
    assert TwoWayLedgeReport.reset_violations(arms=arms) == ["two-way-never seed 1: 4 resets != 0"]


def test_a_run_that_did_not_reach_the_full_budget_is_reported_as_a_violation() -> None:
    """Equal experience is measured, not assumed: a period ending early on
    `InteractionComplete` is not charged the steps it did not take."""
    arms = _square(one_way=([18], [8]), two_way=([18], [17]))
    assert TwoWayLedgeReport.transition_violations(arms=arms) == []
    arms[_ONE_WAY_NEVER]["0"]["transitions"] = 1350
    assert TwoWayLedgeReport.transition_violations(arms=arms) == [
        "one-way-never seed 0: 1350 transitions != 1500"
    ]


def test_a_wrong_family_denominator_is_reported_as_a_violation() -> None:
    arms = _square(one_way=([18], [8]), two_way=([18], [17]))
    assert TwoWayLedgeReport.composition_violations(arms=arms) == []
    arms[_ONE_WAY_NEVER]["0"]["families"]["EMPTY"][0][2] = 3
    assert TwoWayLedgeReport.composition_violations(arms=arms) == [
        "one-way-never seed 0 EMPTY: 3 != 2"
    ]


def test_a_practice_record_whose_trailing_window_practised_is_a_violation() -> None:
    """The last window of `practice_outcomes_per_cycle` covers the final evaluation
    sweep alone and must be empty. If it is not, every cycle index below is off by one
    and the stranding timeline is shifted."""
    arms = _square(one_way=([18], [8]), two_way=([18], [17]))
    assert TwoWayLedgeReport.practice_window_violations(arms=arms) == []
    arms[_ONE_WAY_NEVER]["0"]["practice"][-1] = {"MoveRoom": [3, 3]}
    assert TwoWayLedgeReport.practice_window_violations(arms=arms) == [
        "one-way-never seed 0: trailing practice window has 3 attempts, expected 0"
    ]


def test_a_practice_record_with_a_second_empty_window_is_a_violation() -> None:
    """More than one empty trailing window would make `num_cycles` count fewer periods
    than the run ran, silently dropping a real period from every count below."""
    arms = _arms(
        finals={
            _ONE_WAY_SCHEDULED: [18],
            _ONE_WAY_NEVER: [8],
            _TWO_WAY_SCHEDULED: [18],
            _TWO_WAY_NEVER: [17],
        },
        cycles={arm: [[dict(_LIVE_CYCLE), dict(_LIVE_CYCLE)]] for arm in TwoWayLedgeReport.arms()},
    )
    assert TwoWayLedgeReport.practice_window_violations(arms=arms) == []
    arms[_ONE_WAY_NEVER]["0"]["practice"][1] = {}
    assert TwoWayLedgeReport.practice_window_violations(arms=arms) == [
        "one-way-never seed 0: 1 practice periods with any recorded attempt != 2 expected"
    ]


# ------------------------------------------------------------------------ the penalty


def test_the_penalty_is_scheduled_minus_never_per_seed() -> None:
    """The sign convention the whole report rests on: a POSITIVE penalty means removing
    the reset cost the agent tasks."""
    arms = _square(one_way=([18, 16, 6], [8, 10, 6]), two_way=([18, 16, 6], [17, 16, 4]))
    assert TwoWayLedgeReport.penalties(arms=arms, world="one-way") == [10.0, 6.0, 0.0]
    assert TwoWayLedgeReport.penalties(arms=arms, world="two-way") == [1.0, 0.0, 2.0]
    assert TwoWayLedgeReport.direction_counts(penalties=[10.0, 6.0, 0.0]) == (2, 1, 0)
    assert TwoWayLedgeReport.direction_counts(penalties=[1.0, 0.0, -2.0]) == (1, 1, 1)


def test_the_interaction_is_the_one_way_penalty_minus_the_two_way_one() -> None:
    """The headline quantity: how much MORE the reset-free arm loses when the world has
    an irreversible action in it."""
    arms = _square(one_way=([18, 16, 6], [8, 10, 6]), two_way=([18, 16, 6], [17, 16, 4]))
    assert TwoWayLedgeReport.interaction_differences(arms=arms) == [9.0, 6.0, -2.0]


def test_final_solved_and_pooled_counts_sum_the_three_families() -> None:
    arms = _square(one_way=([18, 16], [8, 10]), two_way=([18, 16], [17, 16]))
    assert TwoWayLedgeReport.final_solved(arms=arms, arm=_ONE_WAY_SCHEDULED) == [18, 16]
    assert TwoWayLedgeReport.pooled_counts(arms=arms, arm=_ONE_WAY_SCHEDULED) == (34, 60)
    assert TwoWayLedgeReport.pooled_counts(arms=arms, arm=_ONE_WAY_SCHEDULED, family="TRASH") == (
        34,
        28,
    )
    assert TwoWayLedgeReport.pooled_counts(arms=arms, arm=_ONE_WAY_NEVER, family="EMPTY") == (0, 4)


def test_the_pooled_penalty_is_the_difference_of_two_pooled_rates() -> None:
    arms = _square(one_way=([15, 15], [5, 5]), two_way=([15, 15], [15, 15]))
    # 30/60 against 10/60 is 50.0% - 16.7% = +33.3pp.
    assert TwoWayLedgeReport.pooled_penalty(arms=arms, world="one-way") == pytest.approx(
        33.33, abs=0.01
    )
    assert TwoWayLedgeReport.pooled_penalty(arms=arms, world="two-way") == pytest.approx(0.0)


# --------------------------------------------------------------------------- sensitivity


def test_the_minimum_detectable_effect_matches_a_hand_computation() -> None:
    """2.801585 * sqrt(0.8071*0.1929/140 + 0.2786*0.7214/140) * 100 = 14.14pp."""
    mde = TwoProportionSensitivity.minimum_detectable_effect(
        counts_a=(113, 140), counts_b=(39, 140)
    )
    assert mde is not None
    assert mde == pytest.approx(14.14, abs=0.01)


def test_the_printed_constant_is_the_one_the_formula_actually_uses() -> None:
    """`_MDE_CONSTANT` is printed in the sensitivity header, so it has to be the number
    the formula evaluates with -- a header that drifts from the computation is worse
    than no header at all."""
    first, second = 113 / 140, 39 / 140
    variance = first * (1 - first) / 140 + second * (1 - second) / 140
    assert TwoProportionSensitivity.minimum_detectable_effect(
        counts_a=(113, 140), counts_b=(39, 140)
    ) == pytest.approx(100.0 * _MDE_CONSTANT * math.sqrt(variance))


def test_a_ceiling_versus_ceiling_comparison_reports_no_sensitivity_rather_than_zero() -> None:
    """Every variance term is exactly zero there, so the formula returns 0.0pp -- which
    reads as "this design detects arbitrarily small effects" and means the opposite."""
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(20, 20), counts_b=(20, 20))
        is None
    )
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(0, 20), counts_b=(0, 20))
        is None
    )
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(20, 20), counts_b=(15, 20))
        is not None
    )
    assert (
        TwoProportionSensitivity.minimum_detectable_effect(counts_a=(0, 0), counts_b=(1, 2)) is None
    )


# ------------------------------------------------------------------------- stranding


def test_a_cycle_with_no_pickup_or_throw_attempt_is_stranded() -> None:
    """The measured definition of the mechanism: the robot is past the ledge, so the
    only skills left to it are walking and pressing the button on its side."""
    assert TwoWayLedgeReport.is_stranded(window=_STRANDED_CYCLE) is True
    assert TwoWayLedgeReport.is_stranded(window={}) is True
    assert TwoWayLedgeReport.is_stranded(window={"MoveRoom": [12, 12]}) is True
    # A skill recorded with zero attempts is the same as one absent altogether.
    assert TwoWayLedgeReport.is_stranded(window={"ThrowTrash": [0, 0], "MoveRoom": [9, 9]}) is True


def test_a_single_throw_or_pickup_attempt_makes_a_cycle_live() -> None:
    """Deliberately generous: one attempt in a whole period counts. It is the PRESENCE
    of item experience the stranding claim is about, and any threshold above 1 would be
    a free parameter chosen after seeing the data."""
    assert (
        TwoWayLedgeReport.is_stranded(window={"MoveRoom": [12, 12], "ThrowTrash": [1, 0]}) is False
    )
    assert (
        TwoWayLedgeReport.is_stranded(window={"MoveRoom": [12, 12], "PickupRecycling": [1, 1]})
        is False
    )


def _stranding_arms() -> dict:
    """Three seeds, three practice periods. Seed 0 strands from cycle 1 on, seed 1 never
    strands, seed 2 strands only in the last cycle."""
    live, dead = dict(_LIVE_CYCLE), dict(_STRANDED_CYCLE)
    return _arms(
        finals={
            _ONE_WAY_SCHEDULED: [18, 16, 6],
            _ONE_WAY_NEVER: [8, 10, 6],
            _TWO_WAY_SCHEDULED: [18, 16, 6],
            _TWO_WAY_NEVER: [17, 16, 4],
        },
        cycles={
            _ONE_WAY_NEVER: [[live, dead, dead], [live, live, live], [live, live, dead]],
            _ONE_WAY_SCHEDULED: [[live] * 3] * 3,
            _TWO_WAY_SCHEDULED: [[live] * 3] * 3,
            _TWO_WAY_NEVER: [[live] * 3] * 3,
        },
    )


def test_stranded_cycles_are_counted_per_seed_and_the_first_one_is_named() -> None:
    arms = _stranding_arms()
    assert TwoWayLedgeReport.num_cycles(arms=arms) == 3
    assert TwoWayLedgeReport.stranded_cycles(arms=arms, arm=_ONE_WAY_NEVER) == [
        [False, True, True],
        [False, False, False],
        [False, False, True],
    ]
    assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=_ONE_WAY_NEVER) == 3
    assert TwoWayLedgeReport.seeds_that_strand(arms=arms, arm=_ONE_WAY_NEVER) == ["0", "2"]
    assert TwoWayLedgeReport.first_stranded_cycle(arms=arms, arm=_ONE_WAY_NEVER) == [1, None, 2]
    assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=_ONE_WAY_SCHEDULED) == 0
    assert TwoWayLedgeReport.seeds_that_strand(arms=arms, arm=_TWO_WAY_NEVER) == []


def test_the_trailing_no_practice_window_is_not_counted_as_a_stranded_cycle() -> None:
    """Every run's practice record ends with an empty window covering the final
    evaluation sweep. Counting it would add one stranded cycle to every seed of every
    arm, and would make even the scheduled arms look as if they strand."""
    arms = _stranding_arms()
    assert len(arms[_ONE_WAY_SCHEDULED]["0"]["practice"]) == 4
    assert TwoWayLedgeReport.num_cycles(arms=arms) == 3
    assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=_ONE_WAY_SCHEDULED) == 0


def test_a_ten_cycle_run_records_eleven_windows_and_still_reports_ten_cycles() -> None:
    """The shape of the real runs, at their real size: 11 windows per seed, the eleventh
    empty. The denominator has to be 10 seeds x 10 cycles = 100, and an arm whose every
    period picked something up has to report 0/100 stranded -- not 10/110."""
    seeds, cycles = 10, 10
    arms = _arms(
        finals={arm: [20] * seeds for arm in TwoWayLedgeReport.arms()},
        cycles={
            arm: [[dict(_LIVE_CYCLE) for _ in range(cycles)] for _ in range(seeds)]
            for arm in TwoWayLedgeReport.arms()
        },
    )
    assert len(arms[_TWO_WAY_NEVER]["0"]["practice"]) == 11
    assert TwoWayLedgeReport.num_cycles(arms=arms) == 10
    assert TwoWayLedgeReport.practice_window_violations(arms=arms) == []
    for arm in TwoWayLedgeReport.arms():
        assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=arm) == 0
        assert TwoWayLedgeReport.seeds_that_strand(arms=arms, arm=arm) == []


def test_item_attempts_are_reported_per_seed_per_cycle_for_the_heatmap() -> None:
    arms = _stranding_arms()
    # 3 PickupTrash + 3 ThrowTrash in a live period; 0 in a stranded one.
    assert TwoWayLedgeReport.item_attempts(arms=arms, arm=_ONE_WAY_NEVER) == [
        [6, 0, 0],
        [6, 6, 6],
        [6, 6, 0],
    ]


# ------------------------------------------------------------------------ aggregation


def _stats_json(*, solved_trash: int, resets: int) -> str:
    """One run's `stats.json`, with the designed 14/14/2 composition and two sweeps."""
    goals = [_TRASH] * 14 + [_RECYCLING] * 14 + [_EMPTY] * 2

    def sweep(*, transitions: int, solved: int) -> EvaluationBreakdown:
        return EvaluationBreakdown(
            num_online_transitions=transitions,
            outcomes=tuple(
                TaskOutcome(task_index=index, goal=goal, solved=goal == _TRASH and index < solved)
                for index, goal in enumerate(goals)
            ),
        )

    breakdowns = [sweep(transitions=0, solved=0), sweep(transitions=1500, solved=solved_trash)]
    return Metrics(
        evaluations=[
            (breakdown.num_online_transitions, breakdown.num_solved(), len(breakdown.outcomes))
            for breakdown in breakdowns
        ],
        breakdowns=breakdowns,
        num_practice_resets=resets,
        practice_outcomes_per_cycle=[
            {
                "MoveRoom": SkillPracticeTally(num_attempts=10, num_successes=10),
                "ThrowTrash": SkillPracticeTally(num_attempts=3, num_successes=2),
            },
            {},
        ],
    ).model_dump_json()


def test_aggregating_reads_stats_json_back_through_metrics(*, tmp_path: Path) -> None:
    """The committed aggregate is what survives -- raw sweep directories live outside
    the repo -- so it has to carry every quantity the report and the figures need."""
    roots = {}
    for arm, solved, resets in (
        (_ONE_WAY_SCHEDULED, 12, 10),
        (_ONE_WAY_NEVER, 4, 0),
        (_TWO_WAY_SCHEDULED, 13, 10),
        (_TWO_WAY_NEVER, 12, 0),
    ):
        run = tmp_path / arm / "ees" / "0"
        run.mkdir(parents=True)
        (run / "stats.json").write_text(_stats_json(solved_trash=solved, resets=resets))
        roots[arm] = tmp_path / arm

    aggregate = TwoWayLedgeReport.aggregate(arm_dirs=roots)
    assert sorted(aggregate) == sorted(TwoWayLedgeReport.arms())
    entry = aggregate[_ONE_WAY_SCHEDULED]["0"]
    assert entry["resets"] == 10
    assert entry["transitions"] == 1500
    assert entry["families"]["TRASH"] == [[0, 0, 14], [1500, 12, 14]]
    assert entry["families"]["EMPTY"] == [[0, 0, 2], [1500, 0, 2]]
    assert entry["practice"] == [{"MoveRoom": [10, 10], "ThrowTrash": [3, 2]}, {}]

    assert TwoWayLedgeReport.composition_violations(arms=aggregate) == []
    assert TwoWayLedgeReport.reset_violations(arms=aggregate) == []
    assert TwoWayLedgeReport.transition_violations(arms=aggregate) == []
    assert TwoWayLedgeReport.practice_window_violations(arms=aggregate) == []
    assert TwoWayLedgeReport.penalties(arms=aggregate, world="one-way") == [8.0]
    assert TwoWayLedgeReport.penalties(arms=aggregate, world="two-way") == [1.0]
    assert TwoWayLedgeReport.interaction_differences(arms=aggregate) == [7.0]


def test_aggregating_rejects_an_arm_name_that_is_not_a_cell_of_the_square(
    *, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="not a cell of the 2x2"):
        TwoWayLedgeReport.aggregate(arm_dirs={"twoway-never": tmp_path})


def test_aggregating_rejects_an_incomplete_square(*, tmp_path: Path) -> None:
    run = tmp_path / "ees" / "0"
    run.mkdir(parents=True)
    (run / "stats.json").write_text(_stats_json(solved_trash=12, resets=10))
    with pytest.raises(ValueError, match="missing arms"):
        TwoWayLedgeReport.aggregate(arm_dirs={_ONE_WAY_SCHEDULED: tmp_path})


class TestTheCommittedAggregate:
    """The experiment log's headline numbers, asserted against the **committed**
    aggregate rather than a scratchpad path.

    The raw sweep directories are committed too, but the aggregate is what every figure
    and every printed number is regenerated from, so it is the artifact worth pinning. A
    refactor that silently changes how a family is bucketed, how a cycle is counted or
    how the penalty is signed would otherwise keep passing the synthetic tests above
    while quietly rewriting the published result.

    These are `x/y` counts, not percentages, for the same reason the log is: EMPTY's
    20/20 and TRASH's 140 sit on very different denominators."""

    AGGREGATE = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "experiment-logs"
        / "2026-08-06-reset-free-two-way-ledge.json"
    )

    @staticmethod
    def _arms() -> dict:
        return TwoWayLedgeReport.load_arms(json_path=TestTheCommittedAggregate.AGGREGATE)

    def test_the_committed_aggregate_holds_the_full_square_of_ten_seeds(self) -> None:
        arms = TestTheCommittedAggregate._arms()
        assert sorted(arms) == sorted(TwoWayLedgeReport.arms())
        assert len(TwoWayLedgeReport.seeds(arms=arms)) == 10
        assert TwoWayLedgeReport.num_cycles(arms=arms) == 10

    def test_every_manipulation_check_passes(self) -> None:
        """Asserted, not merely printed: a run whose reset count or budget drifted would
        invalidate the comparison before any outcome was worth reading."""
        arms = TestTheCommittedAggregate._arms()
        assert TwoWayLedgeReport.reset_violations(arms=arms) == []
        assert TwoWayLedgeReport.transition_violations(arms=arms) == []
        assert TwoWayLedgeReport.composition_violations(arms=arms) == []
        assert TwoWayLedgeReport.practice_window_violations(arms=arms) == []

    def test_the_one_way_arms_reproduce_the_published_reset_free_ab(self) -> None:
        """PR #115's committed per-seed final counts, reproduced exactly by fresh runs at
        this branch. This is what makes `--two-way-ledge` provably inert when off."""
        arms = TestTheCommittedAggregate._arms()
        assert TwoWayLedgeReport.final_solved(arms=arms, arm="one-way-scheduled") == [
            8,
            16,
            18,
            18,
            16,
            16,
            18,
            17,
            6,
            18,
        ]
        assert TwoWayLedgeReport.final_solved(arms=arms, arm="one-way-never") == [
            8,
            10,
            6,
            5,
            7,
            11,
            8,
            12,
            6,
            12,
        ]

    def test_the_four_pooled_final_counts(self) -> None:
        arms = TestTheCommittedAggregate._arms()
        assert TwoWayLedgeReport.pooled_counts(arms=arms, arm="one-way-scheduled") == (151, 300)
        assert TwoWayLedgeReport.pooled_counts(arms=arms, arm="one-way-never") == (85, 300)
        assert TwoWayLedgeReport.pooled_counts(arms=arms, arm="two-way-scheduled") == (276, 300)
        assert TwoWayLedgeReport.pooled_counts(arms=arms, arm="two-way-never") == (144, 300)

    def test_the_per_family_counts_the_log_reports(self) -> None:
        arms = TestTheCommittedAggregate._arms()
        expected = {
            "one-way-scheduled": {"TRASH": (113, 140), "RECYCLING": (18, 140), "EMPTY": (20, 20)},
            "one-way-never": {"TRASH": (39, 140), "RECYCLING": (26, 140), "EMPTY": (20, 20)},
            "two-way-scheduled": {"TRASH": (134, 140), "RECYCLING": (122, 140), "EMPTY": (20, 20)},
            "two-way-never": {"TRASH": (51, 140), "RECYCLING": (73, 140), "EMPTY": (20, 20)},
        }
        for arm, families in expected.items():
            for family, counts in families.items():
                assert (
                    TwoWayLedgeReport.pooled_counts(arms=arms, arm=arm, family=family) == counts
                ), f"{arm} {family}"

    def test_the_mechanism_was_removed_completely(self) -> None:
        """The claim the whole experiment turns on, and the reason it is measured rather
        than assumed. The one-way `never` row independently reproduces the published
        follow-up's 74 post-onset periods and its "seed 1 alone never strands"."""
        arms = TestTheCommittedAggregate._arms()
        assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm="one-way-never") == 74
        assert len(TwoWayLedgeReport.seeds_that_strand(arms=arms, arm="one-way-never")) == 9
        for arm in ("two-way-scheduled", "two-way-never"):
            assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=arm) == 0, arm
            assert TwoWayLedgeReport.seeds_that_strand(arms=arms, arm=arm) == [], arm

    def test_the_scheduled_reset_bounds_stranding_rather_than_preventing_it(self) -> None:
        """21/100, not 0/100, in 10/10 seeds. A fact about what the control arm actually
        is, and easy to assume away."""
        arms = TestTheCommittedAggregate._arms()
        assert TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm="one-way-scheduled") == 21
        assert len(TwoWayLedgeReport.seeds_that_strand(arms=arms, arm="one-way-scheduled")) == 10

    def test_the_penalty_grew_rather_than_shrank(self) -> None:
        arms = TestTheCommittedAggregate._arms()
        assert TwoWayLedgeReport.penalties(arms=arms, world="one-way") == [
            0,
            6,
            12,
            13,
            9,
            5,
            10,
            5,
            0,
            6,
        ]
        assert TwoWayLedgeReport.penalties(arms=arms, world="two-way") == [
            13,
            9,
            13,
            17,
            16,
            8,
            14,
            15,
            11,
            16,
        ]
        assert TwoWayLedgeReport.pooled_penalty(arms=arms, world="one-way") == pytest.approx(22.0)
        assert TwoWayLedgeReport.pooled_penalty(arms=arms, world="two-way") == pytest.approx(44.0)

    def test_the_interaction_is_negative_in_every_seed(self) -> None:
        """The positive control's actual claim, and the direction it failed in. 10/10
        seeds, so the exact two-sided sign-flip p is its floor of 2/1024."""
        arms = TestTheCommittedAggregate._arms()
        interaction = TwoWayLedgeReport.interaction_differences(arms=arms)
        assert interaction == [-13, -3, -1, -4, -7, -3, -4, -10, -11, -10]
        assert all(d < 0 for d in interaction)

    def test_seed_one_is_the_seed_that_gained_nothing(self) -> None:
        """The consistency check that makes the result land: #115's seed 1 is the only
        reset-free seed that never stranded, and it is the only one of the ten that did
        not improve when stranding was removed."""
        arms = TestTheCommittedAggregate._arms()
        one_way = TwoWayLedgeReport.final_solved(arms=arms, arm="one-way-never")
        two_way = TwoWayLedgeReport.final_solved(arms=arms, arm="two-way-never")
        gains = [b - a for a, b in zip(one_way, two_way, strict=True)]
        assert gains[1] == 0
        assert [i for i, g in enumerate(gains) if g == 0] == [1]
        assert sum(1 for g in gains if g > 0) == 9

    def test_the_empty_family_supports_no_inference_rather_than_being_a_null(self) -> None:
        arms = TestTheCommittedAggregate._arms()
        for world in ("one-way", "two-way"):
            scheduled, never = TwoWayLedgeReport.world_arms(world=world)
            assert (
                TwoProportionSensitivity.minimum_detectable_effect(
                    counts_a=TwoWayLedgeReport.pooled_counts(
                        arms=arms, arm=scheduled, family="EMPTY"
                    ),
                    counts_b=TwoWayLedgeReport.pooled_counts(arms=arms, arm=never, family="EMPTY"),
                )
                is None
            ), world
