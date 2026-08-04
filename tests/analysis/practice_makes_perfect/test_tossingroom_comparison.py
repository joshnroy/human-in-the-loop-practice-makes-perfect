"""The first tests under `tests/analysis/`. They exist because
`TossingRoomComparison.wilcoxon_signed_rank` is a hand-rolled significance test, and a
p-value is the one kind of output whose wrongness is invisible on inspection -- a sign
error or a botched tie rule produces a plausible number, not a crash. This project has
already published a wrong p-value once (an unpaired test on a paired design), so the
replacement gets pinned against values derivable by hand rather than trusted.

Expected values below are computed by enumeration on paper, not by running the code and
recording what it said.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroom_comparison import TossingRoomComparison


def _write_run(*, root: Path, seed: int, num_total: int) -> None:
    directory = root / "ees" / str(seed)
    directory.mkdir(parents=True)
    (directory / "stats.json").write_text(
        json.dumps({"evaluations": [[0, 1, num_total], [100, 2, num_total]], "task_name": "t"})
    )


def test_wilcoxon_reaches_its_floor_when_every_pair_moves_the_same_way() -> None:
    """Ten pairs all favouring one arm is the most extreme outcome available, so the
    two-sided p is exactly 2 / 2**10 -- one enumeration in each tail. This is also the
    power statement for the grid: n = 10 cannot produce a p below 0.002 however large
    the effect, so a null result at this n is 'underpowered', not 'no difference'."""
    result = TossingRoomComparison.wilcoxon_signed_rank(first=list(range(1, 11)), second=[0.0] * 10)
    assert result["n"] == 10
    assert result["statistic"] == 55.0  # ranks 1..10, all positive
    assert result["p"] == 2 / 2**10


def test_wilcoxon_matches_a_hand_enumerated_five_pair_case() -> None:
    """Differences (-1, 2, 3, -4, 5) give |d| ranks 1..5 and W+ = 2 + 3 + 5 = 10 against
    a null mean of 7.5. Of the 32 sign assignments, 10 give a sum <= 5 and 10 give >= 10,
    so the exact two-sided p is 20/32."""
    result = TossingRoomComparison.wilcoxon_signed_rank(
        first=[-1.0, 2.0, 3.0, -4.0, 5.0], second=[0.0] * 5
    )
    assert result["statistic"] == 10.0
    assert result["p"] == 20 / 32


def test_wilcoxon_averages_ranks_across_tied_magnitudes() -> None:
    """Differences (1, -1, 2, 2): the two magnitude-1 pairs share rank 1.5 and the two
    magnitude-2 pairs share 3.5, so W+ = 1.5 + 3.5 + 3.5 = 8.5. Without the tie
    correction the statistic would land on an integer, which is the visible symptom."""
    result = TossingRoomComparison.wilcoxon_signed_rank(
        first=[1.0, -1.0, 2.0, 2.0], second=[0.0] * 4
    )
    assert result["statistic"] == 8.5


def test_wilcoxon_drops_tied_pairs_and_reports_the_reduced_n() -> None:
    """Saturating arms tie often, and a test silently run on three pairs must not read
    as one run on ten -- so the effective n comes back out, and with it the p floor of
    2 / 2**3 = 0.25 that three pairs impose."""
    result = TossingRoomComparison.wilcoxon_signed_rank(
        first=[1.0, 1.0, 2.0, 3.0, 4.0], second=[1.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert result["n"] == 3
    assert result["p"] == 2 / 2**3


def test_wilcoxon_on_two_identical_arms_is_not_significant() -> None:
    """Every pair ties, so there is nothing to test and p must be 1.0 -- not a
    division-by-zero, and not a spuriously small number from an empty rank sum."""
    result = TossingRoomComparison.wilcoxon_signed_rank(first=[1.0, 2.0], second=[1.0, 2.0])
    assert result["n"] == 0
    assert result["p"] == 1.0


def test_transitions_to_reach_finds_the_first_crossing_not_the_last() -> None:
    """A saturating curve reaches the threshold once and stays; the interesting number
    is when it got there, so a later re-crossing must not overwrite it."""
    curve = {0: 20.0, 150: 60.0, 300: 100.0, 450: 90.0, 600: 100.0}
    assert TossingRoomComparison.transitions_to_reach(curve=curve, threshold=100.0) == 300


def test_transitions_to_reach_is_none_when_a_seed_never_gets_there() -> None:
    """Reported as None rather than as the final transition count: a seed that never
    reached the threshold has no speed, and imputing the budget would flatter it."""
    curve = {0: 20.0, 150: 60.0, 300: 90.0}
    assert TossingRoomComparison.transitions_to_reach(curve=curve, threshold=100.0) is None


def test_realised_test_composition_is_14_14_2_on_every_seed() -> None:
    """The denominator every Tossing Room percentage is measured against. It is fixed at
    30 test tasks, and fixed *identically on every seed* -- which is the point of the
    fixed-composition change, and the thing that silently varied (16/10/4 at seed 0,
    11/12/7 at seed 1) before it. Pinned here rather than trusted, because composition
    drift would move every reported number without breaking anything."""
    composition = TossingRoomComparison.realised_test_composition(
        num_test_tasks=30, seeds=[str(seed) for seed in range(10)]
    )
    assert composition == {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}


def test_realised_test_composition_needs_num_test_tasks_to_match_the_run() -> None:
    """At 10 test tasks the composition is 4/4/2, not a scaled 14/14/2 -- so a caller
    that passes the wrong `num_test_tasks` gets a wrong denominator rather than an
    error. This is pinned because two `scripts/` probes did exactly that, drawing 30
    tasks from a `TossingRoomTasks` built for 10 and silently realising 12/12/6."""
    ten = TossingRoomComparison.realised_test_composition(num_test_tasks=10, seeds=["0"])
    assert ten == {"TRASH": 4, "RECYCLING": 4, "EMPTY": 2}
    assert sum(ten.values()) == 10


def test_composition_check_rejects_runs_scored_on_a_different_test_set(*, tmp_path: Path) -> None:
    """The composition is replicated from TossingRoomTasks, not read out of the runs, so
    on its own it would happily describe a test set the arms were never scored on. Every
    evaluation records its own num_total, so the two are cross-checked -- this is the
    guard against explaining a page of percentages with the wrong denominator."""
    root = tmp_path / "arm"
    _write_run(root=root, seed=0, num_total=10)
    with pytest.raises(ValueError, match="does not match the runs being read"):
        TossingRoomComparison.print_test_composition(
            num_test_tasks=30, seeds=["0"], arms=[("arm", root)]
        )


def test_composition_check_passes_when_the_runs_agree(*, tmp_path: Path) -> None:
    """The complement, so the guard above cannot degenerate into always raising."""
    root = tmp_path / "arm"
    _write_run(root=root, seed=0, num_total=30)
    TossingRoomComparison.print_test_composition(
        num_test_tasks=30, seeds=["0"], arms=[("arm", root)]
    )


# --- the counts behind every published success rate ---------------------------------
#
# The log reports rates as `x/y`, and neither number is recovered by multiplying a
# percentage by a seed count: `Metrics.record_evaluation` writes num_solved/num_total as
# the primary record, so both integers were on disk from the start. The release arms'
# triples are committed as `2026-08-04-tossingroom-arms.json` so that stays checkable
# once the results directory is gone; these tests check it.
#
# Expected percentages are quoted from the log, not recomputed from the JSON -- a test
# that derives its own expectation would agree with any data it was handed.

ARMS_JSON = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "experiment-logs"
    / "2026-08-04-tossingroom-arms.json"
)

PUBLISHED_SORTED_FINALS = {
    "ees-1000": [23.3, 53.3, 53.3, 53.3, 100, 100, 100, 100, 100, 100],
    "ees-10000": [83.3, 93.3, 96.7, 100, 100, 100, 100, 100, 100, 100],
    "ees-100000": [56.7, 90.0, 90.0, 96.7, 96.7, 100, 100, 100, 100, 100],
    "random-skills": [0, 0, 0, 0, 0, 3.3, 3.3, 3.3, 3.3, 3.3],
}
PUBLISHED_POOLED_FINALS = {
    "ees-1000": (235, 300),
    "ees-10000": (292, 300),
    "ees-100000": (279, 300),
    "random-skills": (5, 300),
}


@pytest.mark.parametrize("arm", sorted(PUBLISHED_SORTED_FINALS))
def test_committed_counts_reproduce_the_published_per_seed_finals(*, arm: str) -> None:
    """Dividing the recorded numerator by the recorded denominator must give back the
    percentage the log published, to the decimal it was rounded to. A mismatch would
    mean a published rate was wrong -- a finding well beyond the formatting."""
    arms = json.loads(ARMS_JSON.read_text())
    percentages = sorted(100.0 * c[-1][1] / c[-1][2] for c in arms[arm].values())
    assert percentages == pytest.approx(PUBLISHED_SORTED_FINALS[arm], abs=0.05)


@pytest.mark.parametrize("arm", sorted(PUBLISHED_POOLED_FINALS))
def test_committed_counts_pool_to_the_published_arm_totals(*, arm: str) -> None:
    """78.3% is 235 of 300 evaluation episodes, summed rather than averaged. Every seed
    runs the same 30 tasks, so pooling and averaging agree -- pinned here because the
    arm table would otherwise be quoting one while claiming the other."""
    arms = json.loads(ARMS_JSON.read_text())
    curves = list(arms[arm].values())
    assert {total for c in curves for _, _, total in c} == {30}
    pooled = (sum(c[-1][1] for c in curves), sum(c[-1][2] for c in curves))
    assert pooled == PUBLISHED_POOLED_FINALS[arm]


def test_all_three_ees_arms_share_their_first_sweep_task_for_task() -> None:
    """The sanity check the log leans on: the sampler-iteration budget cannot matter
    before any training has happened, so the pre-practice counts must agree seed for
    seed -- as counts, which a rounded 25.7% could not have shown."""
    arms = json.loads(ARMS_JSON.read_text())
    firsts = {
        arm: [(c[0][1], c[0][2]) for _, c in sorted(arms[arm].items(), key=lambda kv: int(kv[0]))]
        for arm in ["ees-1000", "ees-10000", "ees-100000"]
    }
    assert firsts["ees-1000"] == firsts["ees-10000"] == firsts["ees-100000"]
    assert sum(solved for solved, _ in firsts["ees-1000"]) == 77


def test_per_seed_counts_returns_the_recorded_integers_not_a_rounded_rate(
    *, tmp_path: Path
) -> None:
    """7/30 rounds to 23.3%, and 23.3% cannot be inverted back to 7 without knowing the
    denominator -- which is the whole reason the counts are read rather than derived."""
    root = tmp_path / "arm"
    directory = root / "ees" / "4"
    directory.mkdir(parents=True)
    (directory / "stats.json").write_text(
        json.dumps({"evaluations": [[0, 7, 30], [100, 30, 30]], "task_name": "t"})
    )
    assert TossingRoomComparison.per_seed_counts(root=root, method="ees") == {
        "4": {0: (7, 30), 100: (30, 30)}
    }
    assert TossingRoomComparison.pooled_endpoints(root=root, method="ees") == {
        "first": (7, 30),
        "final": (30, 30),
        "worst": (30, 30),
    }
