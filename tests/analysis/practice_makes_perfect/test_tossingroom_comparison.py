"""The first tests under `tests/analysis/`. They exist because
`TossingRoomComparison.wilcoxon_signed_rank` is a hand-rolled significance test, and a
p-value is the one kind of output whose wrongness is invisible on inspection -- a sign
error or a botched tie rule produces a plausible number, not a crash. This project has
already published a wrong p-value once (an unpaired test on a paired design), so the
replacement gets pinned against values derivable by hand rather than trusted.

Expected values below are computed by enumeration on paper, not by running the code and
recording what it said.
"""

from analysis.practice_makes_perfect.tossingroom_comparison import TossingRoomComparison


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
