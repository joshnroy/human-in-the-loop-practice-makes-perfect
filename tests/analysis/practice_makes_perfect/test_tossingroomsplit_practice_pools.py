"""The two derived quantities the Tossing Room practice-pool audit turns on.

Both tests exist because a plausible-looking alternative implementation gets the audit's
conclusion wrong: averaging the per-window series instead of summing it hides the single
informed draw that starts the rising curve, and including the trailing evaluation-only
window makes every rising curve end in a fall.
"""

from analysis.practice_makes_perfect.tossingroomsplit_practice_pools import (
    TossingRoomSplitPracticePools,
)
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics

_THROW = "ThrowRecycling"


def _metrics(*, windows: list[dict[str, SkillPracticeTally]]) -> Metrics:
    """A `Metrics` carrying only what these two functions read: the per-window practice
    tallies, plus one `evaluations` entry per window so `window_transitions` lines up."""
    return Metrics(
        evaluations=[(100 * index, 0, 0) for index in range(len(windows))],
        practice_outcomes_per_cycle=windows,
    )


def test_informed_vs_random_reports_no_inference_when_an_arm_is_empty() -> None:
    """A skill practiced only before its classifier was ever fitted has zero informed
    draws. That is a real state, and it must come back as "no inference supported" rather
    than as a p-value -- an empty arm silently reported as p = 1.0 would read as a
    measured null result when nothing was measured at all."""
    runs = [
        _metrics(
            windows=[
                {
                    _THROW: SkillPracticeTally(
                        num_attempts=4,
                        num_successes=1,
                        num_random_attempts=4,
                        num_random_successes=1,
                    )
                },
                {},
            ]
        )
    ]
    result = TossingRoomSplitPracticePools.informed_vs_random(runs=runs, skill_name=_THROW)
    assert result["informed"] == (0, 0)
    assert result["random"] == (1, 4)
    assert result["p"] is None
    assert result["delta_pp"] is None


def test_pool_trajectory_sums_across_seeds_and_drops_the_evaluation_only_window() -> None:
    """One seed's single informed draw must survive into the trajectory, and the trailing
    evaluation-only window must not appear as a structural zero.

    Three windows in, two out: window 2 holds no practice by construction. A mean over the
    two seeds would report 0.5 for the second bucket, which is not a count of anything.
    """
    quiet = _metrics(
        windows=[
            {_THROW: SkillPracticeTally(num_attempts=2, num_successes=0, num_random_attempts=2)},
            {_THROW: SkillPracticeTally(num_attempts=2, num_successes=0, num_random_attempts=2)},
            {},
        ]
    )
    loud = _metrics(
        windows=[
            {_THROW: SkillPracticeTally(num_attempts=2, num_successes=0, num_random_attempts=2)},
            {
                _THROW: SkillPracticeTally(
                    num_attempts=2,
                    num_successes=1,
                    num_informed_attempts=1,
                    num_informed_successes=1,
                    num_random_attempts=1,
                )
            },
            {},
        ]
    )
    trajectory = TossingRoomSplitPracticePools.pool_trajectory(
        runs=[quiet, loud], skill_name=_THROW, field="num_informed_attempts", num_buckets=2
    )
    # Two buckets, not three: the trailing evaluation-only window is excluded.
    assert trajectory == [0, 1]
