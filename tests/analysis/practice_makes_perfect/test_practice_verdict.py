"""The two gates the amended starvation-versus-inability rule adds, each in isolation.

`PracticeVerdict.classify` is exercised end-to-end, on committed sweeps, through
`tossing3d_practice_diagnosis.py` and `test_tossing3d_practice_diagnosis.py`. What is
pinned *here* is the arithmetic of each new gate on its own: the power threshold, the
plateau test stated so a reader can apply it by eye, and the empty-arm guard.

The cases that drove the amendment ran through the Tossing Room Split practice-pools
report, which retired with that environment in PR #141. Their tallies survive
below as the literal numbers the gates are checked against -- `ThrowRecycling` at the
standard budget (11/56 informed, MDE 20.87 points) against its own 10x arm (901/982,
MDE 6.34 points), and that skill's informed successes by fifths in both arms.
"""

from collections.abc import Sequence

import pytest

from analysis.practice_makes_perfect.practice_verdict import PracticeVerdict


def test_the_power_gate_is_the_equivalence_margin_the_cell_asserts() -> None:
    """The threshold is not a round number chosen for convenience: `inability` claims the
    informed rate sits within `EQUIVALENCE_MARGIN` of the control, and an equivalence claim
    at a 10-point margin is only meaningful if the design could have detected 10 points.

    So the gate is `MDE <= EQUIVALENCE_MARGIN`, with the MDE this project's standard
    `2.801585 * sqrt(p_bar (1 - p_bar) (1/n1 + 1/n2))`."""
    assert PracticeVerdict.EQUIVALENCE_MARGIN == 0.10
    pooled = (11 + 11) / (56 + 57)
    expected = 2.801585 * (pooled * (1 - pooled) * (1 / 56 + 1 / 57)) ** 0.5
    assert PracticeVerdict.minimum_detectable_effect(
        successes_a=11, attempts_a=56, successes_b=11, attempts_b=57
    ) == pytest.approx(expected, rel=1e-9)
    # The standard arm's own MDE: 20.87 points, twice the margin the cell would assert.
    assert PracticeVerdict.minimum_detectable_effect(
        successes_a=11, attempts_a=56, successes_b=11, attempts_b=57
    ) == pytest.approx(0.2087, abs=5e-5)
    assert not PracticeVerdict.has_power(
        successes_a=11, attempts_a=56, successes_b=11, attempts_b=57
    )
    # The 10x arm's, on the same skill: 6.34 points, inside the margin.
    assert PracticeVerdict.minimum_detectable_effect(
        successes_a=901, attempts_a=982, successes_b=203, attempts_b=922
    ) == pytest.approx(0.0634, abs=5e-5)
    assert PracticeVerdict.has_power(
        successes_a=901, attempts_a=982, successes_b=203, attempts_b=922
    )


@pytest.mark.parametrize(
    ("trajectory", "rising"),
    [
        # ThrowRecycling's informed successes by fifths at the standard budget.
        ([0, 0, 2, 3, 6], True),
        # The same skill's, at 10x: the peak is behind it.
        ([99, 197, 211, 196, 198], False),
        ([40, 40, 40, 40, 40], False),
        ([6, 3, 2, 0, 0], False),
        # A single bucket cannot show a trend either way.
        ([7], False),
        ([], False),
    ],
)
def test_a_trajectory_is_still_rising_when_its_last_bucket_is_a_strict_maximum(
    *, trajectory: Sequence[int], rising: bool
) -> None:
    """The plateau test, stated so a reader can apply it by eye: the curve is still rising
    when it ends on a value it has never reached before.

    Deliberately conservative in the direction of *not* concluding inability. A flat but
    noisy curve that happens to peak in its final bucket falls to `indeterminate` rather
    than to `inability`, which is the cheaper error: `inability` is a claim that a
    representation has to change."""
    assert PracticeVerdict.is_still_rising(trajectory=trajectory) is rising


def test_an_empty_arm_supports_no_inference_rather_than_a_verdict() -> None:
    """A skill practiced only before its classifier was ever fitted has zero informed
    draws. Dividing by that would make `inability` fire on a skill that was never asked."""
    cell, reasoning = PracticeVerdict.classify(
        informed_successes=0,
        informed_attempts=0,
        control_successes=11,
        control_attempts=57,
        informed_success_trajectory=[0, 0, 0],
    )
    assert cell == "indeterminate"
    assert "no inference" in reasoning
