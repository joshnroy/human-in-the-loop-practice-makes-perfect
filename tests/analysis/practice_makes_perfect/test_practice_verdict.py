"""The amended starvation-versus-inability rule.

The load-bearing cases are pinned against the **committed** sweeps under
`docs/experiment-logs/`, not against invented fixtures, because the defect this module
fixes is a false verdict on real data: PR #127's rule assigns `ThrowRecycling` to
`inability` at the standard budget, and PR #131's 10x arm refutes that directly -- the
same sampler, same seeds, same code, goes from 11/56 informed draws (p = 1.0000) to
901/982 (+69.73pp, p < 0.0001) when only the budget changes.

A fixture could have been written to pass either way. These two tallies could not.

The constructed fixtures below exist for the cases the committed data does not contain:
that `inability` still fires when inability is genuinely demonstrated, and that each of
the two new gates blocks it *on its own* with the other satisfied. A rule that never
concludes anything would pass a test suite made only of the first two cases.
"""

import functools
from collections.abc import Sequence
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.practice_diagnostics import PracticeDiagnostics
from analysis.practice_makes_perfect.practice_verdict import PracticeVerdict
from analysis.practice_makes_perfect.tossingroomsplit_practice_pools import (
    TossingRoomSplitPracticePools,
)
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics

_LOGS = Path(__file__).resolve().parents[3] / "docs" / "experiment-logs"
_STANDARD = _LOGS / "2026-08-06-tossingroomsplit-practice-pools-data" / "ees"
_TEN_TIMES = _LOGS / "2026-08-06-tossingroomsplit-practice-pools-10x-data" / "ees"
_RECYCLING = "ThrowRecycling"
_TRASH = "ThrowTrash"


@functools.lru_cache(maxsize=None)
def _runs(*, method_dir: str) -> tuple[Metrics, ...]:
    """Ten seeds of committed `stats.json`, loaded once for the whole module. Cached
    because every case below reads the same two sweeps and the 10x arm is ~590k lines."""
    return tuple(PracticeDiagnostics.load_runs(method_dir=Path(method_dir)))


def _committed_verdict(*, method_dir: Path, skill_name: str) -> tuple[str, str]:
    """The amended rule applied to one committed sweep, with that skill's **own**
    epsilon-random pool as the control.

    The control has to come from inside the same runs: it is the uniform-draw baseline
    under the identical task distribution, which is what makes the comparison a statement
    about the classifier rather than about how hard the tasks happened to be."""
    runs = list(_runs(method_dir=str(method_dir)))
    pooled = PracticeDiagnostics.totals(runs=runs)[skill_name]
    return PracticeVerdict.classify(
        informed_successes=pooled.num_informed_successes,
        informed_attempts=pooled.num_informed_attempts,
        control_successes=pooled.num_random_successes,
        control_attempts=pooled.num_random_attempts,
        informed_success_trajectory=TossingRoomSplitPracticePools.pool_trajectory(
            runs=runs, skill_name=skill_name, field="num_informed_successes", num_buckets=5
        ),
    )


def _flat_windows(
    *, num_windows: int, informed_per_window: int, successes_per_window: int
) -> list[dict[str, SkillPracticeTally]]:
    """`num_windows` identical practice windows plus the trailing evaluation-only one.

    Every window carries an epsilon-random arm of the same size and the same success
    count, so the control rate equals the informed rate exactly and the only thing the
    caller varies is how much data there is. Written to satisfy #119's tally validator
    rather than around it: the three stored pools stay disjoint subsets of the attempts,
    and the stored successes stay a subset of the successes."""
    window = {
        _RECYCLING: SkillPracticeTally(
            num_attempts=2 * informed_per_window,
            num_successes=2 * successes_per_window,
            num_informed_attempts=informed_per_window,
            num_informed_successes=successes_per_window,
            num_random_attempts=informed_per_window,
            num_random_successes=successes_per_window,
        )
    }
    return [*[dict(window) for _ in range(num_windows)], {}]


def _fixture_verdict(*, windows: list[dict[str, SkillPracticeTally]]) -> tuple[str, str]:
    runs = [
        Metrics(
            evaluations=[(100 * index, 0, 0) for index in range(len(windows))],
            practice_outcomes_per_cycle=windows,
        )
    ]
    pooled = PracticeDiagnostics.totals(runs=runs)[_RECYCLING]
    trajectory = TossingRoomSplitPracticePools.pool_trajectory(
        runs=runs, skill_name=_RECYCLING, field="num_informed_successes", num_buckets=len(windows)
    )
    return PracticeVerdict.classify(
        informed_successes=pooled.num_informed_successes,
        informed_attempts=pooled.num_informed_attempts,
        control_successes=pooled.num_random_successes,
        control_attempts=pooled.num_random_attempts,
        informed_success_trajectory=trajectory,
    )


# ----------------------------------------------------------------- the refuted verdict


def test_the_standard_budget_recycling_tally_is_not_called_inability() -> None:
    """**The case that is wrong on `main`.** `ThrowRecycling` at 2,500 transitions is
    11/56 informed against 11/57 epsilon-random, with informed successes still climbing
    (0, 0, 2, 3, 6 by fifths). PR #131's 10x arm shows this same sampler reaching 901/982,
    so `inability` is refuted by measurement, not merely argued against."""
    cell, reasoning = _committed_verdict(method_dir=_STANDARD, skill_name=_RECYCLING)
    assert cell != "inability"
    assert cell == "indeterminate"
    # Both new gates fail here, and the reasoning has to say which -- "indeterminate"
    # with no reason is the same unhelpful answer as a wrong verdict.
    assert "11/56" in reasoning
    assert "11/57" in reasoning


def test_the_ten_times_budget_recycling_tally_is_called_learned() -> None:
    """The same skill, same seeds, same code, ten times the budget: 901/982 informed
    against 203/922 epsilon-random. A rule that cannot name this outcome would have to
    call the plainest learning on this project `indeterminate`."""
    cell, reasoning = _committed_verdict(method_dir=_TEN_TIMES, skill_name=_RECYCLING)
    assert cell == "learned"
    assert "901/982" in reasoning
    assert "203/922" in reasoning


def test_the_standard_budget_trash_tally_is_called_learned() -> None:
    """`ThrowTrash` is the within-sweep contrast that makes the recycling verdict a
    statement about *that skill's* budget rather than about the standard budget as such:
    208/301 against 61/310 in the very same runs is resolved at 2,500 transitions."""
    cell, _reasoning = _committed_verdict(method_dir=_STANDARD, skill_name=_TRASH)
    assert cell == "learned"


# ------------------------------------------------------- the rule still concludes things


def test_inability_fires_when_a_powered_plateaued_sampler_sits_at_its_control() -> None:
    """The cell must remain reachable, or the amendment has replaced a false verdict with
    no verdict at all.

    500 informed draws against 500 epsilon-random, both at 200 successes, on a flat
    trajectory. 500 is not a hypothetical number on this project: `ThrowRecycling`'s own
    10x arm makes 982 informed draws."""
    cell, reasoning = _fixture_verdict(
        windows=_flat_windows(num_windows=5, informed_per_window=100, successes_per_window=40)
    )
    assert cell == "inability"
    assert "200/500" in reasoning


def test_a_still_rising_final_bucket_blocks_inability_on_its_own() -> None:
    """Power held satisfied, plateau alone withheld. The final bucket carries more
    informed successes than any bucket before it -- the shape `ThrowRecycling` has at the
    standard budget -- and that is the starvation signature, so the cell must not fire."""
    windows = _flat_windows(num_windows=5, informed_per_window=100, successes_per_window=40)
    windows[4] = {
        _RECYCLING: SkillPracticeTally(
            num_attempts=200,
            num_successes=124,
            num_informed_attempts=100,
            num_informed_successes=62,
            num_random_attempts=100,
            num_random_successes=62,
        )
    }
    cell, reasoning = _fixture_verdict(windows=windows)
    assert cell == "indeterminate"
    assert "still rising" in reasoning


def test_too_few_informed_draws_blocks_inability_on_its_own() -> None:
    """Plateau held satisfied, power alone withheld. Exactly the same rates and exactly
    the same flat shape as the firing case above, with a twentieth of the data: 25 draws
    against 25 cannot resolve the 10-point margin the cell asserts, so "the informed draws
    land at the control rate" is a statement about the sample size, not the sampler."""
    cell, reasoning = _fixture_verdict(
        windows=_flat_windows(num_windows=5, informed_per_window=5, successes_per_window=2)
    )
    assert cell == "indeterminate"
    assert "underpowered" in reasoning


# ------------------------------------------------------------------- the two gates alone


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
