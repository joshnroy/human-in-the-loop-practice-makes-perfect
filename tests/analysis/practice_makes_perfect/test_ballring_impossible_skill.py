"""Tests for the Ball-Ring `PlaceBallOnTable` analysis.

These pin the analysis to the *real* committed runs under
`docs/experiment-logs/2026-08-06-ballring-placeballontable/`, so the numbers quoted in
that log entry cannot drift away from the data without a test failing.

The load-bearing one is `test_every_execution_lands_in_the_uninformative_pool`. `stats.json`
stores only three of the four `SamplerConsultation` pools; `UNINFORMATIVE` is the
*derived* remainder, and it is the whole answer to "which pool do the 3280 executions
land in". If that derivation is ever computed as, say, `attempts - random - informed`
(which silently folds in `NO_SAMPLER`), this fails.
"""

from pathlib import Path

from analysis.practice_makes_perfect.ballring_impossible_skill import BallRingImpossibleSkill

_RESULTS_ROOT = (
    Path(__file__).parents[3] / "docs/experiment-logs/2026-08-06-ballring-placeballontable"
)
_SKILL = "PlaceBallOnTable"


def test_loads_all_ten_committed_seeds():
    runs = BallRingImpossibleSkill.load_runs(results_root=_RESULTS_ROOT)
    assert sorted(runs) == list(range(10))


def test_the_skill_never_succeeds_in_any_seed():
    """0/3280 executions, and 0 in every individual seed -- not one lucky seed."""
    runs = BallRingImpossibleSkill.load_runs(results_root=_RESULTS_ROOT)
    pooled = BallRingImpossibleSkill.pool_totals(runs=runs, skill_name=_SKILL)
    assert pooled.num_attempts == 3280
    assert pooled.num_successes == 0
    for seed in range(10):
        per_seed = BallRingImpossibleSkill.pool_totals(runs={seed: runs[seed]}, skill_name=_SKILL)
        assert per_seed.num_successes == 0
        assert per_seed.num_attempts > 0


def test_every_execution_lands_in_the_uninformative_pool():
    """3280/3280 UNINFORMATIVE: a sampler exists and was consulted, and could not rank.

    Not `NO_SAMPLER` (the skill declares `param_dim = 2`, so a sampler IS built) and not
    `EPSILON_RANDOM` -- with `--exploration-epsilon 0.5` the naive expectation is ~half
    the draws being coin flips, and the real answer is 0/3280, because the
    single-class shortcut returns before `sample` ever consults epsilon.
    """
    runs = BallRingImpossibleSkill.load_runs(results_root=_RESULTS_ROOT)
    pooled = BallRingImpossibleSkill.pool_totals(runs=runs, skill_name=_SKILL)
    assert pooled.num_uninformative_attempts == 3280
    assert pooled.num_random_attempts == 0
    assert pooled.num_informed_attempts == 0
    assert pooled.num_unparameterized_attempts == 0


def test_essentially_every_execution_is_as_the_practice_target():
    """3275 selections against 3280 executions, with 0 declined-perfect and 0 unreachable.

    The point of the assertion: executions are NOT mostly prefix side-effects. At most
    5/3280 could be, which is what makes "EES is deliberately practising an impossible
    skill" the right reading rather than "it happens to walk through it".
    """
    runs = BallRingImpossibleSkill.load_runs(results_root=_RESULTS_ROOT)
    targets = BallRingImpossibleSkill.target_totals(runs=runs, skill_name=_SKILL)
    assert targets.num_selected == 3275
    assert targets.num_declined_perfect == 0
    assert targets.num_unreachable == 0
    pooled = BallRingImpossibleSkill.pool_totals(runs=runs, skill_name=_SKILL)
    assert abs(pooled.num_attempts - targets.num_selected) <= 5


def test_ees_never_learns_to_stop_selecting_it():
    """Selection share RISES over training rather than decaying to zero.

    This is the finding about the competence model: `skip_perfect` drops a skill whose
    measured success rate is exactly 1.0, and there is no symmetric rule for one whose
    rate is exactly 0.0, so a provably impossible skill stays maximally attractive.
    Asserted as "the last third of practice selects it at least as often as the first
    third", pooled over seeds, so it does not hinge on any single cycle.
    """
    runs = BallRingImpossibleSkill.load_runs(results_root=_RESULTS_ROOT)
    shares = BallRingImpossibleSkill.selection_share_per_cycle(runs=runs, skill_name=_SKILL)
    practice = [s for s in shares if s.num_selected_total > 0]
    third = len(practice) // 3
    early = practice[:third]
    late = practice[-third:]
    early_share = sum(s.num_selected for s in early) / sum(s.num_selected_total for s in early)
    late_share = sum(s.num_selected for s in late) / sum(s.num_selected_total for s in late)
    assert late_share >= early_share
    # And it is a majority of all practice targets by the end, not a residual tail.
    assert late_share > 0.5


def test_the_decisive_skill_is_the_contrast_case():
    """PlaceCupWithoutBallOnTable is learnable and reads completely differently.

    Included so the `PlaceBallOnTable` numbers cannot be read as "the instrument reports
    zeros for everything": the same loader on the same runs shows a skill with real
    informed draws and real successes.
    """
    runs = BallRingImpossibleSkill.load_runs(results_root=_RESULTS_ROOT)
    pooled = BallRingImpossibleSkill.pool_totals(runs=runs, skill_name="PlaceCupWithoutBallOnTable")
    assert pooled.num_attempts == 3190
    assert pooled.num_successes == 2002
    assert pooled.num_informed_attempts == 1067
    assert pooled.num_random_attempts == 1041
    assert pooled.num_uninformative_attempts == 1082
