import pickle

import numpy as np

from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

# Rows are [bias, speed, release]; the bias column is dropped (bounds None).
BOUNDS = (None, (100.0, 140.0), (500.0, 850.0))


def _sampler(**kwargs: object) -> LearnedSkillSampler:
    return LearnedSkillSampler(
        skill_name="toss",
        param_dim=2,
        seed=0,
        sampler_model="gp",
        gp_input_bounds=BOUNDS,
        **kwargs,  # type: ignore[arg-type]
    )


def _row(*, speed: float, release: float) -> list[float]:
    return [1.0, speed, release]


def _grid() -> tuple[list[list[float]], list[np.ndarray]]:
    speeds = np.linspace(100.0, 140.0, 10)
    releases = np.linspace(500.0, 850.0, 10)
    candidates = [np.array([s, r]) for s in speeds for r in releases]
    return [_row(speed=float(c[0]), release=float(c[1])) for c in candidates], candidates


def _blob_sampler() -> LearnedSkillSampler:
    """Successes in a blob around (130, 700); failures sampled far from it, all in
    the low-speed half of the box."""
    sampler = _sampler()
    for speed in (126.0, 130.0, 134.0):
        for release in (660.0, 700.0, 740.0):
            sampler.observe(sampler_input=_row(speed=speed, release=release), success=True)
    for speed in (100.0, 106.0, 112.0):
        for release in (520.0, 680.0, 840.0):
            sampler.observe(sampler_input=_row(speed=speed, release=release), success=False)
    return sampler


def _in_blob(*, params: np.ndarray) -> bool:
    return 122.0 <= params[0] <= 138.0 and 620.0 <= params[1] <= 780.0


def test_unfitted_gp_falls_back_to_a_uniform_uninformed_draw() -> None:
    sampler = _sampler()
    rows, candidates = _grid()
    for explore in (True, False):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=explore)
        assert not choice.was_random
        assert not choice.was_informed
    assert not sampler.is_fitted


def test_gp_refits_after_every_observation() -> None:
    sampler = _sampler()
    sampler.observe(sampler_input=_row(speed=120.0, release=700.0), success=True)
    assert sampler.is_fitted
    before = sampler.score_inputs(sampler_inputs=[_row(speed=120.0, release=700.0)])[0]
    sampler.observe(sampler_input=_row(speed=120.0, release=700.0), success=False)
    after = sampler.score_inputs(sampler_inputs=[_row(speed=120.0, release=700.0)])[0]
    assert after < before


def test_eval_picks_inside_the_success_blob() -> None:
    sampler = _blob_sampler()
    rows, candidates = _grid()
    choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False)
    assert _in_blob(params=choice.params)
    assert choice.was_informed and not choice.was_random


def test_eval_lcb_also_picks_inside_the_blob() -> None:
    sampler = _blob_sampler()
    sampler.gp_eval = "lcb"
    rows, candidates = _grid()
    choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False)
    assert _in_blob(params=choice.params)


def test_practice_ucb_prefers_untested_next_to_successes_over_tested_bad() -> None:
    sampler = _blob_sampler()
    untested_near = _row(speed=140.0, release=700.0)  # beside the blob, never tried
    tested_bad = _row(speed=106.0, release=680.0)  # surrounded by observed failures
    ucb = sampler.ucb_scores(sampler_inputs=[untested_near, tested_bad])
    assert ucb[0] > ucb[1]
    rows, candidates = _grid()
    for _ in range(20):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=True)
        assert not choice.was_random and choice.was_informed
        assert choice.params[0] > 115.0


def test_one_miss_inside_a_success_region_does_not_floor_ucb_there() -> None:
    sampler = _blob_sampler()
    sampler.observe(sampler_input=_row(speed=130.0, release=700.0), success=False)
    rows, _ = _grid()
    ucb = sampler.ucb_scores(sampler_inputs=[*rows, _row(speed=130.0, release=700.0)])
    assert ucb[-1] > min(ucb) + 1.0


def test_gp_sampler_pickles_and_scores_identically() -> None:
    sampler = _blob_sampler()
    rows, _ = _grid()
    restored = pickle.loads(pickle.dumps(sampler))
    assert restored.sampler_model == "gp"
    np.testing.assert_allclose(
        restored.score_inputs(sampler_inputs=rows), sampler.score_inputs(sampler_inputs=rows)
    )
    np.testing.assert_allclose(
        restored.ucb_scores(sampler_inputs=rows), sampler.ucb_scores(sampler_inputs=rows)
    )


def test_random_warmup_draws_uninformed_until_enough_labels() -> None:
    sampler = _sampler(gp_random_warmup=3)
    rows, candidates = _grid()
    for k in range(3):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=True)
        assert not choice.was_informed
        sampler.observe(sampler_input=_row(speed=110.0 + k, release=600.0), success=k == 0)
    assert sampler.sample(sampler_inputs=rows, candidates=candidates, explore=True).was_informed
    # evaluation is never held back by the practice warm-up
    fresh = _sampler(gp_random_warmup=99)
    fresh.observe(sampler_input=_row(speed=130.0, release=700.0), success=True)
    assert fresh.sample(sampler_inputs=rows, candidates=candidates, explore=False).was_informed


def test_neutral_prior_keeps_unexplored_space_near_one_half() -> None:
    misses = [(100.0, 520.0), (100.0, 840.0), (140.0, 840.0), (120.0, 520.0)]
    far: dict[float | None, float] = {}
    for prior in (None, 0.5):
        sampler = _sampler(gp_prior_rate=prior)
        for speed, release in misses:
            sampler.observe(sampler_input=_row(speed=speed, release=release), success=False)
        sampler.observe(sampler_input=_row(speed=140.0, release=500.0), success=True)
        far[prior] = sampler.score_inputs(sampler_inputs=[_row(speed=120.0, release=700.0)])[0]
    # far from data: the neutral prior says about one half, the default the low run rate
    assert abs(far[0.5] - 0.5) < 0.1
    assert far[None] < far[0.5] - 0.1
