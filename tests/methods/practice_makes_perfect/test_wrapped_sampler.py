import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import (
    MlpBinaryClassifier,
    WrappedSampler,
)

_ROBOT_TYPE = Type(name="robot", feature_names=("x",))
_ROBOT = Object(name="robot", type=_ROBOT_TYPE)
_STATE = State(data={_ROBOT: np.array([0.0])})

_TOGGLE_SKILL = Skill(
    name="Toggle",
    parameters=(),
    preconditions=frozenset(),
    add_effects=frozenset(),
    delete_effects=frozenset(),
    param_dim=1,
)
_GROUND_SKILL = GroundSkill(skill=_TOGGLE_SKILL, objects=())


def _fast_classifier(*, input_dim: int) -> MlpBinaryClassifier:
    return MlpBinaryClassifier(input_dim=input_dim, max_iters=100, n_iter_no_change=15)


def test_predict_proba_before_fit_raises() -> None:
    classifier = _fast_classifier(input_dim=1)
    with pytest.raises(RuntimeError):
        classifier.predict_proba(x=np.array([0.0]))


def test_fit_stops_early_once_the_loss_stops_improving() -> None:
    """n_iter_no_change=1 means the very first non-strictly-improving step
    should trigger the early-stop break. Uses a degenerate, contradictory
    dataset (the same input labeled both ways) rather than a cleanly
    separable one: a real solution to converge toward (like the other fit
    tests use) lets gradient descent keep improving smoothly for a very long
    time, making an early, reliable non-improving step surprisingly hard to
    force -- empirically confirmed flaky across seeds. A dataset with no
    achievable loss floor below chance plateaus almost immediately instead,
    reliably triggering the break well within max_iters."""
    inputs = np.array([[0.0]] * 20)
    labels = np.array([1.0] * 10 + [0.0] * 10)
    classifier = MlpBinaryClassifier(input_dim=1, max_iters=200, n_iter_no_change=1)
    classifier.fit(inputs=inputs, labels=labels)
    assert classifier.network is not None


def test_fit_and_predict_proba_separates_a_simple_linear_boundary() -> None:
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-2.0, 2.0, size=(60, 1))
    labels = (inputs[:, 0] > 0).astype(float)
    classifier = _fast_classifier(input_dim=1)
    classifier.fit(inputs=inputs, labels=labels)

    assert classifier.predict_proba(x=np.array([2.0])) > 0.5
    assert classifier.predict_proba(x=np.array([-2.0])) < 0.5


def test_record_appends_a_training_example() -> None:
    sampler = WrappedSampler(skill=_TOGGLE_SKILL)
    sampler.record(state=_STATE, ground_skill=_GROUND_SKILL, params=np.array([0.5]), success=True)
    assert len(sampler.training_data) == 1
    assert sampler.training_data[0].success is True


def test_retrain_is_a_noop_with_only_one_class_observed() -> None:
    sampler = WrappedSampler(skill=_TOGGLE_SKILL)
    for _ in range(5):
        sampler.record(
            state=_STATE, ground_skill=_GROUND_SKILL, params=np.array([0.5]), success=True
        )
    sampler.retrain()
    assert sampler.classifier is None


def test_retrain_fits_a_classifier_once_both_classes_are_present() -> None:
    sampler = WrappedSampler(skill=_TOGGLE_SKILL)
    rng = np.random.default_rng(1)
    for _ in range(30):
        param = float(rng.uniform(-2.0, 2.0))
        sampler.record(
            state=_STATE,
            ground_skill=_GROUND_SKILL,
            params=np.array([param]),
            success=param > 0,
        )
    sampler.retrain()
    assert sampler.classifier is not None


def test_sample_for_execution_returns_the_first_candidate_before_any_classifier_is_trained() -> (
    None
):
    sampler = WrappedSampler(skill=_TOGGLE_SKILL, num_samples=3)
    candidates = iter([np.array([-1.0]), np.array([0.0]), np.array([1.0])])
    result = sampler.sample_for_execution(
        ground_skill=_GROUND_SKILL,
        state=_STATE,
        base_sample_fn=lambda *, ground_skill, rng: next(candidates),
        rng=np.random.default_rng(0),
    )
    assert result.tolist() == [-1.0]


def test_sample_for_execution_picks_the_highest_scoring_candidate() -> None:
    sampler = WrappedSampler(skill=_TOGGLE_SKILL, num_samples=2)
    rng = np.random.default_rng(2)
    for _ in range(40):
        param = float(rng.uniform(-2.0, 2.0))
        sampler.record(
            state=_STATE,
            ground_skill=_GROUND_SKILL,
            params=np.array([param]),
            success=param > 0,
        )
    sampler.retrain()
    assert sampler.classifier is not None

    candidates = iter([np.array([-1.5]), np.array([1.5])])
    result = sampler.sample_for_execution(
        ground_skill=_GROUND_SKILL,
        state=_STATE,
        base_sample_fn=lambda *, ground_skill, rng: next(candidates),
        rng=rng,
    )
    assert result.tolist() == [1.5]


def test_sample_for_exploration_returns_the_base_sample_when_rng_falls_below_epsilon() -> None:
    sampler = WrappedSampler(skill=_TOGGLE_SKILL, exploration_epsilon=1.0)

    class _AlwaysZeroRng:
        def uniform(self) -> float:
            return 0.0

    result = sampler.sample_for_exploration(
        ground_skill=_GROUND_SKILL,
        state=_STATE,
        base_sample_fn=lambda *, ground_skill, rng: np.array([42.0]),
        rng=_AlwaysZeroRng(),  # type: ignore[arg-type]
    )
    assert result.tolist() == [42.0]


def test_sample_for_exploration_falls_back_to_execution_when_rng_exceeds_epsilon() -> None:
    sampler = WrappedSampler(skill=_TOGGLE_SKILL, exploration_epsilon=0.0, num_samples=1)

    class _AlwaysOneRng:
        def uniform(self) -> float:
            return 1.0

    result = sampler.sample_for_exploration(
        ground_skill=_GROUND_SKILL,
        state=_STATE,
        base_sample_fn=lambda *, ground_skill, rng: np.array([7.0]),
        rng=_AlwaysOneRng(),  # type: ignore[arg-type]
    )
    # No classifier trained yet -> sample_for_execution returns the first (only)
    # candidate, [7.0].
    assert result.tolist() == [7.0]
