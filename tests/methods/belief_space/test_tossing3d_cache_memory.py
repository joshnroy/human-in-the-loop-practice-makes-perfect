"""Search identifiers must not keep discarded hypothetical grid posteriors alive."""

import gc
import sys
import weakref

import pytest

from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    InferenceConfig,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState


def _key(*, model: Tossing3DPracticeModel, belief: BayesianSkillBelief) -> object:
    state = Tossing3DBeliefState(skill_beliefs={"example": belief})
    return model.search_cache_key(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=frozenset()),
        summed_cost=0.0,
        belief_state=state,
        horizon=None,
    )


def test_equivalent_grid_posteriors_have_stable_compact_keys() -> None:
    model = Tossing3DPracticeModel()
    belief = create_bayesian_prior(model="global_curve", engine="grid", seed=3, num_particles=16)
    copied = BayesianSkillBelief.model_validate_json(belief.model_dump_json())
    key = _key(model=model, belief=belief)
    assert key == _key(model=model, belief=copied)
    assert key == _key(model=Tossing3DPracticeModel(), belief=copied)
    assert len(model._belief_signature(belief=belief)) == 32
    assert len(belief.state_weights) > 200000
    changed = belief.condition_outcome(success=True)
    assert _key(model=model, belief=changed) != key
    assert _key(model=model, belief=changed.advance_cycle(training_examples=0)) != _key(
        model=model, belief=changed
    )
    changed_config = belief.model_copy(update={"config": InferenceConfig(sigma_eta=0.01)})
    assert _key(model=model, belief=changed_config) != key


def test_model_and_search_key_release_hypothetical_posterior_buffers() -> None:
    model = Tossing3DPracticeModel()
    belief = create_bayesian_prior(model="global_curve", engine="grid", seed=4, num_particles=16)
    references = []
    keys = []
    for _ in range(30):
        belief = belief.condition_outcome(success=True)
        references.append(weakref.ref(belief))
        # Retain the small search keys as Closed would, without keeping the belief.
        posterior_bytes = belief.state_weights
        before = sys.getrefcount(posterior_bytes)
        key = _key(model=model, belief=belief)
        keys.append(key)
        assert sys.getrefcount(posterior_bytes) == before
    del belief
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(set(keys)) == 30
    assert not hasattr(model, "_belief_ids")


@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_cache_identifier_changes_with_cost_posterior(*, engine) -> None:
    model = Tossing3DPracticeModel()
    belief = create_bayesian_prior(model="local_trend", engine=engine, seed=6, num_particles=32)
    assert _key(model=model, belief=belief) != _key(
        model=model, belief=belief.condition_cost(observed_cost=1.0)
    )
