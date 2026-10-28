"""Count-conditioned competence with one shared Pour parameter sampler."""

import numpy as np

from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
from hitl_pmp.methods.belief_space.pouring_domain import PouringPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief


def test_count_two_example_updates_only_its_competence_clock_and_shared_sampler():
    domain = PouringPracticeDomain.create(payload_count=3, evaluation_counts=(1, 2, 3))
    provider = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    grounds = tuple(provider.ground(name="Pour", request=k) for k in (1, 2, 3))
    model = Tossing3DPracticeModel(domain=domain, ground_skills=grounds)
    state = make_default_tossing3d_belief(domain=domain, num_particles=16)
    result = model.observe_training_example(
        state=state, skill_name="Pour", ground_skill=grounds[1], success=True
    )
    assert result.pending_examples == {"Pour/2": 1}
    assert result.sampler_training["Pour"].successes == 1
    assert set(result.skill_beliefs) >= {"Pour/1", "Pour/2", "Pour/3"}
    assert state.pending_examples == {}


def test_evaluation_value_uses_requested_count_distribution():
    from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis

    domain = PouringPracticeDomain.create(payload_count=3, evaluation_counts=(1, 2, 3))
    theta = {
        name: SkillHypothesis(competence=1, learning_rate=0) for name, _ in domain.belief_sources
    }
    theta["Pour/1"] = SkillHypothesis(competence=0, learning_rate=0)
    assert np.isclose(domain.evaluate_theta(theta=theta), 2 / 3)
    only_one = PouringPracticeDomain.create(payload_count=3, evaluation_counts=(1,))
    assert only_one.evaluate_theta(theta=theta) == 0


def test_count_groundings_have_separate_belief_contracts():
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import skill_belief_model

    domain = PouringPracticeDomain.create(payload_count=3, evaluation_counts=(1, 2, 3))
    provider = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    assert [
        skill_belief_model(
            domain=domain, ground_skill=provider.ground(name="Pour", request=k)
        ).belief_name
        for k in (1, 2, 3)
    ] == ["Pour/1", "Pour/2", "Pour/3"]


def test_shared_sampler_update_requires_count_binding():
    import pytest

    domain = PouringPracticeDomain.create(payload_count=3, evaluation_counts=(1, 2, 3))
    provider = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    model = Tossing3DPracticeModel(
        domain=domain,
        ground_skills=tuple(provider.ground(name="Pour", request=k) for k in (1, 2, 3)),
    )
    state = make_default_tossing3d_belief(domain=domain, num_particles=16)
    with pytest.raises(ValueError, match="selected ground skill"):
        model.observe_training_example(state=state, skill_name="Pour", success=True)


def test_pomdp_initializes_every_count_without_splitting_the_skill():
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = FrozenPourEnvironment()
    provider = FrozenPourSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        pomdp_domain=PouringPracticeDomain.create(payload_count=3, evaluation_counts=(1, 2, 3)),
        pomdp_num_particles=16,
    )
    assert {f"Pour/{k}" for k in (1, 2, 3)} <= set(method.pomdp_state.skill_beliefs)
    assert [s.name for s in method.skills()].count("Pour") == 1
    assert method.sampler(skill_name="Pour", param_dim=3) is method.sampler(
        skill_name="Pour", param_dim=3
    )
