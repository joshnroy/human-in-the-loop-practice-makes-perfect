"""Multiple learned skills use the same observation and search rules."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.methods.belief_space.skill_plans_domain import SkillPlansPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis


def _domain():
    return SkillPlansPracticeDomain(
        deployment_horizon=1,
        deployment_plans=(("SkillA",), ("SkillB",)),
        belief_sources=tuple(
            (name, PracticeExampleSource.SAMPLER) for name in ("SkillA", "SkillB")
        ),
    )


def _grounds():
    return tuple(
        GroundSkill(
            skill=Skill(
                name=name,
                parameters=(),
                preconditions=frozenset(),
                add_effects=frozenset(),
                delete_effects=frozenset(),
                param_dim=3,
                practice_cost=1,
            ),
            objects=(),
        )
        for name in ("SkillA", "SkillB")
    )


@pytest.mark.parametrize("name", ["SkillA", "SkillB"])
def test_training_and_competence_are_isolated_per_skill(*, name):
    domain = _domain()
    grounds = _grounds()
    model = Tossing3DPracticeModel(domain=domain, ground_skills=grounds)
    state = make_default_tossing3d_belief(domain=domain, num_particles=16)
    result = model.observe_training_example(state=state, skill_name=name, success=True)
    other = "SkillB" if name == "SkillA" else "SkillA"
    assert result.pending_examples == {name: 1}
    assert result.sampler_training[name].successes == 1
    assert result.sampler_training[other].successes == 0
    updated = model.observe_outcome(
        state=result,
        ground_skill=next(g for g in grounds if g.skill.name == name),
        success=True,
        was_random_exploration=False,
    )
    assert updated.skill_beliefs[other] is state.skill_beliefs[other]
    assert updated.skill_beliefs[name] is not state.skill_beliefs[name]


@pytest.mark.parametrize("index", [0, 1])
def test_search_forecasts_training_for_each_sampler(*, index):
    domain = _domain()
    grounds = _grounds()
    model = Tossing3DPracticeModel(domain=domain, ground_skills=grounds)
    state = make_default_tossing3d_belief(domain=domain, num_particles=16)
    search = make_tossing3d_search_state(state=state, true_atoms=frozenset())
    assert set(grounds) <= set(model.get_valid_actions(environment_state=search))
    outcomes = model.outcomes(environment_state=search, state=state, action=grounds[index])
    name, other = grounds[index].skill.name, grounds[1 - index].skill.name
    assert np.isclose(sum(p for p, _, _ in outcomes), 1)
    for _, after, _ in outcomes:
        assert after.pending_examples == {name: 1}
        assert after.sampler_training[other] == state.sampler_training[other]
        training = after.sampler_training[name]
        assert training.successes + training.failures == 1


def test_deployment_distribution_weights_skill_improvements():
    domain = _domain()
    theta = {
        "SkillA": SkillHypothesis(competence=0, learning_rate=0),
        "SkillB": SkillHypothesis(competence=1, learning_rate=0),
    }
    assert domain.evaluate_theta(theta=theta) == 0.5
    only_a = domain.model_copy(update={"deployment_plans": (("SkillA",),)})
    assert only_a.evaluate_theta(theta=theta) == 0


@pytest.mark.parametrize("plans", [(), ((),), (("Unknown",),), (("SkillA", "SkillA"),)])
def test_invalid_deployment_plans_are_rejected(*, plans):
    with pytest.raises(ValueError):
        SkillPlansPracticeDomain(**(_domain().model_dump() | {"deployment_plans": plans}))
