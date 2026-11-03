"""Kitchen configuration feeds the existing search and learning machinery."""

import numpy as np
import pytest

from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecyclingEnvironment
from hitl_pmp.environments.kitchen_recycling3d.skill_provider import KitchenRecyclingSkillProvider
from hitl_pmp.methods.belief_space.kitchen_recycling_domain import KitchenRecyclingPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis


def _provider(*, name="cube_control"):
    config = KitchenConfig()
    config = KitchenConfig(objects=(config.object(name=name),), required_objects=(name,))
    env = KitchenRecyclingEnvironment(config=config)
    return KitchenRecyclingSkillProvider(env=env)


def _domain(*, name="cube_control"):
    return KitchenRecyclingPracticeDomain.for_provider(provider=_provider(name=name))


def test_only_the_selected_object_has_beliefs_and_no_context_sharing():
    domain = _domain(name="food_can")
    assert domain.shared_beliefs == ()
    assert not domain.deterministic_skills
    assert domain.sampler_skills == {"Toss_food_can_near", "Toss_food_can_far"}
    assert not any("cube_control" in name for name, _ in domain.belief_sources)
    state = make_default_tossing3d_belief(num_particles=16, domain=domain)
    assert set(state.sampler_training) == domain.sampler_skills
    assert set(state.skill_beliefs) == {name for name, _ in domain.belief_sources}


def test_multithing_configuration_is_rejected_by_single_object_model():
    with pytest.raises(ValueError, match="exactly one"):
        KitchenRecyclingPracticeDomain.for_provider(
            provider=KitchenRecyclingSkillProvider(env=KitchenRecyclingEnvironment())
        )


def test_far_deployment_value_uses_only_robot_skills():
    domain = _domain()
    theta = {
        name: SkillHypothesis(competence=1.0, learning_rate=0.0)
        for name in domain.deployment_skills
    }
    assert domain.evaluate_theta(theta=theta) == pytest.approx(1.0)
    theta["Toss_cube_control_far"] = SkillHypothesis(competence=0.0, learning_rate=0.0)
    assert domain.evaluate_theta(theta=theta) == 0.0
    state = make_default_tossing3d_belief(num_particles=32, domain=domain)
    samples = domain.sample_policy_values(
        beliefs=state.skill_beliefs, rng=np.random.default_rng(0), count=10
    )
    assert samples.shape == (10,)
    assert np.all((samples >= 0) & (samples <= 1))


@pytest.mark.parametrize(
    ("pick", "toss", "opened", "expected"),
    [(0.5, 1.0, 0.0, 0.5), (0.5, 1.0, 1.0, 0.875), (1.0, 0.25, 1.0, 0.25)],
)
def test_deployment_surrogate_requires_open_after_failed_pick_and_ends_after_toss(
    *, pick, toss, opened, expected
):
    domain = _domain()
    theta = {
        name: SkillHypothesis(competence=competence, learning_rate=0.0)
        for name, competence in zip(domain.deployment_skills, (pick, toss, opened), strict=True)
    }
    assert domain.evaluate_theta(theta=theta) == pytest.approx(expected)


@pytest.mark.parametrize("context", ["near", "far"])
def test_human_recovery_success_enables_pick_then_toss(*, context):
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state

    provider = _provider()
    method = Tossing3DPomdpMethod(
        env=provider.env,
        skill_provider=provider,
        pomdp_domain=_domain(),
        pomdp_num_particles=16,
        exploration_epsilon=0.0,
    )
    groundings = {ground.skill.name: ground for ground in method._pomdp_model.ground_skills}
    recovery = groundings[f"Recover_cube_control_{context}"]
    pick = groundings[f"Pick_cube_control_{context}"]
    toss = groundings[f"Toss_cube_control_{context}"]
    belief = method.pomdp_state
    atoms = recovery.preconditions
    for action, successor in ((recovery, pick), (pick, toss)):
        outcomes = method._pomdp_model.outcomes(
            environment_state=make_tossing3d_search_state(state=belief, true_atoms=atoms),
            state=belief,
            action=action,
        )
        probability, belief, atoms = outcomes[0]
        assert probability > 0
        assert successor.preconditions.issubset(atoms)


@pytest.mark.parametrize("context", ["near", "far"])
def test_sampler_forecasts_and_failure_effects_stay_in_their_context(*, context):
    from hitl_pmp.core.problem.tasks.types import GroundAtom
    from hitl_pmp.environments.kitchen_recycling3d.predicates import HAND_EMPTY, NEEDS_RECOVERY
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state

    provider = _provider()
    method = Tossing3DPomdpMethod(
        env=provider.env,
        skill_provider=provider,
        pomdp_domain=_domain(),
        pomdp_num_particles=16,
        exploration_epsilon=0.0,
    )
    name = f"Toss_cube_control_{context}"
    other = "Toss_cube_control_far" if context == "near" else "Toss_cube_control_near"
    ground = next(skill for skill in method._pomdp_model.ground_skills if skill.skill.name == name)
    before = ground.preconditions
    released = frozenset({
        GroundAtom(predicate=HAND_EMPTY, objects=(provider.env.robot,)),
        GroundAtom(predicate=NEEDS_RECOVERY, objects=(provider.env.items[0],)),
    })
    initial = method.pomdp_state
    method.observe_symbolic_transition(
        ground_skill=ground,
        before_atoms=before,
        after_atoms=released,
        success=False,
        was_random_exploration=False,
    )
    assert method.pomdp_state is initial
    outcomes = method._pomdp_model.outcomes(
        environment_state=make_tossing3d_search_state(state=initial, true_atoms=before),
        state=initial,
        action=ground,
    )
    assert len(outcomes) == 2
    assert outcomes[1][2] == released
    assert sum(probability for probability, _, _ in outcomes) == pytest.approx(1.0)
    for _, updated, _ in outcomes:
        assert updated.pending_examples == {name: 1}
        assert updated.sampler_training[other] == initial.sampler_training[other]
        assert updated.skill_beliefs[other] == initial.skill_beliefs[other]
