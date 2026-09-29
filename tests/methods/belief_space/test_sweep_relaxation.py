"""Small-instance checks for the proposed exact stock continuation bound."""

from functools import cache

import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.sweep_deployment_model import SweepDeploymentExpectation
from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief


class _UncachedExpectimax(ExpectimaxPlanner):
    def cached_solve_belief_space_expectimax(self, **kwargs):
        return self.solve_belief_space_expectimax(**kwargs)


@cache
def _fixture():
    predicates = tuple(
        Predicate(name=f"P{i}", types=(), holds=lambda s, o: False) for i in range(3)
    )
    atoms = tuple(GroundAtom(predicate=p, objects=()) for p in predicates)
    skills = tuple(
        GroundSkill(
            skill=Skill(
                name=name,
                parameters=(),
                preconditions=frozenset({LiftedAtom(predicate=predicates[i], variables=())}),
                add_effects=frozenset({LiftedAtom(predicate=predicates[i + 1], variables=())}),
                delete_effects=frozenset({LiftedAtom(predicate=predicates[i], variables=())}),
                param_dim=1 if i else 0,
                practice_cost=3 if i else 5,
            ),
            objects=(),
        )
        for i, name in enumerate(("Recover", "Stock"))
    )
    beliefs = {
        skill.skill.name: WeightedHypothesisBelief(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(
                        competence=c, learning_rate=0.1, cost=skill.evaluate_practice_cost()
                    ),
                    probability=0.5,
                )
                for c in (0.2, 0.8)
            )
        )
        for skill in skills
    }
    return skills, atoms, Tossing3DBeliefState(skill_beliefs=beliefs)


@pytest.mark.parametrize("cost_lambda", [None, 0.0, 0.03])
@pytest.mark.parametrize("horizon", [0, 1, 2, 3])
def test_relaxed_value_bounds_exhaustive_search(*, cost_lambda, horizon):
    from hitl_pmp.methods.belief_space.sweep_relaxed_bound import SweepRelaxedStockBound

    skills, atoms, state = _fixture()
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=("Stock",),
        random_competences={"Stock": 0.3},
        deployment=SweepDeploymentExpectation(
            ordered_skills=(skills[1],),
            initial_atoms=frozenset({atoms[1]}),
            goal_atoms=frozenset({atoms[2]}),
            horizon=2,
        ),
        linear_cost_lambda=cost_lambda,
    )
    state = state.model_copy(update={"accumulated_cost": 4.0})
    exact, _ = _UncachedExpectimax(use_model_j=True, observation_probability_weight=0.1).solve(
        environment_state=make_tossing3d_search_state(
            state=state, true_atoms=frozenset({atoms[0]})
        ),
        belief_state=state,
        summed_cost=4,
        horizon=horizon,
        model=model,
        num_samples=1,
    )
    bound = SweepRelaxedStockBound(model=model)
    upper = bound.value(state=state, horizon=horizon)
    assert upper + 2e-14 >= exact
    if horizon == 0:
        assert upper == pytest.approx(exact, rel=0, abs=2e-14)
    assert bound.value(state=state, horizon=horizon) == upper


@pytest.mark.parametrize("engine", ["particle", "grid"])
@pytest.mark.parametrize("evidence", ["all", "non_epsilon", "informed"])
def test_bound_keeps_sampler_context_and_learned_failure_dependencies(*, engine, evidence):
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.sweep_observation_model import SweepBeliefs
    from hitl_pmp.methods.belief_space.sweep_relaxed_bound import SweepRelaxedStockBound
    from hitl_pmp.methods.belief_space.types.belief_state import SamplerTrainingState
    from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
    from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount
    from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace

    skills, atoms, _ = _fixture()
    state = SweepBeliefs.prior(
        skill_names=("Recover", "Stock"),
        trainable_skill_names=("Stock",),
        seed=11,
        num_particles=64,
        config=InferenceConfig(),
        engine=engine,
    )
    state = state.model_copy(
        update={
            "pending_examples": {"Stock": 2, "Recover": 7},
            "sampler_training": {
                "Stock": SamplerTrainingState(
                    successes=1, failures=1, fitted_successes=1, fitted_failures=1
                )
            },
        }
    )
    failures = (
        FailureEffectCount(
            ground_skill=skills[1],
            before_atoms=frozenset({atoms[1]}),
            was_random_exploration=False,
            add_effects=frozenset({atoms[0]}),
            delete_effects=frozenset({atoms[1]}),
        ),
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=("Stock",),
        random_competences={"Stock": 0.1},
        deployment=SweepDeploymentExpectation(
            ordered_skills=(skills[1],),
            initial_atoms=frozenset({atoms[1]}),
            goal_atoms=frozenset({atoms[2]}),
            horizon=3,
        ),
        linear_cost_lambda=0.03,
        competence_evidence=CompetenceEvidence(evidence),
        failure_effect_counts=failures,
    )
    source_signatures = {name: belief.signature() for name, belief in state.skill_beliefs.items()}
    trace = SearchTrace()
    exact, _ = _UncachedExpectimax(use_model_j=True, observation_probability_weight=0.1).solve(
        environment_state=make_tossing3d_search_state(
            state=state, true_atoms=frozenset({atoms[0]})
        ),
        belief_state=state,
        summed_cost=0,
        horizon=3,
        model=model,
        num_samples=1,
        trace=trace,
    )
    bound = SweepRelaxedStockBound(model=model)
    upper = bound.value(state=state, horizon=3)
    assert upper + 2e-14 >= exact
    for event in trace.events:
        if event["event"] == "action_value":
            assert upper + 2e-14 >= event["value"]
    assert {
        name: belief.signature() for name, belief in state.skill_beliefs.items()
    } == source_signatures
    assert state.pending_examples == {"Stock": 2, "Recover": 7}
    assert bound.relaxed.failure_effect_counts == failures
    assert bound.relaxed.competence_evidence is model.competence_evidence
    assert bound.relaxed.random_competences == model.random_competences
    assert tuple(skill.skill.name for skill in bound.relaxed.ground_skills) == ("Stock",)
