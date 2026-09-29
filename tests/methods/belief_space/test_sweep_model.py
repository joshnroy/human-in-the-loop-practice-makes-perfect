"""Sweep deployment integrates repeated attempts under one latent competence."""

from functools import cache

import pytest

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.belief_space.sweep_deployment_model import SweepDeploymentExpectation
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief


@cache
def _chain():
    predicates = [Predicate(name=f"Stage{i}", types=(), holds=lambda s, o: False) for i in range(4)]
    atoms = [GroundAtom(predicate=p, objects=()) for p in predicates]
    skills = tuple(
        GroundSkill(
            skill=Skill(
                name=name,
                parameters=(),
                preconditions=frozenset({LiftedAtom(predicate=predicates[i], variables=())}),
                add_effects=frozenset({LiftedAtom(predicate=predicates[i + 1], variables=())}),
                delete_effects=frozenset({LiftedAtom(predicate=predicates[i], variables=())}),
                param_dim=1,
                practice_cost=1,
            ),
            objects=(),
        )
        for i, name in enumerate(("OpenDrawer", "PickWiper", "Sweep"))
    )
    return skills, frozenset({atoms[0]}), frozenset({atoms[-1]})


def _belief(*, values):
    return WeightedHypothesisBelief(
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=v, learning_rate=0),
                probability=1 / len(values),
            )
            for v in values
        )
    )


def test_shared_latent_competence_is_not_resampled_on_retry():
    skills, initial, goal = _chain()
    beliefs = {s.skill.name: _belief(values=(0, 1)) for s in skills}
    evaluator = SweepDeploymentExpectation(
        ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
    )
    assert evaluator.evaluate(beliefs=beliefs) == pytest.approx(1 / 8)
    assert evaluator.evaluate(
        beliefs={name: _belief(values=(0.5,)) for name in beliefs}
    ) == pytest.approx(0.5)


def test_all_goal_atoms_required_and_horizon_counts_every_action():
    skills, initial, goal = _chain()
    beliefs = {s.skill.name: _belief(values=(1,)) for s in skills}
    for horizon, expected in ((2, 0), (3, 1), (5, 1)):
        evaluator = SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=horizon
        )
        assert evaluator.evaluate(beliefs=beliefs) == expected
    extra = GroundAtom(
        predicate=Predicate(name="MissingCube", types=(), holds=lambda s, o: False), objects=()
    )
    evaluator = SweepDeploymentExpectation(
        ordered_skills=skills, initial_atoms=initial, goal_atoms=goal | {extra}, horizon=5
    )
    assert evaluator.evaluate(beliefs=beliefs) == 0


def test_model_b_attempt_clocks_and_cost_evidence_are_shared():
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.sweep_observation_model import SweepBeliefs

    skills, _, _ = _chain()
    state = SweepBeliefs.prior(
        skill_names=tuple(s.skill.name for s in skills),
        trainable_skill_names=("Sweep",),
        seed=1,
        num_particles=64,
        config=InferenceConfig(),
    )
    models, _ = SweepBeliefs.models(ground_skills=skills, trainable_skill_names=("Sweep",))
    before = state.skill_beliefs["Sweep"].mean_competence()
    observed = models[skills[-1]].observe_outcome(
        state=state, success=True, was_random_exploration=True, observed_cost=3.0
    )
    assert observed.skill_beliefs["Sweep"].mean_competence() == pytest.approx(before)
    assert observed.skill_beliefs["Sweep"].mean_cost() == pytest.approx(3.0, abs=0.5)
    observed = models[skills[-1]].observe_training_example(state=observed, success=True)
    assert observed.pending_examples["Sweep"] == 1
    assert not observed.sampler_training["Sweep"].fitted_mixed_classes
    observed = models[skills[-1]].observe_training_example(state=observed, success=False)
    assert observed.pending_examples["Sweep"] == 2
    assert observed.sampler_training["Sweep"].refitted().fitted_mixed_classes
    fixed = models[skills[0]].observe_outcome(
        state=observed, success=False, was_random_exploration=False
    )
    assert fixed.pending_examples["OpenDrawer"] == 1


def test_generic_model_obeys_provider_effects_and_known_human_only():
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
    from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState

    skills, initial, goal = _chain()
    state = Tossing3DBeliefState(
        skill_beliefs={s.skill.name: _belief(values=(0.2, 0.8)) for s in skills}
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=(),
        random_competences={},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
        linear_cost_lambda=0.01,
    )
    search = make_tossing3d_search_state(state=state, true_atoms=initial)
    assert model.get_valid_actions(environment_state=search) == [skills[0]]
    branches = model.transition_outcomes(
        environment_state=search, practice_action=skills[0], belief_state=state
    )
    assert len(branches) == 2
    assert sum(probability for _, _, probability in branches) == pytest.approx(1)
    assert {next_state.true_atoms for next_state, _, _ in branches} == {
        initial,
        skills[0].add_effects,
    }
    assert all(
        next_state.state.pending_examples["OpenDrawer"] == 1 for next_state, _, _ in branches
    )
    human_model = model.model_copy(update={"human_skill_names": ("OpenDrawer",)})
    human_branches = human_model.transition_outcomes(
        environment_state=search, practice_action=skills[0], belief_state=state
    )
    assert len(human_branches) == 1
    assert human_branches[0][2] == 1
    assert human_branches[0][0].state.skill_beliefs == state.skill_beliefs
    assert model.J(belief_state=state, summed_cost=2, num_samples=1) - model.J(
        belief_state=state, summed_cost=3, num_samples=99
    ) == pytest.approx(0.01)


def test_exact_expectation_matches_full_cartesian_model_integration():
    from itertools import product

    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
    from hitl_pmp.methods.belief_space.types.sweep_theta import SweepTheta

    skills, initial, goal = _chain()
    state = Tossing3DBeliefState(
        skill_beliefs={s.skill.name: _belief(values=(0.2, 0.8)) for s in skills}
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=(),
        random_competences={},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
        linear_cost_lambda=0,
    )
    names = tuple(state.skill_beliefs)
    expected = sum(
        model.evaluate_policy(
            sampled_theta=SweepTheta(
                skills={
                    name: SkillHypothesis(competence=value, learning_rate=0)
                    for name, value in zip(names, values, strict=True)
                }
            )
        )
        / 8
        for values in product((0.2, 0.8), repeat=3)
    )
    assert model.J(belief_state=state, summed_cost=0, num_samples=1) == pytest.approx(expected)


class _ChainProvider(SkillProvider):
    def skills(self):
        return tuple(ground.skill for ground in _chain()[0])

    def predicates(self):
        return tuple({
            atom.predicate
            for ground in _chain()[0]
            for atom in (*ground.preconditions, *ground.add_effects)
        })

    def objects(self):
        return ()

    def types(self):
        return ()

    def sample_params(self, *, ground_skill, rng):
        raise AssertionError("symbolic test must not execute a controller")

    def compute_action(self, *, ground_skill, params, state):
        raise AssertionError("symbolic test must not execute a controller")


def test_method_constructs_without_toss_skills_and_preserves_action_budget():
    from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
    from hitl_pmp.methods.belief_space.sweep_method import SweepPomdpMethod
    from hitl_pmp.methods.practice_makes_perfect.ees_method import STOP_SKILL

    _, initial, goal = _chain()
    method = SweepPomdpMethod(
        env=LightSwitchEnvironment(),
        skill_provider=_ChainProvider(),
        pomdp_search_depth=5,
        pomdp_num_particles=32,
        pomdp_num_samples=1,
        random_competences={name: 0.2 for name in ("OpenDrawer", "PickWiper", "Sweep")},
        deployment_initial_atoms=initial,
        deployment_goal_atoms=goal,
        pomdp_linear_cost_lambda=0.001,
    )
    assert set(method.pomdp_state.skill_beliefs) == {"OpenDrawer", "PickWiper", "Sweep"}
    assert len(method._pomdp_model.ground_skills) == 3
    method.observe_practice_action_budget(remaining_actions=0)
    assert method.select_skill_to_practice(true_atoms=initial) == [STOP_SKILL]

    method.observe_practice_action_budget(remaining_actions=1)
    choice = method.select_skill_to_practice(true_atoms=initial)
    assert len(choice) == 1
    assert choice[0] == STOP_SKILL or choice[0].skill.name == "OpenDrawer"
    assert "OpenDrawer" in method.practice_action_values()


def test_deployment_cache_ignores_only_irrelevant_recovery_beliefs(*, monkeypatch):
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
    from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState

    skills, initial, goal = _chain()
    beliefs = {s.skill.name: _belief(values=(0.2, 0.8)) for s in skills}
    beliefs["Recovery"] = _belief(values=(0.1, 0.9))
    state = Tossing3DBeliefState(skill_beliefs=beliefs)
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=(),
        random_competences={},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
        linear_cost_lambda=0.03,
    )
    evaluated_names = []
    original = SweepDeploymentExpectation.evaluate

    def counted(self, *, beliefs):  # noqa: PLR0917
        evaluated_names.append(set(beliefs))
        return original(self, beliefs=beliefs)

    monkeypatch.setattr(SweepDeploymentExpectation, "evaluate", counted)
    first = model.J(belief_state=state, summed_cost=2, num_samples=1)
    changed = state.model_copy(
        update={
            "skill_beliefs": {**beliefs, "Recovery": _belief(values=(0.9,))},
            "pending_examples": {"Recovery": 19},
        }
    )
    second = model.J(belief_state=changed, summed_cost=3, num_samples=1)
    assert first - second == pytest.approx(0.03)
    assert evaluated_names == [{"OpenDrawer", "PickWiper", "Sweep"}]
    changed_stock = changed.model_copy(update={"pending_examples": {"OpenDrawer": 2}})
    value = model.J(belief_state=changed_stock, summed_cost=0, num_samples=1)
    assert len(evaluated_names) == 2
    projected = refit_belief_state(state=changed_stock)
    assert value == original(model.deployment, beliefs=projected.skill_beliefs)
    changed_deployment = model.model_copy(
        update={"deployment": model.deployment.model_copy(update={"horizon": 2})}
    )
    assert changed_deployment.J(belief_state=state, summed_cost=0, num_samples=1) == 0


@pytest.mark.parametrize("pending", [0, 1, 9])
def test_cached_terminal_value_is_exact_for_particle_forecasts(*, pending):
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.sweep_observation_model import SweepBeliefs
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state

    skills, initial, goal = _chain()
    names = tuple(skill.skill.name for skill in skills)
    state = SweepBeliefs.prior(
        skill_names=names + ("Recovery",),
        trainable_skill_names=names,
        seed=17,
        num_particles=64,
        config=InferenceConfig(),
    )
    state = state.model_copy(
        update={"pending_examples": {name: pending for name in names + ("Recovery",)}}
    )
    deployment = SweepDeploymentExpectation(
        ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=names,
        random_competences={name: 0.25 for name in names},
        deployment=deployment,
        linear_cost_lambda=0.03,
    )
    projected = refit_belief_state(state=state)
    expected = deployment.evaluate(beliefs=projected.skill_beliefs) - 0.03 * 2
    assert model.J(belief_state=state, summed_cost=2, num_samples=1) == expected
    counts = EmpiricalFailureEffects.observe(
        counts=(),
        ground_skill=skills[0],
        before_atoms=initial,
        after_atoms=goal,
        was_random_exploration=False,
    )
    changed = model.model_copy(update={"failure_effect_counts": counts})
    expected_changed = deployment.model_copy(update={"failure_effect_counts": counts}).evaluate(
        beliefs=projected.skill_beliefs
    )
    assert changed.J(belief_state=state, summed_cost=0, num_samples=1) == expected_changed
    assert expected_changed > expected + 0.03 * 2


def test_signature_memo_matches_original_digest_and_does_not_retain_beliefs():
    import gc
    import weakref

    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel

    skills, initial, goal = _chain()
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=(),
        random_competences={},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
    )
    belief = _belief(values=(0.2, 0.8))
    digest = Tossing3DPracticeModel._belief_signature(belief=belief)
    assert model._belief_signature(belief=belief) == digest
    assert model._belief_signature(belief=belief) == digest
    reference = weakref.ref(belief)
    del belief
    gc.collect()
    assert reference() is None
    next_belief = _belief(values=(0.3, 0.7))
    assert model._belief_signature(belief=next_belief) == Tossing3DPracticeModel._belief_signature(
        belief=next_belief
    )
    assert model._belief_signature(belief=next_belief) != digest


def test_canonical_exp22c_grid_engine_is_forwarded_without_changing_cost_prior():
    # EXP-22c B lambda3e-6 seed0 config: grid25x16,1024costparticles,weight0.
    from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
    from hitl_pmp.methods.belief_space.competence_inference import (
        InferenceConfig,
        create_bayesian_prior,
    )
    from hitl_pmp.methods.belief_space.sweep_method import SweepPomdpMethod

    _, initial, goal = _chain()
    method = SweepPomdpMethod(
        env=LightSwitchEnvironment(),
        skill_provider=_ChainProvider(),
        pomdp_search_depth=6,
        pomdp_num_particles=1024,
        pomdp_num_samples=100,
        pomdp_inference_engine="grid",
        pomdp_grid_competence_bins=25,
        pomdp_grid_learning_rate_bins=16,
        pomdp_observation_probability_weight=0.0,
        pomdp_linear_cost_lambda=3e-6,
        random_competences={name: 0.25 for name in ("OpenDrawer", "PickWiper", "Sweep")},
        deployment_initial_atoms=initial,
        deployment_goal_atoms=goal,
    )
    for index, (name, belief) in enumerate(sorted(method.pomdp_state.skill_beliefs.items())):
        expected = create_bayesian_prior(
            model="local_trend",
            engine="grid",
            seed=index,
            num_particles=1024,
            config=InferenceConfig(),
        )
        assert belief.signature() == expected.signature(), name
        assert belief.state_count == 25 * 16
        assert belief.cost_belief.num_particles == 1024
    assert method.pomdp_planner.observation_probability_weight == 0.0


def test_factored_imagined_update_reuses_only_matching_skill_evidence(*, monkeypatch):
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import SkillBeliefModel
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
    from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState

    skills, initial, goal = _chain()
    state = Tossing3DBeliefState(
        skill_beliefs={s.skill.name: _belief(values=(0.2, 0.8)) for s in skills}
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=(),
        random_competences={},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
        linear_cost_lambda=0.01,
    )
    calls = []
    original = SkillBeliefModel.observe_outcome

    def counted(self, **kwargs):  # noqa: PLR0917
        calls.append(kwargs["success"])
        return original(self, **kwargs)

    monkeypatch.setattr(SkillBeliefModel, "observe_outcome", counted)

    def outcomes(*, state):
        return model.outcomes(
            environment_state=make_tossing3d_search_state(state=state, true_atoms=initial),
            state=state,
            action=skills[0],
        )

    first = outcomes(state=state)
    changed = state.model_copy(
        update={
            "accumulated_cost": 7.0,
            "skill_beliefs": {**state.skill_beliefs, "Sweep": _belief(values=(0.3,))},
            "pending_examples": {"Sweep": 4},
        }
    )
    second = outcomes(state=changed)
    assert calls == [True, False]
    for (p1, b1, a1), (p2, b2, a2) in zip(first, second, strict=True):
        assert (p1, a1) == (p2, a2)
        assert b1.skill_beliefs["OpenDrawer"] == b2.skill_beliefs["OpenDrawer"]
        assert b2.skill_beliefs["Sweep"] == changed.skill_beliefs["Sweep"]
        assert b2.pending_examples == {"Sweep": 4, "OpenDrawer": 1}
        assert b2.accumulated_cost == b1.accumulated_cost + 7
    changed_evidence = changed.model_copy(
        update={"skill_beliefs": {**changed.skill_beliefs, "OpenDrawer": _belief(values=(0.4,))}}
    )
    outcomes(state=changed_evidence)
    assert calls == [True, False, True, False]


@pytest.mark.parametrize("engine", ["particle", "grid"])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_exact_caches_preserve_root_choice_and_all_action_values(*, engine, depth):
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.sweep_observation_model import SweepBeliefs
    from hitl_pmp.methods.belief_space.sweep_transition_model import SweepTransitions
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
    from hitl_pmp.methods.belief_space.types.belief_state import SamplerTrainingState
    from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace

    class ReferenceModel(SweepPracticeModel):
        def outcomes(self, *, environment_state, state, action):
            return SweepTransitions.outcomes(
                environment_state=environment_state,
                state=state,
                action=action,
                ground_skills=self.ground_skills,
                effects=self._effects,
                exploration_epsilon=self.exploration_epsilon,
                trainable_skill_names=self.trainable_skill_names,
                human_skill_names=self.human_skill_names,
                random_competences=self.random_competences,
                failure_effect_counts=self.failure_effect_counts,
                competence_evidence=self.competence_evidence,
            )

        def J(self, *, belief_state, summed_cost, num_samples):
            projected = refit_belief_state(state=belief_state)
            value = self.deployment.evaluate(beliefs=projected.skill_beliefs)
            return self.G(policy_value=value, summed_cost=summed_cost)

        def search_cache_key(self, **kwargs):
            # Disable the search transposition table as well as the model caches.
            return object()

    skills, initial, goal = _chain()
    names = tuple(s.skill.name for s in skills)
    state = SweepBeliefs.prior(
        skill_names=names,
        trainable_skill_names=names,
        seed=3,
        num_particles=64,
        config=InferenceConfig(),
        engine=engine,
    )
    # Include both epsilon branches and class-count updates after a mixed fit.
    state = state.model_copy(
        update={
            "sampler_training": {
                name: SamplerTrainingState().observe(success=True).observe(success=False).refitted()
                for name in names
            }
        }
    )
    kwargs = dict(
        ground_skills=skills,
        trainable_skill_names=names,
        random_competences={name: 0.25 for name in names},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills, initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
        linear_cost_lambda=3e-6,
    )
    results = []
    roots = []
    for cls in (ReferenceModel, SweepPracticeModel):
        trace = SearchTrace()
        planner = ExpectimaxPlanner(use_model_j=True, observation_probability_weight=0.0)
        results.append(
            planner.solve(
                environment_state=make_tossing3d_search_state(state=state, true_atoms=initial),
                summed_cost=0,
                belief_state=state,
                horizon=depth,
                model=cls(**kwargs),
                num_samples=1,
                trace=trace,
            )
        )
        roots.append([
            (event["action"], event["value"])
            for event in trace.events
            if event["event"] == "action_value"
        ])
    assert results[0] == results[1]
    assert roots[0] == roots[1]


@pytest.mark.parametrize("cost_lambda", [None, 0.0, 3e-6])
@pytest.mark.parametrize("surprise", [0.0, 0.1])
def test_exact_recovery_suffix_pruning_preserves_value_and_horizon(*, cost_lambda, surprise):
    from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
    from hitl_pmp.methods.belief_space.sweep_expectimax import SweepExpectimaxPlanner
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
    from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState

    skills, initial, goal = _chain()
    # First two skills only restore reachability; only Sweep enters J.
    deployment = SweepDeploymentExpectation(
        ordered_skills=(skills[2],), initial_atoms=skills[1].add_effects, goal_atoms=goal, horizon=3
    )
    state = Tossing3DBeliefState(
        skill_beliefs={skill.skill.name: _belief(values=(0.2, 0.8)) for skill in skills}
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=("Sweep",),
        random_competences={"Sweep": 0.25},
        deployment=deployment,
        linear_cost_lambda=cost_lambda,
    )
    for horizon in (1, 2, 3, 4):
        reference = ExpectimaxPlanner(use_model_j=True, observation_probability_weight=surprise)
        optimized = SweepExpectimaxPlanner(
            use_model_j=True, observation_probability_weight=surprise
        )
        args = dict(
            environment_state=make_tossing3d_search_state(state=state, true_atoms=initial),
            belief_state=state,
            summed_cost=0,
            horizon=horizon,
            model=model,
            num_samples=1,
        )
        expected = reference.solve(**args)
        actual = optimized.solve(**args)
        assert actual == expected
        if horizon >= 2:
            assert optimized.pruned_recovery_suffixes > 0
            assert optimized.next_node < reference.next_node


def test_relaxed_pruning_keeps_learned_failure_shortcuts():
    from hitl_pmp.methods.belief_space.sweep_expectimax import SweepExpectimaxPlanner
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
    from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
    from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount

    skills, initial, goal = _chain()
    state = Tossing3DBeliefState(
        skill_beliefs={skill.skill.name: _belief(values=(0.2, 0.8)) for skill in skills}
    )
    for shortcut in (False, True):
        model = SweepPracticeModel(
            ground_skills=skills,
            trainable_skill_names=("Sweep",),
            random_competences={"Sweep": 0.25},
            deployment=SweepDeploymentExpectation(
                ordered_skills=(skills[2],),
                initial_atoms=skills[1].add_effects,
                goal_atoms=goal,
                horizon=3,
            ),
            failure_effect_counts=(
                FailureEffectCount(
                    ground_skill=skills[0],
                    before_atoms=initial,
                    was_random_exploration=False,
                    add_effects=skills[2].preconditions,
                    delete_effects=frozenset(),
                ),
            )
            if shortcut
            else (),
        )
        planner = SweepExpectimaxPlanner(use_model_j=True)
        planner.solve(
            environment_state=make_tossing3d_search_state(state=state, true_atoms=initial),
            belief_state=state,
            summed_cost=0,
            horizon=0,
            model=model,
            num_samples=1,
        )
        assert planner._deployment_attempt_reachable(atoms=initial, horizon=2) is shortcut
        assert planner._deployment_attempt_reachable(atoms=initial, horizon=3)


@pytest.mark.parametrize("horizon", [2, 3])
def test_full_sweep_root_action_values_match_reference_with_pruning(*, horizon):
    from hitl_pmp.environments.sweep_drawer3d.symbolic import SWEEP_PREDICATES, SweepSymbols
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
    from hitl_pmp.methods.belief_space.sweep_expectimax import SweepExpectimaxPlanner
    from hitl_pmp.methods.belief_space.sweep_model import SweepPracticeModel
    from hitl_pmp.methods.belief_space.sweep_observation_model import SweepBeliefs
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
    from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace

    def atoms(*, names):
        return frozenset(
            GroundAtom(predicate=SWEEP_PREDICATES[name], objects=(SweepSymbols.SCENE,))
            for name in names
        )

    skills = tuple(
        GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,)) for skill in SweepSymbols.skills()
    ) + (SweepSymbols.human_reset(cost=3),)
    initial = atoms(
        names=(
            "HandEmpty",
            "WiperHome",
            "DrawerClosed",
            "DrawerNotOpen",
            "RobotHome",
            "AnyCubeInPile",
            *(f"{name}{i}" for i in range(5) for name in ("InPile", "SweepReachable")),
        )
    )
    current = atoms(
        names=(
            "HoldingWiper",
            "DrawerOpen",
            "DrawerNotClosed",
            "RobotAway",
            *(f"{name}{i}" for i in range(5) for name in ("Loose", "Blocked")),
        )
    )
    goal = atoms(names=tuple(f"InDrawer{i}" for i in range(5)))
    state = SweepBeliefs.prior(
        skill_names=tuple(skill.skill.name for skill in skills),
        trainable_skill_names=SweepSymbols.TRAINABLE,
        seed=0,
        num_particles=1024,
        config=InferenceConfig(),
        engine="grid",
    )
    model = SweepPracticeModel(
        ground_skills=skills,
        trainable_skill_names=SweepSymbols.TRAINABLE,
        human_skill_names=(skills[-1].skill.name,),
        random_competences={name: 0.25 for name in SweepSymbols.TRAINABLE},
        deployment=SweepDeploymentExpectation(
            ordered_skills=skills[:3], initial_atoms=initial, goal_atoms=goal, horizon=5
        ),
        linear_cost_lambda=3e-6,
    )
    results = []
    roots = []
    for cls in (ExpectimaxPlanner, SweepExpectimaxPlanner):
        trace = SearchTrace()
        planner = cls(use_model_j=True)
        results.append(
            planner.solve(
                environment_state=make_tossing3d_search_state(state=state, true_atoms=current),
                belief_state=state,
                summed_cost=0,
                horizon=horizon,
                model=model,
                num_samples=1,
                trace=trace,
            )
        )
        roots.append([
            (event["action"], event["value"])
            for event in trace.events
            if event["event"] == "action_value"
        ])
    assert results[0] == results[1]
    assert roots[0] == roots[1]
