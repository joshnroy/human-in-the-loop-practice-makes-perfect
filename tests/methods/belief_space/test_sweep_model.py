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
