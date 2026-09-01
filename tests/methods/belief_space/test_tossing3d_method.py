import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    mean_competence,
)
from hitl_pmp.planning.grounding import SkillGrounder


def _build(**kwargs: object) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    config = {"pomdp_horizon": 2, "pomdp_practice_cost": 0.001, **kwargs}
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        **config,
    )


def _grounding(*, method: Tossing3DPomdpMethod, name: str) -> GroundSkill:
    universe = SkillGrounder.all_possible_ground_atoms(
        objects=method.objects(), predicates=method.predicates()
    )
    return next(
        skill
        for skill in SkillGrounder.applicable_ground_skills(
            skills=method.skills(), objects=method.objects(), true_atoms=universe
        )
        if skill.skill.name == name
    )


def test_selector_uses_current_symbolic_state_without_starting_simulator() -> None:
    # Keep this seeded grounding regression independent of the CLI sampling default.
    method = _build(pomdp_num_samples=1)
    pick = _grounding(method=method, name=PICK_SKILL)
    selection = method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert selection == [pick]
    assert method.env._backend is None  # noqa: SLF001 (pin lazy simulator construction)


def test_pick_costs_practice_but_does_not_change_toss_belief() -> None:
    method = _build()
    pick = _grounding(method=method, name=PICK_SKILL)
    before = method.pomdp_state
    method.record_action_cost(ground_skill=pick)
    method.observe_outcome(ground_skill=pick, success=True)
    after = method.pomdp_state
    assert after.accumulated_cost == 1.0
    assert after.toss_belief == before.toss_belief
    assert mean_competence(belief=after.pick_belief) > mean_competence(belief=before.pick_belief)
    assert after.pending_pick_examples == 1


def test_toss_evidence_and_training_are_separate_until_refit() -> None:
    method = _build()
    toss = _grounding(method=method, name=TOSS_SKILL)
    before = method.pomdp_state
    method.observe_outcome(ground_skill=toss, success=True, was_random_exploration=False)
    conditioned = method.pomdp_state
    assert mean_competence(belief=conditioned.toss_belief) > mean_competence(
        belief=before.toss_belief
    )
    assert conditioned.pending_training_examples == 0
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=4, sampler_input=[0.0], success=True
    )
    assert method.pomdp_state.pending_training_examples == 1


def test_invalid_method_configuration_is_rejected_early() -> None:
    with pytest.raises(ValidationError):
        _build(pomdp_horizon=-1)


def test_default_practice_policy_does_not_bypass_a_pomdp_stop() -> None:
    method = _build(pomdp_horizon=0)
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    toss = _grounding(method=method, name=TOSS_SKILL)
    goal = Goal(atoms=frozenset(a for a in toss.add_effects if a.predicate.name == "InBin"))
    policy = method.get_practice_policy(task=Task(initial_state=state, goal=goal))
    with pytest.raises(InteractionComplete):
        policy(state)
    assert method.pomdp_state.accumulated_cost == 0.0


def test_reset_cost_is_charged_at_dispatch_without_another_selection() -> None:
    method = _build(ask_for_reset_cube_bin_cost=0.25)
    reset = method.skill_provider.human_cube_bin_reset_skill()
    assert reset is not None
    method.record_action_cost(ground_skill=reset)
    assert method.pomdp_state.accumulated_cost == 0.25


def test_new_practice_session_resets_cost_without_forgetting_learning() -> None:
    method = _build(sampler_max_train_iters=2)
    pick = _grounding(method=method, name=PICK_SKILL)
    toss = _grounding(method=method, name=TOSS_SKILL)
    for success in (True, False):
        method.record_action_cost(ground_skill=pick)
        method.observe_outcome(ground_skill=pick, success=success)
        method.observe_sampler_outcome(
            skill_name=TOSS_SKILL, param_dim=4, sampler_input=[float(success)], success=success
        )
    method.fit_samplers()
    before = method.pomdp_state
    sampler = method.sampler(skill_name=TOSS_SKILL, param_dim=4)
    predictions = sampler.score_inputs(sampler_inputs=[[0.0], [1.0]])
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    task = Task(initial_state=state, goal=Goal(atoms=toss.add_effects))
    method.get_task_policy(task=task)
    method.observe_environment_reset(state=state)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert method.pomdp_state.accumulated_cost == 2

    method.get_practice_policy(task=task)
    assert method.pomdp_state == before.model_copy(update={"accumulated_cost": 0.0})
    assert method.sampler(skill_name=TOSS_SKILL, param_dim=4) is sampler
    assert sampler.score_inputs(sampler_inputs=[[0.0], [1.0]]) == predictions
    method.record_action_cost(ground_skill=pick)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert method.pomdp_state.accumulated_cost == 1
