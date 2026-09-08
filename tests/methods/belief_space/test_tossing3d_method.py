import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    mean_competence,
    mean_cost,
    mean_learning_rate,
)
from hitl_pmp.methods.belief_space.types.particle_filter_belief import ParticleFilterBelief
from hitl_pmp.planning.grounding import SkillGrounder


def _build(**kwargs: object) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    reset_cost = float(kwargs.pop("human_reset_practice_cost", 5.0))
    config = {"pomdp_search_depth": 2, **kwargs}
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, human_reset_practice_cost=reset_cost),
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


def test_unit_robot_cost_comes_from_the_shared_skill_provider() -> None:
    method = _build()
    assert {skill.evaluate_practice_cost() for skill in method.skills()} == {1.0}
    assert {
        skill.evaluate_practice_cost() for skill in Tossing3DSkillProvider(env=method.env).skills()
    } == {1.0}


def test_pick_costs_practice_but_does_not_change_toss_belief() -> None:
    method = _build()
    pick = _grounding(method=method, name=PICK_SKILL)
    before = method.pomdp_state
    method.record_action_cost(ground_skill=pick)
    dispatched = method.pomdp_state
    assert dispatched.accumulated_cost == 1.0
    assert dispatched.skill_beliefs == before.skill_beliefs
    method.observe_outcome(ground_skill=pick, success=True)
    after = method.pomdp_state
    assert after.accumulated_cost == 1.0
    assert after.skill_beliefs[TOSS_SKILL] == before.skill_beliefs[TOSS_SKILL]
    assert mean_competence(belief=after.skill_beliefs[PICK_SKILL]) > mean_competence(
        belief=before.skill_beliefs[PICK_SKILL]
    )
    assert after.pending_examples[PICK_SKILL] == 1
    assert isinstance(after.skill_beliefs[PICK_SKILL], ParticleFilterBelief)
    assert isinstance(before.skill_beliefs[PICK_SKILL], ParticleFilterBelief)
    assert abs(mean_cost(belief=after.skill_beliefs[PICK_SKILL]) - 1.0) < abs(
        mean_cost(belief=before.skill_beliefs[PICK_SKILL]) - 1.0
    )


def test_record_action_cost_only_records_realized_cost() -> None:
    method = _build()
    pick = _grounding(method=method, name=PICK_SKILL)
    for _ in range(21):
        method.record_action_cost(ground_skill=pick)
    assert method.pomdp_state.accumulated_cost == 21.0


def test_theta_charts_are_read_only_and_show_reset_beliefs() -> None:
    method = _build(human_reset_practice_cost=0.00001)
    before = method.pomdp_state
    values = method.practice_skill_competences()
    assert values["PickCube (belief mean)"] == 0.5
    assert values["OpenGripper (belief mean)"] == 0.5
    assert values["MoveToTossLocationAndToss (belief mean)"] == mean_competence(
        belief=before.skill_beliefs[TOSS_SKILL]
    )
    assert values["ask_for_reset_cube_bin_only (belief mean)"] == mean_competence(
        belief=before.skill_beliefs[RESET_SKILL]
    )
    assert "STOP" not in values
    rates = method.practice_skill_learning_rates()
    assert rates["PickCube (belief mean)"] == pytest.approx(0.5)
    assert rates["OpenGripper (belief mean)"] == pytest.approx(0.5)
    assert rates["MoveToTossLocationAndToss (belief mean)"] == pytest.approx(0.5)
    assert rates["ask_for_reset_cube_bin_only (belief mean)"] == mean_learning_rate(
        belief=before.skill_beliefs[RESET_SKILL]
    )
    assert method.pomdp_state == before


def test_toss_evidence_and_training_are_separate_until_refit() -> None:
    method = _build()
    toss = _grounding(method=method, name=TOSS_SKILL)
    before = method.pomdp_state
    method.observe_outcome(ground_skill=toss, success=True, was_random_exploration=False)
    conditioned = method.pomdp_state
    assert mean_competence(belief=conditioned.skill_beliefs[TOSS_SKILL]) > mean_competence(
        belief=before.skill_beliefs[TOSS_SKILL]
    )
    assert conditioned.pending_examples.get(TOSS_SKILL, 0) == 0
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=4, sampler_input=[0.0], success=True
    )
    assert method.pomdp_state.pending_examples[TOSS_SKILL] == 1


def test_invalid_method_configuration_is_rejected_early() -> None:
    with pytest.raises(ValidationError):
        _build(pomdp_search_depth=-1)


def test_default_practice_policy_does_not_bypass_a_pomdp_stop() -> None:
    method = _build(pomdp_search_depth=0)
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
    method = _build(human_reset_practice_cost=0.005)
    reset_skill = method.human_skills()[0]
    reset = next(
        skill
        for skill in SkillGrounder.applicable_ground_skills(
            skills=(reset_skill,),
            objects=method.objects(),
            true_atoms=SkillGrounder.all_possible_ground_atoms(
                objects=method.objects(), predicates=method.predicates()
            ),
        )
    )
    method.record_action_cost(ground_skill=reset)
    assert method.pomdp_state.accumulated_cost == 0.005


def test_completed_human_reset_jointly_updates_performance_cost_and_training() -> None:
    observed_cost = 0.00001
    method = _build(human_reset_practice_cost=observed_cost)
    reset_skill = method.human_skills()[0]
    reset = next(
        skill
        for skill in SkillGrounder.applicable_ground_skills(
            skills=(reset_skill,),
            objects=method.objects(),
            true_atoms=SkillGrounder.all_possible_ground_atoms(
                objects=method.objects(), predicates=method.predicates()
            ),
        )
    )
    before = method.pomdp_state.skill_beliefs[RESET_SKILL]

    method.record_action_cost(ground_skill=reset)
    after_dispatch = method.pomdp_state.skill_beliefs[RESET_SKILL]
    assert after_dispatch == before
    reset_state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    method.observe_help_granted(state=reset_state)

    after = method.pomdp_state.skill_beliefs[RESET_SKILL]
    assert isinstance(before, ParticleFilterBelief)
    assert after == before.condition_execution(success=True, observed_cost=observed_cost)
    assert mean_competence(belief=after) > mean_competence(belief=before)
    assert abs(mean_cost(belief=after) - observed_cost) < abs(
        mean_cost(belief=before) - observed_cost
    )
    assert method.pomdp_state.pending_examples[RESET_SKILL] == 1


def test_cost_outside_the_shared_particle_support_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cost observations must be at most"):
        _build(human_reset_practice_cost=20.01)


def test_new_practice_session_resets_cost_without_forgetting_learning(*, tmp_path: Path) -> None:
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
    assert method.pomdp_state.accumulated_cost == 2.0

    method.decision_log = tmp_path / "decisions.jsonl"
    method.get_practice_policy(task=task)
    assert method.pomdp_state == before.model_copy(update={"accumulated_cost": 0.0})
    assert method.sampler(skill_name=TOSS_SKILL, param_dim=4) is sampler
    assert sampler.score_inputs(sampler_inputs=[[0.0], [1.0]]) == predictions
    assert '"event": "session_start"' in method.decision_log.read_text()
    method.record_action_cost(ground_skill=pick)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert method.pomdp_state.accumulated_cost == 1.0
    decision = json.loads(method.decision_log.read_text().splitlines()[-1])
    assert set(decision["learning_rates"]) == {
        "PickCube (belief mean)",
        "MoveToTossLocationAndToss (belief mean)",
        "OpenGripper (belief mean)",
        "ask_for_reset_cube_bin_only (belief mean)",
    }
    assert decision["improvement_potentials"]
    stop_value = method.practice_action_values()["STOP"]
    for skill_name, potential in decision["improvement_potentials"].items():
        assert potential == pytest.approx(method.practice_action_values()[skill_name] - stop_value)
    summary = next(event for event in decision["search"] if event["event"] == "search_summary")
    assert summary["expanded_nodes"] > 0
    assert summary["cache_requests"] >= summary["expanded_nodes"]
    assert summary["chance_outcomes"] > 0


def test_end_cycle_logs_exact_learning_rate_observations(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _build()
    method.decision_log = tmp_path / "decisions.jsonl"
    method._cycle_start_competences = {  # noqa: SLF001 - exercise cycle boundary logging
        name: belief.mean_competence()
        for name, belief in method.pomdp_state.skill_beliefs.items()
    }
    method._pomdp_state = method.pomdp_state.model_copy(  # noqa: SLF001
        update={"pending_examples": {PICK_SKILL: 1}}
    )
    monkeypatch.setattr(Tossing3DPomdpMethod, "observe_environment_reset", lambda self, **_: None)
    monkeypatch.setattr(Tossing3DEnvironment, "get_current_state", lambda self: object())

    method.end_cycle()

    refit = json.loads(method.decision_log.read_text().splitlines()[-1])
    assert refit["event"] == "refit"
    assert refit["learning_rate_observations"] == {PICK_SKILL: 0.0}
    assert refit["learning_rate_observation_counts"] == {PICK_SKILL: 1}
