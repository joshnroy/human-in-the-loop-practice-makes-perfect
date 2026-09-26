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
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.planner import BeliefSpacePlanner
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    NON_HUMAN_RESET_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    RESET_SKILLS,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    mean_competence,
    mean_cost,
    mean_learning_rate,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace
from hitl_pmp.methods.belief_space.types.stop_action import STOP_ACTION, StopAction
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
    # The sampled belief and newly available reset choices may change which applicable
    # action wins; this regression is about using the supplied symbolic state lazily.
    method = _build(pomdp_num_samples=1)
    pick = _grounding(method=method, name=PICK_SKILL)
    selection = method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert len(selection) == 1
    assert selection[0].preconditions <= pick.preconditions
    assert selection[0].skill.name != TOSS_SKILL
    assert method.env._backend is None  # noqa: SLF001 (pin lazy simulator construction)


def test_determinized_selector_is_seeded_and_does_not_start_simulator() -> None:
    methods = [
        _build(
            pomdp_num_samples=1,
            pomdp_solver="determinized_astar",
            pomdp_max_search_iterations=4,
        )
        for _ in range(2)
    ]
    selections = []
    for method in methods:
        pick = _grounding(method=method, name=PICK_SKILL)
        selections.append(method.select_skill_to_practice(true_atoms=pick.preconditions))
        assert method.env._backend is None  # noqa: SLF001
    assert selections[0] == selections[1]


@pytest.mark.parametrize("remaining_actions", [0, 1, 3])
def test_determinized_selector_receives_real_action_budget(
    *, remaining_actions: int, tmp_path: Path
) -> None:
    decision_log = tmp_path / "bounded.jsonl"
    method = _build(
        pomdp_num_samples=1,
        pomdp_num_particles=32,
        pomdp_solver="determinized_astar",
        pomdp_search_depth=0,
        pomdp_max_search_iterations=5,
        decision_log=decision_log,
    )
    pick = _grounding(method=method, name=PICK_SKILL)
    method.observe_practice_action_budget(remaining_actions=remaining_actions)
    selection = method.select_skill_to_practice(true_atoms=pick.preconditions)
    decision = json.loads(decision_log.read_text().splitlines()[-1])
    summary = next(event for event in decision["search"] if event["event"] == "search_summary")

    assert decision["remaining_practice_actions"] == remaining_actions
    assert summary["effective_action_horizon"] == remaining_actions
    assert summary["selected_path_depth"] <= remaining_actions
    assert summary["max_depth_reached"] <= remaining_actions
    if remaining_actions == 0:
        assert selection[0].skill.name == "STOP"
        assert summary["action_transitions_evaluated"] == 0
    else:
        assert summary["action_transitions_evaluated"] > 0


def test_expectimax_horizon_does_not_exceed_remaining_practice_actions(*, tmp_path: Path) -> None:
    decision_log = tmp_path / "expectimax_bounded.jsonl"
    method = _build(pomdp_search_depth=5, decision_log=decision_log)
    pick = _grounding(method=method, name=PICK_SKILL)
    method.observe_practice_action_budget(remaining_actions=0)
    assert method.select_skill_to_practice(true_atoms=pick.preconditions)[0].skill.name == "STOP"
    decision = json.loads(decision_log.read_text().splitlines()[-1])
    assert decision["horizon"] == 0


def test_expectimax_uses_model_j_and_records_the_configured_penalty(*, tmp_path: Path) -> None:
    decision_log = tmp_path / "expectimax_exact.jsonl"
    method = _build(
        pomdp_observation_probability_weight=0.001,
        pomdp_search_depth=6,
        decision_log=decision_log,
    )
    assert isinstance(method.pomdp_planner, ExpectimaxPlanner)
    assert method.pomdp_planner.use_model_j
    assert method.pomdp_planner.observation_probability_weight == 0.001
    pick = _grounding(method=method, name=PICK_SKILL)
    method.observe_practice_action_budget(remaining_actions=1)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    decision = json.loads(decision_log.read_text().splitlines()[-1])
    summary = next(event for event in decision["search"] if event["event"] == "search_summary")
    assert decision["horizon"] == 1
    assert summary["max_depth_reached"] == 1
    assert decision["observation_probability_weight"] == 0.001
    assert summary["stop_value_source"] == "model_J"


def test_selector_accepts_injected_planner(*, tmp_path: Path) -> None:
    class InjectedPlanner(BeliefSpacePlanner):  # type: ignore[type-arg]
        name = "injected"
        calls = 0

        def solve(self, **kwargs: object) -> tuple[float, StopAction]:  # type: ignore[override]
            del kwargs
            self.calls += 1
            return 0.25, STOP_ACTION

    planner = InjectedPlanner()
    decision_log = tmp_path / "injected.jsonl"
    method = _build(pomdp_planner=planner, decision_log=decision_log)
    pick = _grounding(method=method, name=PICK_SKILL)

    assert method.select_skill_to_practice(true_atoms=pick.preconditions)[0].skill.name == "STOP"
    assert planner.calls == 1
    decision = json.loads(decision_log.read_text().splitlines()[-1])
    assert decision["solver"] == "injected"


def test_action_value_diagnostics_distinguish_parameterized_reset_destinations() -> None:
    seed_method = _build()
    resets = seed_method.skill_provider.human_cube_bin_reset_skills()

    class ResetValuePlanner(BeliefSpacePlanner):  # type: ignore[type-arg]
        name = "reset_values"

        def solve(self, **kwargs: object) -> tuple[float, StopAction]:  # type: ignore[override]
            trace = kwargs["trace"]
            assert isinstance(trace, SearchTrace)
            trace.record(event="stop_value", node=0, value=0.0)
            for value, reset in enumerate(resets, start=1):
                trace.record(
                    event="action_value",
                    node=0,
                    action=reset.model_dump(mode="json", fallback=str),
                    value=float(value),
                )
            return 0.0, STOP_ACTION

    method = _build(pomdp_planner=ResetValuePlanner())
    pick = _grounding(method=method, name=PICK_SKILL)
    method.select_skill_to_practice(true_atoms=pick.preconditions)

    keys = set(method.practice_action_values())
    reset_keys = {key for key in keys if key.startswith(f"{RESET_SKILL}(")}
    assert len(reset_keys) == 2
    assert any("robot_side" in key for key in reset_keys)
    assert any("opposite_side" in key for key in reset_keys)


def test_the_search_model_grounds_every_reset_mechanism_for_both_destinations() -> None:
    """The belief-space planner chooses the reset side itself: both the human and
    the automatic reset are in its action set once per destination, and both
    destinations are applicable from the same real state."""
    method = _build()
    model = method._pomdp_model  # noqa: SLF001
    # The model grounds over every possible atom, so the robot-side variable is also
    # bound to the opposite side; those groundings are never applicable (RobotAtSide
    # holds only for the robot's own side) and are not destinations.
    resets = [
        g
        for g in model.ground_skills
        if g.skill.name in RESET_SKILLS and g.objects[-2].name == "robot_side"
    ]
    assert sorted((g.skill.name, g.objects[-1].name) for g in resets) == sorted(
        (name, side) for name in RESET_SKILLS for side in ("robot_side", "opposite_side")
    )
    pick = _grounding(method=method, name=PICK_SKILL)
    state = make_tossing3d_search_state(
        state=method.pomdp_state, true_atoms=pick.preconditions | resets[0].preconditions
    )
    offered = {
        (g.skill.name, g.objects[-1].name)
        for g in model.get_valid_actions(environment_state=state)
        if g.skill.name in RESET_SKILLS
    }
    assert {side for _name, side in offered} == {"robot_side", "opposite_side"}


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
    assert after.pending_examples.get(PICK_SKILL, 0) == 1
    assert isinstance(after.skill_beliefs[PICK_SKILL], BayesianSkillBelief)
    assert isinstance(before.skill_beliefs[PICK_SKILL], BayesianSkillBelief)
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
    assert values["PickCube (belief mean)"] == pytest.approx(2 / 3, abs=0.03)
    assert values["OpenGripper (belief mean)"] == pytest.approx(2 / 3, abs=0.03)
    assert values["MoveToTossLocationAndToss (belief mean)"] == mean_competence(
        belief=before.skill_beliefs[TOSS_SKILL]
    )
    assert values["ask_for_reset_cube_bin_only (belief mean)"] == mean_competence(
        belief=before.skill_beliefs[RESET_SKILL]
    )
    assert "STOP" not in values
    rates = method.practice_skill_learning_rates()
    assert rates["PickCube (belief mean)"] == pytest.approx(0.04, abs=0.005)
    assert rates["OpenGripper (belief mean)"] == pytest.approx(0.04, abs=0.005)
    assert rates["MoveToTossLocationAndToss (belief mean)"] == pytest.approx(0.04, abs=0.005)
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
    with pytest.raises(ValidationError):
        _build(pomdp_max_search_iterations=0)
    with pytest.raises(ValidationError):
        _build(pomdp_observation_probability_weight=-0.1)
    with pytest.raises(ValidationError):
        _build(pomdp_solver="unknown")


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


def test_completed_human_reset_jointly_updates_performance_and_cost_without_training() -> None:
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
    assert isinstance(before, BayesianSkillBelief)
    assert after == before.condition_execution(success=True, observed_cost=observed_cost)
    assert mean_competence(belief=after) > mean_competence(belief=before)
    assert abs(mean_cost(belief=after) - observed_cost) < abs(
        mean_cost(belief=before) - observed_cost
    )
    assert method.pomdp_state.pending_examples.get(RESET_SKILL, 0) == 1


def test_completed_non_human_reset_updates_only_its_own_joint_belief() -> None:
    method = _build()
    reset_skill = next(
        skill for skill in method.human_skills() if skill.name == NON_HUMAN_RESET_SKILL
    )
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
    human_before = method.pomdp_state.skill_beliefs[RESET_SKILL]
    automatic_before = method.pomdp_state.skill_beliefs[NON_HUMAN_RESET_SKILL]

    method.record_action_cost(ground_skill=reset)
    method.observe_help_granted(
        state=Tossing3DState(
            data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
            abstract_atoms=frozenset(),
        )
    )

    automatic_after = method.pomdp_state.skill_beliefs[NON_HUMAN_RESET_SKILL]
    assert automatic_after == automatic_before.condition_execution(success=True, observed_cost=5.0)
    assert method.pomdp_state.skill_beliefs[RESET_SKILL] == human_before
    assert method.pomdp_state.pending_examples.get(NON_HUMAN_RESET_SKILL, 0) == 1
    assert method.pomdp_state.pending_examples.get(RESET_SKILL, 0) == 0


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
        "non_human_reset_cube_bin_only (belief mean)",
    }
    assert decision["improvement_potentials"]
    stop_value = method.practice_action_values()["STOP"]
    for skill_name, potential in decision["improvement_potentials"].items():
        assert potential == pytest.approx(method.practice_action_values()[skill_name] - stop_value)
    summary = next(event for event in decision["search"] if event["event"] == "search_summary")
    assert summary["expanded_nodes"] > 0
    assert summary["cache_requests"] >= summary["expanded_nodes"]
    assert summary["chance_outcomes_enumerated"] > 0


def test_end_cycle_logs_training_counts_without_learning_rate_observations(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _build()
    method.decision_log = tmp_path / "decisions.jsonl"
    method._pomdp_state = method.pomdp_state.model_copy(  # noqa: SLF001
        update={"pending_examples": {PICK_SKILL: 1}}
    )
    monkeypatch.setattr(Tossing3DPomdpMethod, "observe_environment_reset", lambda self, **_: None)
    monkeypatch.setattr(Tossing3DEnvironment, "get_current_state", lambda self: object())

    method.end_cycle()

    refit = json.loads(method.decision_log.read_text().splitlines()[-1])
    assert refit["event"] == "refit"
    assert refit["learning_rate_evidence"] == "success_failure_only"
    assert refit["training_examples"] == {PICK_SKILL: 1}
    assert "learning_rate_observations" not in refit


def test_duplicate_reset_has_an_independent_joint_particle_belief() -> None:
    method = _build()
    human = method.pomdp_state.skill_beliefs[RESET_SKILL]
    automatic = method.pomdp_state.skill_beliefs[NON_HUMAN_RESET_SKILL]

    assert isinstance(human, BayesianSkillBelief)
    assert isinstance(automatic, BayesianSkillBelief)
    assert human.signature() != automatic.signature()
    assert set(method.practice_skill_costs()) >= {
        f"{RESET_SKILL} (belief mean)",
        f"{NON_HUMAN_RESET_SKILL} (belief mean)",
    }


def test_a_granted_movables_reset_clears_the_starved_pool_record() -> None:
    """A starved pool is keyed on a symbolic state, but its cause is geometry: after a
    movables reset the bin is somewhere else, so last placement's starvations are no
    longer evidence. Both registries -- EES's selection filter and the search model's
    action mask -- are cleared, not only at a new session."""
    method = _build()
    toss = _grounding(method=method, name=TOSS_SKILL)
    atoms = toss.preconditions
    method.record_starved_parameter_pool(ground_skill=toss, true_atoms=atoms)
    assert method.starved_ground_skills(true_atoms=atoms) == {toss}
    assert method._pomdp_model.starved_pools  # noqa: SLF001
    reset = method.skill_provider.human_cube_bin_reset_skills()[1]
    method.record_action_cost(ground_skill=reset)
    method.observe_help_granted(state=Tossing3DState.model_construct())
    assert method.starved_ground_skills(true_atoms=atoms) == frozenset()
    assert method._pomdp_model.starved_pools == ()  # noqa: SLF001
