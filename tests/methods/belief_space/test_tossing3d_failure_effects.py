"""Failed skill executions teach effects without adding competence evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import SamplerTrainingState
from hitl_pmp.planning.grounding import SkillGrounder


def _method(**kwargs: object) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_num_particles=32,
        pomdp_linear_cost_lambda=0.0003,
        **kwargs,
    )


def _skill(*, method: Tossing3DPomdpMethod, name: str) -> GroundSkill:
    return next(
        skill
        for skill in method._pomdp_model.ground_skills  # noqa: SLF001
        if skill.skill.name == name
    )


def _atoms(*, method: Tossing3DPomdpMethod, names: set[str]) -> frozenset[GroundAtom]:
    return frozenset(
        atom
        for atom in SkillGrounder.all_possible_ground_atoms(
            objects=method.objects(), predicates=method.predicates()
        )
        if atom.predicate.name in names
    )


def _state(*, method: Tossing3DPomdpMethod, atoms: frozenset[GroundAtom]) -> Tossing3DState:
    upstream_names = {"Reachable": "MovableIsDownX", "InBin": "MovableInGoalRegion"}
    return Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(
            (
                upstream_names.get(atom.predicate.name, atom.predicate.name),
                tuple(obj.name for obj in atom.objects[:1])
                if atom.predicate.name == "InBin"
                else tuple(obj.name for obj in atom.objects),
            )
            for atom in atoms
            if atom.predicate.name != "ClosedEmpty"
        ),
    )


def test_toss_failure_distribution_learns_retained_and_released_outcomes_without_extra_evidence(
    *, tmp_path: Path
) -> None:
    method = _method(exploration_epsilon=0.0, decision_log=tmp_path / "effects.jsonl")
    toss = _skill(method=method, name=TOSS_SKILL)
    before = toss.preconditions
    released = _atoms(method=method, names={"HandEmpty", "OnGround"})
    initial_model = method._pomdp_model  # noqa: SLF001
    initial_belief = method.pomdp_state
    search_state = make_tossing3d_search_state(state=initial_belief, true_atoms=before)
    baseline = initial_model.outcomes(
        environment_state=search_state, state=initial_belief, action=toss
    )

    for after in (before, before, released):
        method.observe_symbolic_transition(
            ground_skill=toss,
            before_atoms=before,
            after_atoms=after,
            success=False,
            was_random_exploration=False,
        )

    model = method._pomdp_model  # noqa: SLF001
    assert initial_model.failure_effect_counts == ()
    assert method.pomdp_state is initial_belief
    assert [record.count for record in model.failure_effect_counts] == [2, 1]
    outcomes = model.outcomes(environment_state=search_state, state=initial_belief, action=toss)
    assert outcomes[0] == baseline[0]  # Successful execution stays unchanged.
    assert [atoms for _, _, atoms in outcomes[1:]] == [before, released]
    assert [probability for probability, _, _ in outcomes[1:]] == pytest.approx([
        baseline[1][0] * 2 / 3,
        baseline[1][0] / 3,
    ])
    assert sum(probability for probability, _, _ in outcomes) == pytest.approx(1.0)
    assert outcomes[1][1] is outcomes[2][1]
    assert outcomes[1][1] == baseline[1][1]  # Same single S/F update, cost and training count.
    diagnostics = json.loads((tmp_path / "effects.jsonl").read_text().splitlines()[-1])
    assert diagnostics["before_atoms"] == sorted(map(str, before))
    assert diagnostics["after_atoms"] == sorted(map(str, released))
    assert [row["context_total"] for row in diagnostics["failure_effect_counts"]] == [3, 3]


def test_failure_effects_use_only_matching_context_and_exploration_mode() -> None:
    method = _method(exploration_epsilon=0.5)
    # Epsilon can fire only after fitting both label classes.
    method._pomdp_state = method.pomdp_state.model_copy(  # noqa: SLF001
        update={
            "sampler_training": {
                TOSS_SKILL: SamplerTrainingState(
                    successes=1, failures=1, fitted_successes=1, fitted_failures=1
                )
            }
        }
    )
    toss = _skill(method=method, name=TOSS_SKILL)
    before = toss.preconditions
    released = _atoms(method=method, names={"HandEmpty", "OnGround"})
    method.observe_symbolic_transition(
        ground_skill=toss,
        before_atoms=before,
        after_atoms=released,
        success=False,
        was_random_exploration=True,
    )
    model = method._pomdp_model  # noqa: SLF001
    counts = model.failure_effect_counts
    assert EmpiricalFailureEffects.outcomes(
        counts=counts, ground_skill=toss, true_atoms=before, was_random_exploration=False
    ) == ((1.0, before),)
    assert EmpiricalFailureEffects.outcomes(
        counts=counts, ground_skill=toss, true_atoms=before, was_random_exploration=True
    ) == ((1.0, released),)
    different_context = before | _atoms(method=method, names={"OnGround"})
    assert EmpiricalFailureEffects.outcomes(
        counts=counts,
        ground_skill=toss,
        true_atoms=different_context,
        was_random_exploration=True,
    ) == ((1.0, different_context),)

    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=before),
        state=method.pomdp_state,
        action=toss,
    )
    assert sum(probability for probability, _, _ in outcomes) == pytest.approx(1.0)
    # Policy success/failure, then epsilon-random success/failure.
    assert outcomes[1][2] == before
    assert outcomes[3][2] == released
    assert outcomes[3][0] == pytest.approx(0.5 * 0.75)
    assert outcomes[3][1].skill_beliefs[TOSS_SKILL] == method.pomdp_state.skill_beliefs[TOSS_SKILL]


@pytest.mark.parametrize("skill_name", [PICK_SKILL, OPEN_GRIPPER_SKILL])
def test_binary_failure_branches_share_the_original_belief_update(*, skill_name: str) -> None:
    method = _method()
    skill = _skill(method=method, name=skill_name)
    before = skill.preconditions
    # Failed attempts can change reachability while missing their success effects.
    after = before ^ _atoms(method=method, names={"Reachable"})
    initial_model = method._pomdp_model  # noqa: SLF001
    search_state = make_tossing3d_search_state(state=method.pomdp_state, true_atoms=before)
    baseline = initial_model.outcomes(
        environment_state=search_state, state=method.pomdp_state, action=skill
    )
    for post_atoms in (before, after):
        method.observe_symbolic_transition(
            ground_skill=skill,
            before_atoms=before,
            after_atoms=post_atoms,
            success=False,
            was_random_exploration=False,
        )
    outcomes = method._pomdp_model.outcomes(  # noqa: SLF001
        environment_state=search_state, state=method.pomdp_state, action=skill
    )
    assert outcomes[0] == baseline[0]
    assert sum(probability for probability, _, _ in outcomes) == pytest.approx(1.0)
    assert outcomes[1][1] == outcomes[2][1] == baseline[1][1]
    assert outcomes[1][0] == outcomes[2][0] == pytest.approx(baseline[1][0] / 2)
    assert {outcomes[1][2], outcomes[2][2]} == {before, after}


def test_success_and_hypothetical_search_do_not_train_failure_effects() -> None:
    method = _method(exploration_epsilon=0.0)
    toss = _skill(method=method, name=TOSS_SKILL)
    before = toss.preconditions
    method.observe_symbolic_transition(
        ground_skill=toss,
        before_atoms=before,
        after_atoms=toss.add_effects,
        success=True,
        was_random_exploration=False,
    )
    assert method._pomdp_model.failure_effect_counts == ()  # noqa: SLF001
    method.observe_symbolic_transition(
        ground_skill=toss,
        before_atoms=before,
        after_atoms=before,
        success=False,
        was_random_exploration=False,
    )
    model = method._pomdp_model  # noqa: SLF001
    counts = model.failure_effect_counts
    state = method.pomdp_state
    DeterminizedAStarPlanner(max_iterations=10, seed=0).solve(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=before),
        summed_cost=state.accumulated_cost,
        belief_state=state,
        horizon=0,
        remaining_actions=3,
        model=model,
        num_samples=1,
    )
    assert model.failure_effect_counts is counts
    assert model.failure_effect_counts[0].count == 1
    assert method.pomdp_state is state


@pytest.mark.parametrize("flush", ["before_reset", "end_cycle"])
def test_pending_failure_flush_uses_actual_pre_and_post_atoms_exactly_once(
    *, flush: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    method = _method(decision_log=tmp_path / "flush.jsonl")
    pick = _skill(method=method, name=PICK_SKILL)
    before_atoms = pick.preconditions
    after_atoms = (before_atoms - _atoms(method=method, names={"HandEmpty"})) | _atoms(
        method=method, names={"ClosedEmpty"}
    )
    before = _state(method=method, atoms=before_atoms)
    after = _state(method=method, atoms=after_atoms)
    task = Task(initial_state=before, goal=Goal(atoms=pick.add_effects))
    monkeypatch.setattr(Tossing3DPomdpMethod, "select_skill_to_practice", lambda self, **_: [pick])
    monkeypatch.setattr(
        Tossing3DPomdpMethod,
        "execute_ground_skill",
        lambda self, **_: (LabeledAction(action=np.zeros(1), label="pick"), None),
    )
    policy = method.get_practice_policy(task=task)
    policy(before)
    if flush == "before_reset":
        # The environment deliberately points somewhere stale. Only the explicit
        # state supplied to this hook describes the failed execution.
        monkeypatch.setattr(Tossing3DEnvironment, "get_current_state", lambda self: before)
        method.observe_environment_reset(state=after)
        method.observe_environment_reset(state=before)
    else:
        monkeypatch.setattr(Tossing3DEnvironment, "get_current_state", lambda self: after)
        method.end_cycle()
        method.observe_environment_reset(state=before)

    counts = method._pomdp_model.failure_effect_counts  # noqa: SLF001
    assert len(counts) == 1
    assert counts[0].before_atoms == before_atoms
    assert counts[0].add_effects == after_atoms - before_atoms
    assert counts[0].delete_effects == before_atoms - after_atoms
    assert counts[0].count == 1
    events = [json.loads(line) for line in (tmp_path / "flush.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "outcome" for event in events) == 1
    assert sum(event["event"] == "failure_transition" for event in events) == 1


def test_evaluation_episode_does_not_train_failure_kernel(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _method()
    pick = _skill(method=method, name=PICK_SKILL)
    before = _state(method=method, atoms=pick.preconditions)
    after = _state(method=method, atoms=frozenset())
    monkeypatch.setattr(Tossing3DPomdpMethod, "plan_to", lambda self, **_: [pick])
    monkeypatch.setattr(
        Tossing3DPomdpMethod,
        "execute_ground_skill",
        lambda self, **_: (LabeledAction(action=np.zeros(1), label="pick"), None),
    )
    policy = method.get_task_policy(
        task=Task(initial_state=before, goal=Goal(atoms=pick.add_effects))
    )
    policy(before)
    policy(after)
    assert method._pomdp_model.failure_effect_counts == ()  # noqa: SLF001
