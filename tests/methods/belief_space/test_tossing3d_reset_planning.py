"""Known reset mechanics and symbolic self-loop dominance."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILLS,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import mean_cost, refit_belief_state
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    create_fixed_performance_cost_prior,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief
from hitl_pmp.planning.grounding import SkillGrounder


def _method() -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    return Tossing3DPomdpMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0, pomdp_num_particles=32
    )


def _atom(*, method: Tossing3DPomdpMethod, name: str) -> GroundAtom:
    return next(
        atom
        for atom in SkillGrounder.all_possible_ground_atoms(
            objects=method.objects(), predicates=method.predicates()
        )
        if atom.predicate.name == name
        and (
            not name.endswith("AtSide")
            or atom.objects[-1].name == ("opposite_side" if name == "BinAtSide" else "robot_side")
        )
    )


def _reset(*, method: Tossing3DPomdpMethod, name: str) -> GroundSkill:
    return next(
        skill
        for skill in method._pomdp_model.ground_skills  # noqa: SLF001
        if skill.skill.name == name
        and skill.objects[-2].name == "robot_side"
        and skill.objects[-1].name == "opposite_side"
    )


@pytest.mark.parametrize("reset_name", sorted(RESET_SKILLS))
def test_reset_success_is_known_even_with_zero_performance_posterior(*, reset_name: str) -> None:
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    reset = _reset(method=method, name=reset_name)
    beliefs = dict(method.pomdp_state.skill_beliefs)
    beliefs[reset_name] = create_fixed_performance_cost_prior(
        num_particles=32, seed=10, competence=0.0, learning_rate=0.0
    )
    state = method.pomdp_state.model_copy(update={"skill_beliefs": beliefs})
    before = reset.preconditions | frozenset({_atom(method=method, name="Holding")})
    search_state = make_tossing3d_search_state(state=state, true_atoms=before)
    outcomes = model.outcomes(environment_state=search_state, state=state, action=reset)

    assert len(outcomes) == 1
    probability, next_state, after = outcomes[0]
    assert probability == 1.0
    assert _atom(method=method, name="Holding") not in after
    assert _atom(method=method, name="ClosedEmpty") in after
    assert reset.add_effects <= after
    assert next_state.accumulated_cost == pytest.approx(mean_cost(belief=beliefs[reset_name]))
    assert next_state.skill_beliefs == state.skill_beliefs
    assert next_state.pending_examples == state.pending_examples
    assert (
        model.transition_outcomes(
            environment_state=search_state, belief_state=state, practice_action=reset
        )[0][2]
        == 1.0
    )


@pytest.mark.parametrize("reset_name", sorted(RESET_SKILLS))
def test_real_reset_still_updates_identical_cost_and_performance_filters(
    *, reset_name: str
) -> None:
    method = _method()
    reset = _reset(method=method, name=reset_name)
    before = method.pomdp_state.skill_beliefs[reset_name]
    method.record_action_cost(ground_skill=reset)
    method.observe_help_granted(
        state=Tossing3DState(
            data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
            abstract_atoms=frozenset(),
        )
    )
    expected = before.condition_execution(success=True, observed_cost=5.0)
    assert method.pomdp_state.skill_beliefs[reset_name] == expected
    assert method.pomdp_state.pending_examples.get(reset_name, 0) == 0
    assert method.pomdp_state.accumulated_cost == 5.0
    refitted = refit_belief_state(state=method.pomdp_state)
    assert refitted.skill_beliefs[reset_name] == expected


@pytest.mark.parametrize("gripper", ["HandEmpty", "ClosedEmpty"])
def test_only_symbolic_self_loop_resets_are_omitted(*, gripper: str) -> None:
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    ready = frozenset(
        _atom(method=method, name=name)
        for name in (
            gripper,
            "OnGround",
            "NotHolding",
            "RobotAtSide",
            "CubeAtSide",
            "BinAtSide",
            # A ready cube is graspable; without this atom the reset's GraspClear
            # add-effect makes every destination a non-self-loop, which is the
            # gate working, not this test's subject.
            "GraspClear",
        )
    )
    actions = model.get_valid_actions(
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=ready)
    )
    resets = [action for action in actions if action.skill.name in RESET_SKILLS]
    assert resets  # Moving the bin to the other side is not a self-loop.
    assert all(reset.objects[-1].name == "robot_side" for reset in resets)
    assert {action.skill.name for action in actions if action.skill.name not in RESET_SKILLS} == {
        PICK_SKILL if gripper == "HandEmpty" else OPEN_GRIPPER_SKILL
    }
    for reset_name in RESET_SKILLS:
        assert _reset(method=method, name=reset_name).preconditions <= ready


@pytest.mark.parametrize("condition", ["unreachable", "in_bin", "holding"])
def test_reset_recoveries_remain_available(*, condition: str) -> None:
    method = _method()
    names = {
        "unreachable": {"HandEmpty", "OnGround"},
        "in_bin": {"HandEmpty", "OnGround", "InBin"},
        "holding": {"Holding", "CubeAtSide"},
    }[condition]
    atoms = frozenset(
        _atom(method=method, name=name) for name in (names | {"RobotAtSide", "BinAtSide"})
    )
    actions = method._pomdp_model.get_valid_actions(  # noqa: SLF001
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=atoms)
    )
    assert {action.skill.name for action in actions} >= RESET_SKILLS


@pytest.mark.parametrize("cost_lambda", [0.0, 0.0003])
def test_deleting_a_reset_self_loop_preserves_the_continuation_and_saves_its_cost(
    *, cost_lambda: float
) -> None:
    method = _method()
    model = Tossing3DPracticeModel(
        ground_skills=method._pomdp_model.ground_skills,  # noqa: SLF001
        linear_cost_lambda=cost_lambda,
    )
    pick = next(skill for skill in model.ground_skills if skill.skill.name == PICK_SKILL)
    reset = _reset(method=method, name=sorted(RESET_SKILLS)[0])
    point = WeightedHypothesisBelief(
        hypotheses=(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=1.0, learning_rate=0.0), probability=1.0
            ),
        )
    )
    state = Tossing3DBeliefState(
        skill_beliefs={
            PICK_SKILL: point,
            TOSS_SKILL: point,
            OPEN_GRIPPER_SKILL: point,
            reset.skill.name: create_fixed_performance_cost_prior(
                num_particles=32, seed=10, competence=0.0, learning_rate=0.0
            ),
        }
    )
    ready = make_tossing3d_search_state(
        state=state, true_atoms=pick.preconditions | reset.preconditions | reset.add_effects
    )
    direct_probability, direct_state, direct_atoms = model.outcomes(
        environment_state=ready, state=state, action=pick
    )[0]
    reset_probability, reset_state, reset_atoms = model.outcomes(
        environment_state=ready, state=state, action=reset
    )[0]
    long_probability, long_state, long_atoms = model.outcomes(
        environment_state=make_tossing3d_search_state(state=reset_state, true_atoms=reset_atoms),
        state=reset_state,
        action=pick,
    )[0]

    assert direct_probability == reset_probability == long_probability == 1.0
    assert reset_atoms == ready.true_atoms
    assert long_atoms == direct_atoms
    assert long_state.skill_beliefs == direct_state.skill_beliefs
    assert long_state.pending_examples == direct_state.pending_examples
    saved_cost = mean_cost(belief=state.skill_beliefs[reset.skill.name])
    assert long_state.accumulated_cost - direct_state.accumulated_cost == pytest.approx(saved_cost)
    assert model.J(
        belief_state=direct_state, summed_cost=direct_state.accumulated_cost, num_samples=1
    ) - model.J(
        belief_state=long_state, summed_cost=long_state.accumulated_cost, num_samples=1
    ) == pytest.approx(cost_lambda * saved_cost)
