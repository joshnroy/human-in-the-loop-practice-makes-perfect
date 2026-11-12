"""When the planner's learning curve steps: at the session boundary, or on every example.

A sampler that refits after every practice attempt (the GP) has already learned by the
next attempt, so the planner's model of it steps per example as well -- in the real
update and in the search's imagined attempts alike.
"""

import argparse

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import BIN_ON_OPPOSITE_SIDE
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import (
    SamplerTrainingState,
    Tossing3DBeliefState,
)
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli


def _method(*, update: str | None, sampler_model: str = "gp") -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    extra = {} if update is None else {"pomdp_learning_update": update}
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=5,
        pomdp_inference_engine="grid",
        sampler_model=sampler_model,
        sampler_max_train_iters=1,
        **extra,
    )
    env.current_state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    return method


def _belief(*, state: Tossing3DBeliefState, name: str) -> BayesianSkillBelief:
    belief = state.skill_beliefs[name]
    assert isinstance(belief, BayesianSkillBelief)
    return belief


def _toss(*, method: Tossing3DPomdpMethod, success: bool) -> None:
    ground = next(g for g in method._pomdp_model.ground_skills if g.skill.name == TOSS_SKILL)  # noqa: SLF001
    method.observe_outcome(ground_skill=ground, success=success, was_random_exploration=False)
    bounds = method.skill_provider.sampler_input_bounds(skill_name=TOSS_SKILL) or (None,)
    row = [1.0 if bound is None else (bound[0] + bound[1]) / 2 for bound in bounds]
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=4, sampler_input=row, success=success
    )


def test_the_default_steps_the_learning_curve_at_the_session_boundary() -> None:
    assert Tossing3DPomdpMethod.model_fields["pomdp_learning_update"].default == "session"
    method = _method(update=None, sampler_model="mlp")
    before = _belief(state=method.pomdp_state, name=TOSS_SKILL)
    _toss(method=method, success=True)
    after = _belief(state=method.pomdp_state, name=TOSS_SKILL)
    assert method.pomdp_state.pending_examples[TOSS_SKILL] == 1
    assert (after.cycle_index, after.total_training_examples) == (
        before.cycle_index,
        before.total_training_examples,
    )


def test_every_real_toss_steps_the_toss_learning_curve_and_nothing_else() -> None:
    method = _method(update="example")
    before = method.pomdp_state
    for count, success in enumerate((False, True), start=1):
        conditioned_only = _method(update="session")
        for earlier in (False, True)[: count - 1]:
            _toss(method=conditioned_only, success=earlier)
        _toss(method=method, success=success)
        toss = _belief(state=method.pomdp_state, name=TOSS_SKILL)
        assert toss.total_training_examples == count
        assert toss.incoming_training_examples == 1
        assert toss.cycle_index == _belief(state=before, name=TOSS_SKILL).cycle_index + count
        assert method.pomdp_state.pending_examples.get(TOSS_SKILL, 0) == 0
        training = method.pomdp_state.sampler_training[TOSS_SKILL]
        assert (training.fitted_successes, training.fitted_failures) == (
            training.successes,
            training.failures,
        )
    assert method.pomdp_state.skill_beliefs[PICK_SKILL] == before.skill_beliefs[PICK_SKILL]
    # One filtered belief per toss is saved for the retrospective smoother.
    assert len(method._belief_history[TOSS_SKILL]) == 2  # noqa: SLF001


def test_the_real_step_is_the_session_step_applied_to_one_example() -> None:
    per_example, per_session = _method(update="example"), _method(update="session")
    for method in (per_example, per_session):
        _toss(method=method, success=True)
    expected = _belief(state=per_session.pomdp_state, name=TOSS_SKILL).advance_cycle(
        training_examples=1
    )
    assert _belief(state=per_example.pomdp_state, name=TOSS_SKILL) == expected


def test_the_session_boundary_after_per_example_steps_is_the_zero_example_step() -> None:
    method = _method(update="example")
    _toss(method=method, success=True)
    before = _belief(state=method.pomdp_state, name=TOSS_SKILL)
    method.end_cycle()
    after = _belief(state=method.pomdp_state, name=TOSS_SKILL)
    assert after.total_training_examples == before.total_training_examples == 1
    assert after.incoming_training_examples == 0


@pytest.mark.parametrize("update", ["session", "example"])
def test_search_imagines_a_toss_under_the_same_rule(*, update: str) -> None:
    method = _method(update=update)
    training = SamplerTrainingState(successes=1, failures=1, fitted_successes=1, fitted_failures=1)
    state = method.pomdp_state.model_copy(update={"sampler_training": {TOSS_SKILL: training}})
    model = method._pomdp_model  # noqa: SLF001
    toss = next(g for g in model.ground_skills if g.skill.name == TOSS_SKILL)
    atoms = toss.preconditions | {
        GroundAtom(predicate=BIN_ON_OPPOSITE_SIDE, objects=(method.env.bin, method.env.barrier))
    }
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=frozenset(atoms)),
        state=state,
        action=toss,
    )
    assert outcomes
    start = _belief(state=state, name=TOSS_SKILL)
    for _, next_state, _ in outcomes:
        imagined = _belief(state=next_state, name=TOSS_SKILL)
        fitted = next_state.sampler_training[TOSS_SKILL]
        if update == "session":
            assert next_state.pending_examples[TOSS_SKILL] == 1
            assert imagined.total_training_examples == start.total_training_examples
            assert fitted.fitted_successes + fitted.fitted_failures == 2
        else:
            assert next_state.pending_examples.get(TOSS_SKILL, 0) == 0
            assert imagined.total_training_examples == start.total_training_examples + 1
            assert fitted.fitted_successes + fitted.fitted_failures == 3
            # Nothing is left for the stopping value's projection but the boundary step.
            projected = _belief(state=refit_belief_state(state=next_state), name=TOSS_SKILL)
            assert projected.total_training_examples == imagined.total_training_examples
        assert next_state.skill_beliefs[PICK_SKILL] == state.skill_beliefs[PICK_SKILL]


def test_a_session_fitted_sampler_cannot_be_modelled_as_learning_per_example() -> None:
    with pytest.raises(ValidationError, match="pomdp_learning_update"):
        _method(update="example", sampler_model="mlp")


def test_the_cli_flag_defaults_to_session_and_reaches_the_method() -> None:
    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    assert parser.parse_args([]).pomdp_learning_update == "session"
    args = parser.parse_args(["--pomdp-learning-update", "example"])
    assert args.pomdp_learning_update == "example"
    with pytest.raises(SystemExit):
        parser.parse_args(["--pomdp-learning-update", "toss"])
