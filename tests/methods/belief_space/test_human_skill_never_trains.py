"""A human skill's attempts are evidence about its competence, never training.

Under `local_trend` every OUTCOME attempt is a training example, and each cycle's refit
moves competence by examples x learning rate. That models a robot skill improving with
practice. A person's reliability does not rise because the robot asks more often; in the
pouring planned-carry launch the refit raised Curtis's human-reset belief by more than 0.1
in 11 refits (e.g. 0.385 to 0.653 after 8 attempts, true p = 0.25).
"""

from types import SimpleNamespace

import pytest

from hitl_pmp.methods.belief_space.competence_inference.belief import create_bayesian_prior
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    SkillBeliefModel,
    mean_competence,
    refit_belief_state,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource

NAME = "human_bulk_pour_back"


def _failures_then_a_cycle(*, source: PracticeExampleSource) -> tuple[float, float, dict]:
    state = Tossing3DBeliefState(
        skill_beliefs={
            NAME: create_bayesian_prior(model="local_trend", engine="grid", seed=0, num_particles=8)
        }
    )
    model = SkillBeliefModel(skill=SimpleNamespace(name=NAME), example_source=source)
    for _ in range(8):
        state = model.observe_outcome(state=state, success=False, was_random_exploration=False)
    before = mean_competence(belief=state.skill_beliefs[NAME])
    after = refit_belief_state(state=state, advance_cycle=True)
    return before, mean_competence(belief=after.skill_beliefs[NAME]), dict(state.pending_examples)


def test_a_learning_skills_competence_drifts_up_at_the_cycle_boundary():
    before, after, pending = _failures_then_a_cycle(source=PracticeExampleSource.OUTCOME)
    assert pending[NAME] == 8
    assert after > before + 0.01


def test_a_human_skills_attempts_condition_its_competence_but_never_train_it():
    before, after, pending = _failures_then_a_cycle(source=PracticeExampleSource.NO_TRAINING)
    assert NAME not in pending
    # The cycle's predict step still runs (0.0001 here); a training drift is over 0.01.
    assert after == pytest.approx(before, abs=1e-3)
