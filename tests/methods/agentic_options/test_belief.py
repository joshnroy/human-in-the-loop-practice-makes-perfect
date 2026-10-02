"""Pin compatibility with the existing Tossing3D belief lifecycle."""

import pytest

from hitl_pmp.methods.agentic_options.belief import TossingLearnerBelief
from hitl_pmp.methods.agentic_options.chain_model import SkillChainModel
from hitl_pmp.methods.agentic_options.types import ClusterState, OptionAction
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
    refit_belief_state,
)

from .support import library


def test_code_outcomes_keep_pending_examples_and_existing_refit_forecast() -> None:
    backend = TossingLearnerBelief(linear_cost_lambda=0.1)
    before = make_default_tossing3d_belief(num_particles=32, seed=9, model="local_trend")
    snapshot = before.model_dump_json()
    succeeded = backend.observe(belief=before, skill=TOSS_SKILL, success=True)
    after = backend.observe(belief=succeeded, skill=TOSS_SKILL, success=False)

    assert before.model_dump_json() == snapshot
    assert after.pending_examples[TOSS_SKILL] == 2
    training = after.sampler_training[TOSS_SKILL]
    assert (training.successes, training.failures) == (1, 1)
    assert (training.fitted_successes, training.fitted_failures) == (0, 0)
    after_snapshot = after.model_dump_json()
    expected = Tossing3DPracticeModel(linear_cost_lambda=0.1).J(
        belief_state=after, summed_cost=2.0, num_samples=1
    )
    assert backend.stop_value(belief=after, cost=2.0) == pytest.approx(expected)
    assert after.model_dump_json() == after_snapshot

    advanced = backend.advance_session(belief=after)
    assert advanced == refit_belief_state(state=after, advance_cycle=True)
    assert advanced.pending_examples == {}
    assert advanced.sampler_training[TOSS_SKILL].fitted_mixed_classes


def test_identical_cluster_branches_condition_real_competence_differently() -> None:
    backend = TossingLearnerBelief()
    chain = SkillChainModel(manifest=library(), learner_belief=backend)
    belief = make_default_tossing3d_belief(num_particles=32, seed=3)
    source = ClusterState(cluster_id="near")
    action = OptionAction(option_id="toss")
    posteriors = {
        outcome: chain.compute_next_belief_state(
            belief_state=belief,
            environment_state=source,
            potential_next_environment_state=ClusterState(cluster_id="near", outcome=outcome),
            practice_action=action,
        )
        for outcome in ("success", "failure")
    }
    assert posteriors["success"].skill_beliefs[TOSS_SKILL].mean_competence() > (
        posteriors["failure"].skill_beliefs[TOSS_SKILL].mean_competence()
    )
    assert all(posterior.pending_examples[TOSS_SKILL] == 1 for posterior in posteriors.values())
    assert belief.pending_examples == {}
