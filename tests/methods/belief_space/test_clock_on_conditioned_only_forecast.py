"""`--pomdp-clock-on-conditioned-only` in the planner's forecast: an imagined try the
robot would not count moves neither the belief nor the clock, so search forecasts the
evidence it will actually observe (coordinator, 2026-11-25).

The routes domains (Two Routes, 3D Sweeping) forecast every try of a learned skill as
conditioned. Under the flag the belief state carries `uncounted_skills`, the skills
whose next pick the evidence rule would keep out; an imagined success clears the mark,
as a real one lets the sampler's next fit rank picks.
"""

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.predicates import BIN_ON_OPPOSITE_SIDE
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import SamplerTrainingState
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence

from .test_clock_on_conditioned_only import SWEEP5, _serving, _tossing

E = CompetenceEvidence


def test_the_method_marks_skills_whose_next_pick_would_not_count() -> None:
    method = _serving(clock_on_conditioned_only=True, reset=False)
    # No sampler has a success yet: every parameterised skill's pick is a uniform draw.
    assert method.uncounted_skills() == frozenset({"sweep3", "gather", SWEEP5})
    sampler = method.sampler(skill_name=SWEEP5, param_dim=1)
    for index in range(2):
        sampler.observe(sampler_input=[1.0, float(index)], success=False)
    sampler.fit()
    assert SWEEP5 in method.uncounted_skills()
    sampler.observe(sampler_input=[1.0, 2.0], success=True)
    sampler.fit()
    assert SWEEP5 not in method.uncounted_skills()


@pytest.mark.parametrize("clock_on_conditioned_only", [False, True])
def test_a_toss_forecast_from_a_one_class_fit_counts_on_the_clock_only_without_the_flag(
    *, clock_on_conditioned_only: bool
) -> None:
    method = _tossing(evidence=E.INFORMED, clock_on_conditioned_only=clock_on_conditioned_only)
    training = SamplerTrainingState(failures=1, fitted_failures=1)
    state = method.pomdp_state.model_copy(update={"sampler_training": {TOSS_SKILL: training}})
    model = method._pomdp_model  # noqa: SLF001
    toss = next(g for g in model.ground_skills if g.skill.name == TOSS_SKILL)
    before = toss.preconditions | {
        GroundAtom(predicate=BIN_ON_OPPOSITE_SIDE, objects=(method.env.bin, method.env.barrier))
    }
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=before),
        state=state,
        action=toss,
    )
    clocks = {branch.pending_examples.get(TOSS_SKILL, 0) for _, branch, _ in outcomes}
    assert clocks == ({0} if clock_on_conditioned_only else {1})
