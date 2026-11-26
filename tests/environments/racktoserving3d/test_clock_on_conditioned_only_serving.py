"""`--pomdp-clock-on-conditioned-only` on the Serving toy, the tests that reset it.

They live here, not beside the rest in tests/methods/belief_space, because the toy's
`hard_reset` imports MuJoCo, which needs the OSMesa backend only this CI job installs.
None of them trains a torch sampler: in a worker that has loaded OSMesa, torch's triton
loads a second LLVM and the process aborts (ci.yml, the MuJoCo job's comment). A Serving
test that does both builds its method without the reset instead.
"""

from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from tests.methods.belief_space.test_clock_on_conditioned_only import (
    SWEEP5,
    C,
    _belief,
    _serving,
    _try,
)


def _sweep5_outcomes(*, method: Tossing3DPomdpMethod, state):
    model = method._pomdp_model  # noqa: SLF001
    sweep5 = next(g for g in model.ground_skills if g.skill.name == SWEEP5)
    return model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=sweep5.preconditions),
        state=state,
        action=sweep5,
    )


def _served(*, atoms) -> bool:
    return "ServedFive" in {atom.predicate.name for atom in atoms}


def test_an_imagined_try_on_a_never_succeeded_skill_moves_neither_belief_nor_clock() -> None:
    method = _serving(clock_on_conditioned_only=True)
    state = method.pomdp_state.model_copy(update={"uncounted_skills": frozenset({SWEEP5})})
    outcomes = _sweep5_outcomes(method=method, state=state)
    assert {_served(atoms=atoms) for _, _, atoms in outcomes} == {False, True}
    for _, branch, atoms in outcomes:
        assert branch.skill_beliefs[SWEEP5] == state.skill_beliefs[SWEEP5]
        assert branch.pending_examples.get(SWEEP5, 0) == 0
        # Its first success is what makes the next pick count.
        assert (SWEEP5 in branch.uncounted_skills) is not _served(atoms=atoms)


def test_once_the_skill_has_succeeded_on_the_branch_imagined_tries_count_as_before() -> None:
    method = _serving(clock_on_conditioned_only=True)
    state = method.pomdp_state.model_copy(update={"uncounted_skills": frozenset({SWEEP5})})
    succeeded = next(
        branch
        for _, branch, atoms in _sweep5_outcomes(method=method, state=state)
        if _served(atoms=atoms)
    )
    assert SWEEP5 not in succeeded.uncounted_skills
    for _, branch, atoms in _sweep5_outcomes(method=method, state=succeeded):
        assert branch.pending_examples[SWEEP5] == 1
        belief = branch.skill_beliefs[SWEEP5]
        assert (belief.cycle_successes, belief.cycle_failures) == (
            (1, 0) if _served(atoms=atoms) else (0, 1)
        )


def test_with_nothing_uncounted_the_forecast_is_the_flag_offs() -> None:
    on = _serving(clock_on_conditioned_only=True)
    off = _serving(clock_on_conditioned_only=False)
    assert on.pomdp_state == off.pomdp_state
    assert _sweep5_outcomes(method=on, state=on.pomdp_state) == _sweep5_outcomes(
        method=off, state=off.pomdp_state
    )


def test_without_the_flag_nothing_is_uncounted_and_dumps_are_unchanged() -> None:
    method = _serving(clock_on_conditioned_only=False)
    assert method.uncounted_skills() == frozenset()
    assert "uncounted_skills" not in method.pomdp_state.model_dump(mode="json")
    assert "clock_on_conditioned_only" not in method._pomdp_model.model_dump(mode="json")  # noqa: SLF001


def test_an_excluded_try_moves_neither_the_belief_nor_the_clock() -> None:
    method = _serving(clock_on_conditioned_only=True)
    before = method.pomdp_state
    # sweep5's first try: no success yet, so a uniform pick, kept out of competence.
    for _ in range(3):
        _try(method=method, name=SWEEP5, consultation=C.UNINFORMATIVE, success=False)
    assert method.pomdp_state.pending_examples.get(SWEEP5, 0) == 0
    assert method.pomdp_state.skill_beliefs[SWEEP5] == before.skill_beliefs[SWEEP5]
    method.end_cycle()
    assert _belief(method=method, name=SWEEP5).total_training_examples == 0


def test_a_conditioned_try_advances_both_as_before() -> None:
    method = _serving(clock_on_conditioned_only=True)
    _try(method=method, name=SWEEP5, consultation=C.INFORMED, success=True)
    assert method.pomdp_state.pending_examples[SWEEP5] == 1
    assert _belief(method=method, name=SWEEP5).cycle_successes == 1
    method.end_cycle()
    assert _belief(method=method, name=SWEEP5).total_training_examples == 1


def test_without_the_flag_an_excluded_try_still_advances_the_clock() -> None:
    method = _serving(clock_on_conditioned_only=False)
    _try(method=method, name=SWEEP5, consultation=C.UNINFORMATIVE, success=False)
    assert method.pomdp_state.pending_examples[SWEEP5] == 1
    assert _belief(method=method, name=SWEEP5).cycle_failures == 0
