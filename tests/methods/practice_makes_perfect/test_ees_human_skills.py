"""EES with `score_human_skills`: a human skill is scored, tracked and priced like a
robot skill. Off by default, so Tossing3D's published EES behaviour is unchanged.

Built on SweepOffShelf3D's symbolic layer only; nothing here touches the simulator.
"""

import math

import pytest

from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    SweepOffShelf3DAtoms,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode

PRICE = 2.0


def _build(*, score_human_skills: bool) -> EesMethod:
    env = SweepOffShelf3DEnvironment()
    return EesMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env, human_reset_practice_cost=PRICE),
        score_human_skills=score_human_skills,
    )


def _state(*, on_shelf, in_box) -> SweepOffShelf3DState:
    named = {("HandEmpty", ("robot",)), ("WiperHome", ("wiper_0",))}
    named |= {("OnShelf", (cube.name,)) for cube in on_shelf}
    named |= {("InBox", (cube.name, O.box.name)) for cube in in_box}
    return SweepOffShelf3DState(
        data={}, abstract_atoms=SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(named))
    )


def _reset_cubes(*, method: EesMethod) -> GroundSkill:
    return method.skill_provider.human_cube_bin_reset_skills()[0]


def _dispatch_and_grant(*, method: EesMethod, after: SweepOffShelf3DState) -> None:
    before = _state(on_shelf=(), in_box=O.cubes)
    episode = _EesEpisode(
        method=method, goal=SweepOffShelf3DPredicates.goal_atoms(), practicing=True
    )
    episode._plan = [_reset_cubes(method=method)]  # noqa: SLF001
    with pytest.raises(HumanCubeBinResetRequested):
        episode.step(state=before)
    method.observe_help_granted(state=after)


def _offered_costs(*, method: EesMethod, monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs["ground_skill_costs"])
        return []

    monkeypatch.setattr(method, "_plan_or_raise", capture)
    method.plan_to(
        init_atoms=method.abstract_state(state=_state(on_shelf=(), in_box=O.cubes)),
        goal=SweepOffShelf3DPredicates.goal_atoms(),
        costs=method.skill_costs(),
        practicing=True,
    )
    return captured


def test_scoring_human_skills_is_on_by_default():
    assert EesMethod.model_fields["score_human_skills"].default is True


def test_off_a_human_skill_is_never_scored_and_costs_its_price(*, monkeypatch):
    method = _build(score_human_skills=False)
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    assert "reset_cubes" not in method.practice_outcomes()
    assert method.current_competences() == {}
    reset = _reset_cubes(method=method)
    assert _offered_costs(method=method, monkeypatch=monkeypatch)[reset] == PRICE


def test_a_successful_human_skill_is_scored_a_success():
    method = _build(score_human_skills=True)
    _dispatch_and_grant(method=method, after=_state(on_shelf=O.cubes, in_box=()))
    tally = method.practice_outcomes()["reset_cubes"]
    assert (tally.num_successes, tally.num_attempts) == (1, 1)


def test_a_human_skill_that_moved_nothing_is_scored_a_failure():
    method = _build(score_human_skills=True)
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    tally = method.practice_outcomes()["reset_cubes"]
    assert (tally.num_successes, tally.num_attempts) == (0, 1)


def test_a_human_skill_competence_is_tracked_from_the_robots_own_prior():
    method = _build(score_human_skills=True)
    reset = _reset_cubes(method=method)
    prior = method.prior_alpha / (method.prior_alpha + method.prior_beta)
    assert method.human_skill_competence(ground_skill=reset) == pytest.approx(prior)
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    failed = method.human_skill_competence(ground_skill=reset)
    assert failed < prior
    assert method.current_competences()[reset] == pytest.approx(failed)


def test_a_human_skill_is_priced_by_competence_plus_the_human_price(*, monkeypatch):
    method = _build(score_human_skills=True)
    reset = _reset_cubes(method=method)
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    competence = method.human_skill_competence(ground_skill=reset)
    offered = _offered_costs(method=method, monkeypatch=monkeypatch)[reset]
    assert offered == pytest.approx(PRICE - math.log(competence))


def test_a_human_skill_is_never_a_practice_target():
    """The robot cannot practise the human; scoring the human must not make it a
    candidate EES chooses to practise."""
    method = _build(score_human_skills=True)
    _dispatch_and_grant(method=method, after=_state(on_shelf=O.cubes, in_box=()))
    assert _reset_cubes(method=method) not in method.choose_practice_target()


def test_a_configured_human_price_is_looked_up_by_the_shelf_skills_own_name():
    """Under step accounting a human skill is charged by its own name. The shelf's are
    `reset_cubes` and `reset_wiper`, never Tossing3D's near/far names."""
    from hitl_pmp.core.practice_costs import (
        ChargeFunction,
        HumanCharge,
        PracticeAccounting,
        PracticeCosts,
    )

    method = _build(score_human_skills=False)
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={
                    "reset_cubes": HumanCharge(cost=ChargeFunction(value=7)),
                    "reset_wiper": HumanCharge(cost=ChargeFunction(value=3)),
                },
            )
        )
    )
    cubes, wiper = method.skill_provider.human_cube_bin_reset_skills()
    assert method.skill_provider.human_skill_name(ground_skill=cubes) == "reset_cubes"
    weight = method._practice_accounting.costs.human_weight  # noqa: SLF001
    assert method.practice_action_cost(ground_skill=cubes) == pytest.approx(weight * 7)
    assert method.practice_action_cost(ground_skill=wiper) == pytest.approx(weight * 3)
