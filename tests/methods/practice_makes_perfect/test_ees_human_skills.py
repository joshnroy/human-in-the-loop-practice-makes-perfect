"""EES scores a human skill, tracks it and plans it at -log(competence), like a robot
skill. Neither audited reference explorer defines a human-reset pricing model.

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
from tests.practice_cost_fixtures import CostFixtures

PRICE = 2.0


def _build() -> EesMethod:
    env = SweepOffShelf3DEnvironment()
    method = EesMethod(env=env, skill_provider=SweepOffShelf3DSkillProvider(env=env))
    method.configure_practice_accounting(
        accounting=CostFixtures.accounting(human={"reset_cubes": PRICE, "reset_wiper": PRICE})
    )
    return method


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


def test_a_successful_human_skill_is_scored_a_success():
    method = _build()
    _dispatch_and_grant(method=method, after=_state(on_shelf=O.cubes, in_box=()))
    tally = method.practice_outcomes()["reset_cubes"]
    assert (tally.num_successes, tally.num_attempts) == (1, 1)


def test_a_human_skill_that_moved_nothing_is_scored_a_failure():
    method = _build()
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    tally = method.practice_outcomes()["reset_cubes"]
    assert (tally.num_successes, tally.num_attempts) == (0, 1)


def test_a_human_skill_competence_is_tracked_from_the_robots_own_prior():
    method = _build()
    reset = _reset_cubes(method=method)
    prior = method.prior_alpha / (method.prior_alpha + method.prior_beta)
    assert method.human_skill_competence(ground_skill=reset) == pytest.approx(prior)
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    failed = method.human_skill_competence(ground_skill=reset)
    assert failed < prior
    assert method.current_competences()[reset] == pytest.approx(failed)


def test_a_human_skill_is_planned_at_minus_log_competence_alone(*, monkeypatch):
    """Planning cost is -log(competence), exactly as for a robot skill. What a skill
    costs to execute is not something EES estimates (Josh, 2026-11-08), so the cost
    file's human price never reaches EES's plans."""
    method = _build()
    reset = _reset_cubes(method=method)
    _dispatch_and_grant(method=method, after=_state(on_shelf=(), in_box=O.cubes))
    competence = method.human_skill_competence(ground_skill=reset)
    offered = _offered_costs(method=method, monkeypatch=monkeypatch)[reset]
    assert offered == pytest.approx(-math.log(competence))


def test_the_human_price_does_not_change_a_human_skills_planning_cost(*, monkeypatch):
    cheap, dear = _build(), _build()
    dear.configure_practice_accounting(
        accounting=CostFixtures.accounting(human={"reset_cubes": 50.0, "reset_wiper": 50.0})
    )
    reset = _reset_cubes(method=cheap)
    assert _offered_costs(method=cheap, monkeypatch=monkeypatch)[reset] == pytest.approx(
        _offered_costs(method=dear, monkeypatch=monkeypatch)[reset]
    )


def test_a_human_skill_is_never_a_practice_target():
    """The robot cannot practise the human; scoring the human must not make it a
    candidate EES chooses to practise."""
    method = _build()
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

    method = _build()
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
