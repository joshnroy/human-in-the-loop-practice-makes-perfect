"""EES-always-reset: EES's own practice, plus one human reset to a freshly sampled start
state after every execution of the practice target EES selected.

Symbolic only: the abstraction is stubbed so each step sees exactly the atoms the test
names, and every skill used is parameter-free, so no simulator and no sampler is
involved. Both domains the baseline must run on are covered.
"""

import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.method import (
    HumanCubeBinResetRequested,
    HumanStartResetRequested,
    InteractionComplete,
)
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)
from hitl_pmp.core.problem.human.types import (
    SAMPLED_START_RESET_DESTINATION,
    SAMPLED_START_RESET_SKILL_NAME,
)
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    SweepOffShelf3DAtoms,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode
from hitl_pmp.methods.practice_makes_perfect.practice_target_reset import (
    AlwaysResetAfterPracticeTarget,
)
from hitl_pmp.planning.fast_downward import PlanningFailure

SHELF_PRICE = 2.0
TOSSING_PRICE = 5.0


class _Domain(BaseModel):
    """One domain's method, a target skill, a setup skill that reaches the target's
    preconditions, and a state object to hand the policy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: EesMethod
    target: GroundSkill
    setup: GroundSkill
    state: Any
    price: float


def _rule(*, provider, always_reset: bool) -> AlwaysResetAfterPracticeTarget | None:
    if not always_reset:
        return None
    if not provider.offer_human_reset:  # no domain reset to take the price from
        return AlwaysResetAfterPracticeTarget(cost=provider.human_reset_practice_cost)
    return AlwaysResetAfterPracticeTarget.for_provider(skill_provider=provider)


def _shelf(*, always_reset: bool, offer_human_reset: bool = True) -> _Domain:
    env = SweepOffShelf3DEnvironment()
    provider = SweepOffShelf3DSkillProvider(
        env=env, human_reset_practice_cost=SHELF_PRICE, offer_human_reset=offer_human_reset
    )
    method = EesMethod(
        env=env,
        skill_provider=provider,
        practice_target_reset=_rule(provider=provider, always_reset=always_reset),
    )
    named = {("HandEmpty", ("robot",)), ("WiperHome", ("wiper_0",))}
    named |= {("OnShelf", (cube.name,)) for cube in O.cubes}
    state = SweepOffShelf3DState(
        data={}, abstract_atoms=SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(named))
    )
    return _Domain(
        method=method,
        target=GroundSkill(skill=SweepOffShelf3DSkills.RETURN_WIPER, objects=(O.robot, O.wiper)),
        setup=GroundSkill(skill=SweepOffShelf3DSkills.PICK_WIPER, objects=(O.robot, O.wiper)),
        state=state,
        price=SHELF_PRICE,
    )


def _tossing(*, always_reset: bool, offer_human_reset: bool = True) -> _Domain:
    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(
        env=env, human_reset_practice_cost=TOSSING_PRICE, offer_human_reset=offer_human_reset
    )
    method = EesMethod(
        env=env,
        skill_provider=provider,
        practice_target_reset=_rule(provider=provider, always_reset=always_reset),
    )
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    return _Domain(
        method=method,
        target=GroundSkill(
            skill=Tossing3DSkills.PICK_CUBE,
            objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot, env.bin),
        ),
        setup=GroundSkill(
            skill=Tossing3DSkills.OPEN_GRIPPER, objects=(env.robot, env.cube, env.bin)
        ),
        state=state,
        price=TOSSING_PRICE,
    )


DOMAINS: dict[str, Callable[..., _Domain]] = {"sweepoffshelf3d": _shelf, "tossing3d": _tossing}


class _Atoms:
    """What the stubbed abstraction returns on the next call."""

    def __init__(self) -> None:
        self.now: frozenset[GroundAtom] = frozenset()


@pytest.fixture
def atoms(*, monkeypatch: pytest.MonkeyPatch) -> _Atoms:
    current = _Atoms()
    monkeypatch.setattr(EesMethod, "abstract_state", lambda self, **kwargs: current.now)
    return current


def _select(*, monkeypatch: pytest.MonkeyPatch, candidates: list[GroundSkill]) -> None:
    monkeypatch.setattr(EesMethod, "select_skill_to_practice", lambda self, **kwargs: candidates)


def _episode(*, domain: _Domain) -> _EesEpisode:
    # Practice-only, and the goal is irrelevant: the goal phase is skipped outright.
    return _EesEpisode(method=domain.method, goal=frozenset(), practicing=True, pursue_goal=False)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_executing_the_selected_target_asks_for_exactly_one_start_reset_next_step(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    episode = _episode(domain=domain)

    atoms.now = domain.target.preconditions
    episode.step(state=domain.state)  # executes the target
    with pytest.raises(HumanStartResetRequested) as request:
        episode.step(state=domain.state)
    assert request.value.destination == SAMPLED_START_RESET_DESTINATION
    assert request.value.cost == domain.price
    # Granted; the very next step practises again rather than asking a second time.
    domain.method.observe_help_granted(state=domain.state)
    labeled = episode.step(state=domain.state)
    assert domain.target.skill.name in labeled.label


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_the_reset_is_asked_after_the_targets_outcome_is_observed(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scored against the state the target produced, not the reset scene."""
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    episode = _episode(domain=domain)
    atoms.now = domain.target.preconditions
    episode.step(state=domain.state)

    atoms.now = domain.target.preconditions | domain.target.add_effects
    with pytest.raises(HumanStartResetRequested):
        episode.step(state=domain.state)
    tally = domain.method.practice_outcomes()[domain.target.skill.name]
    assert (tally.num_successes, tally.num_attempts) == (1, 1)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_setup_skills_that_reach_the_targets_preconditions_ask_for_nothing(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    monkeypatch.setattr(EesMethod, "plan_to", lambda self, **kwargs: [domain.setup])
    episode = _episode(domain=domain)

    atoms.now = domain.setup.preconditions  # the target is not yet applicable
    first = episode.step(state=domain.state)
    assert domain.setup.skill.name in first.label
    atoms.now = domain.target.preconditions
    second = episode.step(state=domain.state)  # no reset after the setup skill
    assert domain.target.skill.name in second.label
    with pytest.raises(HumanStartResetRequested):
        episode.step(state=domain.state)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_an_unreachable_target_executes_nothing_and_asks_for_nothing(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed plan to the precondition: EES falls back to one random applicable skill,
    which is not a target EES selected, so no reset follows it."""
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])

    def fail(self, **kwargs):  # noqa: PLR0917 - stands in for a bound method
        raise PlanningFailure("unreachable")

    monkeypatch.setattr(EesMethod, "plan_to", fail)
    episode = _episode(domain=domain)
    atoms.now = domain.setup.preconditions
    episode.step(state=domain.state)  # the random bootstrap, if any is applicable
    policy = domain.method.practice_target_reset
    assert policy is not None
    assert policy.target_executions == 0
    with contextlib.suppress(InteractionComplete):
        episode.step(state=domain.state)
    assert policy.requests == 0


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_reset_owed_when_the_session_ends_is_asked_first_thing_next_session(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every target execution is followed by one reset even when it was the session's
    last action, so the count of resets equals the count of target executions."""
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    atoms.now = domain.target.preconditions
    _episode(domain=domain).step(state=domain.state)  # the session ends here

    with pytest.raises(HumanStartResetRequested):
        _episode(domain=domain).step(state=domain.state)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_an_evaluation_episode_never_asks(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    atoms.now = domain.target.preconditions
    _episode(domain=domain).step(state=domain.state)  # owes one reset
    evaluation = _EesEpisode(method=domain.method, goal=frozenset(), practicing=False)
    evaluation.step(state=domain.state)  # a no-op without a plan, and no request


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_reset_count_equals_target_executions(
    *, name: str, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = DOMAINS[name](always_reset=True)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    monkeypatch.setattr(EesMethod, "plan_to", lambda self, **kwargs: [domain.setup])
    episode = _episode(domain=domain)
    resets = 0
    for step in range(30):
        # Every third step only the setup skill applies, so plans mix setup and target.
        atoms.now = domain.target.preconditions if step % 3 else domain.setup.preconditions
        try:
            episode.step(state=domain.state)
        except HumanStartResetRequested:
            resets += 1
            domain.method.observe_help_granted(state=domain.state)
    policy = domain.method.practice_target_reset
    assert policy is not None
    pending = int(policy.owed)
    assert resets + pending == policy.target_executions > 0
    assert policy.requests == resets


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_the_method_may_ask_even_without_the_domains_partial_resets(*, name: str) -> None:
    domain = DOMAINS[name](always_reset=True, offer_human_reset=False)
    assert domain.method.may_request_human_help()


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_the_price_defaults_to_the_domains_human_reset_price_and_is_configurable(
    *, name: str
) -> None:
    domain = DOMAINS[name](always_reset=True)
    policy = domain.method.practice_target_reset
    assert policy is not None
    assert policy.cost == domain.price
    assert AlwaysResetAfterPracticeTarget(cost=0.25).cost == 0.25


def test_without_a_domain_human_reset_the_price_must_be_given() -> None:
    env = SweepOffShelf3DEnvironment()
    provider = SweepOffShelf3DSkillProvider(env=env, offer_human_reset=False)
    with pytest.raises(ValueError, match="price"):
        AlwaysResetAfterPracticeTarget.for_provider(skill_provider=provider)


def test_under_step_accounting_the_reset_is_charged_by_its_own_name(
    *, atoms: _Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = _shelf(always_reset=True)
    accounting = PracticeAccounting(
        costs=PracticeCosts(
            human_skills={
                SAMPLED_START_RESET_SKILL_NAME: HumanCharge(cost=ChargeFunction(value=7.0))
            },
            human_weight=3.0,
        )
    )
    domain.method.configure_practice_accounting(accounting=accounting)
    _select(monkeypatch=monkeypatch, candidates=[domain.target])
    episode = _episode(domain=domain)
    atoms.now = domain.target.preconditions
    episode.step(state=domain.state)
    with pytest.raises(HumanStartResetRequested) as request:
        episode.step(state=domain.state)
    assert request.value.cost == 21.0


def test_the_start_reset_is_a_movables_reset_request_so_every_harness_counts_it() -> None:
    """The harnesses catch `HumanCubeBinResetRequested` and count, charge and record it;
    the start reset rides that path unchanged."""
    assert issubclass(HumanStartResetRequested, HumanCubeBinResetRequested)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_ees_skill_selection_is_unchanged(*, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real selector and the real planner, run side by side over identical
    observations: the always-reset method executes exactly the actions plain EES does,
    with the resets in between."""
    plain = DOMAINS[name](always_reset=False)
    always = DOMAINS[name](always_reset=True)
    # A fixed symbolic state in which several skills are reachable.
    fixed = plain.target.preconditions | plain.setup.preconditions
    monkeypatch.setattr(EesMethod, "abstract_state", lambda self, **kwargs: fixed)

    def labels(*, domain: _Domain) -> list[str]:
        episode = _episode(domain=domain)
        out: list[str] = []
        while len(out) < 8:
            try:
                out.append(episode.step(state=domain.state).label)
            except HumanStartResetRequested:
                domain.method.observe_help_granted(state=domain.state)
            except InteractionComplete:
                break
        return out

    expected = labels(domain=plain)
    assert expected
    assert labels(domain=always) == expected
    policy = always.method.practice_target_reset
    assert policy is not None
    assert policy.target_executions > 0
