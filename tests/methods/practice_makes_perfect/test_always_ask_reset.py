"""The always-ask reset wrapper: a human full reset to a freshly sampled start state,
forced after every skill with continuous parameters and whenever the goal is
unreachable without a human.

Symbolic: the abstraction, the skill execution and (where named) the planner are
stubbed, so each step sees exactly the atoms a test names. Both domains, and all three
planners (EES, expectimax, determinized A*) -- the wrapper lives in the shared episode.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.method import (
    HumanStartResetRequested,
    InteractionComplete,
)
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)
from hitl_pmp.core.problem.human.types import SAMPLED_START_RESET_DESTINATION
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    SweepOffShelf3DAtoms,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_BIN
from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import (
    PARAMETERIZED_SKILL,
    STOP_ONLY_SESSION,
    STRANDED,
    AlwaysAskReset,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode
from hitl_pmp.planning.fast_downward import PlanningFailure

PRICE = 3.0


class _Domain(BaseModel):
    """A method, the task goal, a skill with continuous parameters, one without, atoms
    under which the domain's own partial reset applies, and a state object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: EesMethod
    goal: frozenset[GroundAtom]
    parameterized: GroundSkill
    plain: GroundSkill
    partial_reset_applies: frozenset[GroundAtom]
    state: Any


def _rule_or_none(*, wrapped: bool) -> AlwaysAskReset | None:
    return AlwaysAskReset(cost=PRICE) if wrapped else None


def _shelf(*, wrapped: bool = True) -> _Domain:
    env = SweepOffShelf3DEnvironment()
    provider = SweepOffShelf3DSkillProvider(env=env)
    method = EesMethod(
        env=env, skill_provider=provider, always_ask_reset=_rule_or_none(wrapped=wrapped)
    )
    return _Domain(
        method=method,
        goal=SweepOffShelf3DPredicates.goal_atoms(),
        parameterized=GroundSkill(
            skill=SweepOffShelf3DSkills.SWEEP_SHELF, objects=(O.robot, O.wiper, O.box, *O.cubes)
        ),
        plain=GroundSkill(
            skill=SweepOffShelf3DSkills.PICK_CUBE_FROM_SHELF, objects=(O.robot, O.cubes[0])
        ),
        partial_reset_applies=provider.human_cube_bin_reset_skills()[0].preconditions,
        state=SweepOffShelf3DState(
            data={obj: np.zeros(obj.type.dim) for obj in (*O.all_objects(), O.scene)},
            abstract_atoms=frozenset(),
        ),
    )


def _tossing_env_and_provider() -> tuple[Tossing3DEnvironment, Tossing3DSkillProvider]:
    env = Tossing3DEnvironment(scene_bg=False)
    return env, Tossing3DSkillProvider(env=env)


def _tossing_domain(*, method: EesMethod, env: Tossing3DEnvironment) -> _Domain:
    return _Domain(
        method=method,
        goal=frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))}),
        parameterized=GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
        ),
        plain=GroundSkill(
            skill=Tossing3DSkills.PICK_CUBE,
            objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot, env.bin),
        ),
        partial_reset_applies=method.skill_provider.human_cube_bin_reset_skills()[0].preconditions,
        state=Tossing3DState(
            data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
            abstract_atoms=frozenset(),
        ),
    )


def _tossing(*, wrapped: bool = True) -> _Domain:
    env, provider = _tossing_env_and_provider()
    method = EesMethod(
        env=env, skill_provider=provider, always_ask_reset=_rule_or_none(wrapped=wrapped)
    )
    return _tossing_domain(method=method, env=env)


def _pomdp(*, solver: str) -> Callable[..., _Domain]:
    def build(*, wrapped: bool = True) -> _Domain:
        env, provider = _tossing_env_and_provider()
        method = Tossing3DPomdpMethod(
            env=env,
            skill_provider=provider,
            pomdp_solver=solver,
            pomdp_num_particles=16,
            always_ask_reset=_rule_or_none(wrapped=wrapped),
        )
        return _tossing_domain(method=method, env=env)

    return build


DOMAINS: dict[str, Callable[..., _Domain]] = {
    "shelf-ees": _shelf,
    "tossing3d-ees": _tossing,
    "tossing3d-expectimax": _pomdp(solver="expectimax"),
    "tossing3d-astar": _pomdp(solver="determinized_astar"),
}


class _World:
    """What the stubbed abstraction returns next, whether the goal is reachable, and
    what the selector proposes."""

    def __init__(self) -> None:
        self.atoms: frozenset[GroundAtom] = frozenset()
        self.reachable = True
        self.candidates: list[GroundSkill] = []


@pytest.fixture
def world(*, monkeypatch: pytest.MonkeyPatch) -> _World:
    current = _World()
    for cls in (EesMethod, Tossing3DPomdpMethod):
        monkeypatch.setattr(cls, "abstract_state", lambda self, **kwargs: current.atoms)
        monkeypatch.setattr(
            cls, "select_skill_to_practice", lambda self, **kwargs: list(current.candidates)
        )

    def plan_to(self, **kwargs):  # noqa: PLR0917 - stands in for a bound method
        if not current.reachable:
            raise PlanningFailure("stranded")
        return []

    def execute(self, *, ground_skill: GroundSkill, state, explore: bool):  # noqa: PLR0917
        del state, explore
        return LabeledAction(action=np.zeros(1), label=ground_skill.skill.name), None

    monkeypatch.setattr(EesMethod, "plan_to", plan_to)
    monkeypatch.setattr(EesMethod, "execute_ground_skill", execute)
    return current


def _episode(*, domain: _Domain, practicing: bool = True) -> _EesEpisode:
    return _EesEpisode(
        method=domain.method, goal=domain.goal, practicing=practicing, pursue_goal=False
    )


def _rule(*, domain: _Domain) -> AlwaysAskReset:
    rule = domain.method.always_ask_reset
    assert rule is not None
    return rule


@pytest.mark.parametrize("name", sorted(DOMAINS))
@pytest.mark.parametrize("succeeded", [True, False], ids=["success", "failure"])
def test_a_parameterized_skill_forces_one_full_reset_next_step(
    *, name: str, succeeded: bool, world: _World
) -> None:
    domain = DOMAINS[name]()
    world.candidates = [domain.parameterized]
    episode = _episode(domain=domain)
    world.atoms = domain.parameterized.preconditions
    assert episode.step(state=domain.state).label == domain.parameterized.skill.name

    if succeeded:
        world.atoms = world.atoms | domain.parameterized.add_effects
    with pytest.raises(HumanStartResetRequested) as request:
        episode.step(state=domain.state)
    assert request.value.destination == SAMPLED_START_RESET_DESTINATION
    assert request.value.cost == PRICE
    assert _rule(domain=domain).forced == {PARAMETERIZED_SKILL: 1}
    assert _rule(domain=domain).skill_attempts == 1
    # Granted; the planner acts again and nothing more is owed.
    domain.method.observe_help_granted(state=domain.state)
    world.atoms = domain.parameterized.preconditions
    assert episode.step(state=domain.state).label == domain.parameterized.skill.name


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_skill_without_continuous_parameters_forces_nothing(*, name: str, world: _World) -> None:
    domain = DOMAINS[name]()
    world.candidates = [domain.plain]
    episode = _episode(domain=domain)
    world.atoms = domain.plain.preconditions
    episode.step(state=domain.state)
    assert episode.step(state=domain.state).label == domain.plain.skill.name
    assert _rule(domain=domain).forced == {}


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_stranded_robot_is_reset(*, name: str, world: _World) -> None:
    domain = DOMAINS[name]()
    world.candidates = [domain.plain]
    world.reachable = False
    world.atoms = domain.plain.preconditions
    with pytest.raises(HumanStartResetRequested):
        _episode(domain=domain).step(state=domain.state)
    assert _rule(domain=domain).forced == {STRANDED: 1}


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_reset_that_does_not_cure_stranding_fires_again_and_is_logged(
    *, name: str, world: _World
) -> None:
    """No cap: each stranded step asks again, and the run length is recorded."""
    domain = DOMAINS[name]()
    world.candidates = [domain.plain]
    world.reachable = False
    world.atoms = domain.plain.preconditions
    episode = _episode(domain=domain)
    for _ in range(3):
        with pytest.raises(HumanStartResetRequested):
            episode.step(state=domain.state)
        domain.method.observe_help_granted(state=domain.state)
    rule = _rule(domain=domain)
    assert rule.forced == {STRANDED: 3}
    assert rule.longest_stranded_run == 3
    assert [event["consecutive_stranded"] for event in rule.events] == [1, 2, 3]
    world.reachable = True
    assert episode.step(state=domain.state).label == domain.plain.skill.name


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_reset_owed_when_the_session_ends_is_forced_first_thing_next_session(
    *, name: str, world: _World
) -> None:
    domain = DOMAINS[name]()
    world.candidates = [domain.parameterized]
    world.atoms = domain.parameterized.preconditions
    _episode(domain=domain).step(state=domain.state)
    with pytest.raises(HumanStartResetRequested):
        _episode(domain=domain).step(state=domain.state)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_evaluation_is_never_reset(*, name: str, world: _World) -> None:
    domain = DOMAINS[name]()
    world.reachable = False
    _episode(domain=domain, practicing=False).step(state=domain.state)
    assert _rule(domain=domain).forced == {}


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_without_the_wrapper_nothing_is_forced(*, name: str, world: _World) -> None:
    domain = DOMAINS[name](wrapped=False)
    world.candidates = [domain.parameterized]
    world.reachable = False
    world.atoms = domain.parameterized.preconditions
    episode = _episode(domain=domain)
    episode.step(state=domain.state)
    assert episode.step(state=domain.state).label == domain.parameterized.skill.name


# --- the wrapped arms offer the planner no partial human reset (Josh: "Off.") ---------


def _offered_to_ees_practice_planner(
    *, method: EesMethod, monkeypatch: pytest.MonkeyPatch
) -> set[str]:
    """The skills `plan_to` hands Fast Downward for a practice plan."""
    offered: set[str] = set()

    def capture(**kwargs):
        offered.update(skill.name for skill in kwargs["skills"])
        return []

    monkeypatch.setattr(method, "_plan_or_raise", capture)
    EesMethod.plan_to(method, init_atoms=frozenset(), goal=frozenset(), costs={}, practicing=True)
    return offered


@pytest.mark.parametrize("build", [_shelf, _tossing], ids=["shelf", "tossing3d"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_only_unwrapped_ees_offers_its_planner_the_partial_resets(
    *, build, wrapped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = build(wrapped=wrapped)
    partial = {
        reset.skill.name for reset in domain.method.skill_provider.human_cube_bin_reset_skills()
    }
    assert partial
    offered = _offered_to_ees_practice_planner(method=domain.method, monkeypatch=monkeypatch)
    assert bool(offered & partial) is not wrapped
    # Removed, not priced out: the robot's own skills are all still there.
    assert {skill.name for skill in domain.method.skills()} <= offered


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_only_unwrapped_belief_space_planners_can_choose_a_partial_reset(
    *, solver: str, wrapped: bool
) -> None:
    domain = _pomdp(solver=solver)(wrapped=wrapped)
    method = domain.method
    assert isinstance(method, Tossing3DPomdpMethod)
    partial = {reset.skill.name for reset in method.skill_provider.movables_reset_skills()}
    actions = {ground.skill.name for ground in method._pomdp_model.ground_skills}  # noqa: SLF001
    assert bool(actions & partial) is not wrapped
    assert bool(method.human_skills()) is not wrapped
    assert {skill.name for skill in method.skills()} <= actions


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_the_stranded_check_offers_no_reset_either_way(
    *, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Robot skills only, wrapped or not -- as before this change."""
    for wrapped in (True, False):
        domain = DOMAINS[name](wrapped=wrapped)
        offered: list[set[str]] = []
        monkeypatch.setattr(
            domain.method,
            "_plan_or_raise",
            lambda offered=offered, **kwargs: (
                offered.append({skill.name for skill in kwargs["skills"]}) or []
            ),
        )
        domain.method.goal_unreachable(true_atoms=frozenset(), goal=domain.goal)
        assert offered == [{skill.name for skill in domain.method.skills()}]


def test_the_pomdp_planner_records_a_forced_reset_and_its_completion(
    *, world: _World, tmp_path
) -> None:
    """Charged to the session and logged apart from the planner's own decisions; its
    completion is consumed without a belief update."""
    env, provider = _tossing_env_and_provider()
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        pomdp_num_particles=16,
        always_ask_reset=AlwaysAskReset(cost=PRICE),
        decision_log=tmp_path / "decisions.jsonl",
    )
    domain = _tossing_domain(method=method, env=env)
    world.candidates = [domain.plain]
    world.reachable = False
    world.atoms = domain.plain.preconditions
    with pytest.raises(HumanStartResetRequested):
        _episode(domain=domain).step(state=domain.state)
    assert method.pomdp_state.accumulated_cost == PRICE
    method.observe_help_granted(state=domain.state)
    events = (tmp_path / "decisions.jsonl").read_text()
    assert '"forced_reset"' in events
    assert '"forced_reset_complete"' in events


def test_under_step_accounting_the_reset_is_charged_by_its_own_name(*, world: _World) -> None:
    domain = _shelf()
    domain.method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={
                    "reset_to_sampled_start": HumanCharge(cost=ChargeFunction(value=7.0))
                },
                human_weight=3.0,
            )
        )
    )
    world.candidates = [domain.plain]
    world.reachable = False
    world.atoms = domain.plain.preconditions
    with pytest.raises(HumanStartResetRequested) as request:
        _episode(domain=domain).step(state=domain.state)
    assert request.value.cost == 21.0


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_stop_only_session_owes_one_forced_reset(*, name: str, world: _World) -> None:
    """The third trigger, read by the step protocol at the end of a session that was
    only a STOP: priced like the others and counted under its own reason."""
    del world
    domain = DOMAINS[name]()
    assert domain.method.stop_only_session_reset() == PRICE
    assert _rule(domain=domain).forced == {STOP_ONLY_SESSION: 1}
    assert DOMAINS[name](wrapped=False).method.stop_only_session_reset() is None


def test_the_pomdp_planner_takes_a_stop_only_reset_before_any_dispatch(
    *, world: _World, tmp_path
) -> None:
    """A run whose first session is only a STOP has dispatched nothing yet; its forced
    reset is still charged, completed and logged without a belief update."""
    del world
    env, provider = _tossing_env_and_provider()
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        pomdp_num_particles=16,
        always_ask_reset=AlwaysAskReset(cost=PRICE),
        decision_log=tmp_path / "decisions.jsonl",
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={"reset_to_sampled_start": HumanCharge(cost=ChargeFunction(value=1))}
            )
        )
    )
    domain = _tossing_domain(method=method, env=env)
    method.stop_only_session_reset()
    method.observe_execution_cost(cost=1.0, steps=1, complete=True)
    method.observe_help_granted(state=domain.state)
    events = (tmp_path / "decisions.jsonl").read_text()
    assert f'"reason": "{STOP_ONLY_SESSION}"' in events
    assert '"forced_reset_complete"' in events
    assert method._execution_durations == {}  # noqa: SLF001


# --- the trigger is read off the skill definitions ------------------------------------


def _any_grounding(*, skill: Skill) -> GroundSkill:
    _, provider = _tossing_env_and_provider()
    shelf = SweepOffShelf3DSkillProvider(env=SweepOffShelf3DEnvironment())
    objects = (*provider.objects(), *shelf.objects())
    return GroundSkill(
        skill=skill,
        objects=tuple(
            next(obj for obj in objects if obj.type == parameter.type)
            for parameter in skill.parameters
        ),
    )


def test_only_the_toss_and_the_sweep_have_continuous_parameters() -> None:
    _, tossing = _tossing_env_and_provider()
    shelf = SweepOffShelf3DSkillProvider(env=SweepOffShelf3DEnvironment())
    triggering = {
        skill.name
        for provider in (tossing, shelf)
        for skill in provider.skills()
        if AlwaysAskReset.triggers(ground_skill=_any_grounding(skill=skill))
    }
    assert triggering == {"MoveToTossLocationAndToss", "SweepShelf"}


@pytest.mark.parametrize(
    ("environment", "skills"),
    [
        ("sweepoffshelf3d", lambda: SweepOffShelf3DSkillProvider(env=SweepOffShelf3DEnvironment())),
        ("tossing3d", lambda: _tossing_env_and_provider()[1]),
        ("tossing3d same-side", lambda: SameSideSkills),
    ],
)
def test_each_always_ask_environment_has_exactly_one_parameterized_skill(
    *, environment: str, skills
) -> None:
    """The reset trigger is "after the environment's parameterized skill" (Josh,
    2026-10-21), which presumes there is one. A second one would silently add a trigger.
    These are the domains that implement the full reset always-ask requests."""
    triggering = [
        skill.name
        for skill in skills().skills()
        if AlwaysAskReset.triggers(ground_skill=_any_grounding(skill=skill))
    ]
    assert len(triggering) == 1, (environment, triggering)


def test_the_trigger_follows_the_definition_not_the_name() -> None:
    """Give the pick a continuous parameter and it triggers; take the toss's away and it
    does not."""
    pick = Tossing3DSkills.PICK_CUBE.model_copy(update={"param_dim": 1})
    toss = Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS.model_copy(update={"param_dim": 0})
    assert AlwaysAskReset.triggers(ground_skill=_any_grounding(skill=pick))
    assert not AlwaysAskReset.triggers(ground_skill=_any_grounding(skill=toss))


def test_the_price_defaults_to_the_domains_human_reset_price() -> None:
    _, provider = _tossing_env_and_provider()
    assert AlwaysAskReset.for_provider(skill_provider=provider).cost == (
        provider.human_reset_practice_cost
    )
    shelf = SweepOffShelf3DSkillProvider(
        env=SweepOffShelf3DEnvironment(), human_reset_practice_cost=2.0
    )
    assert AlwaysAskReset.for_provider(skill_provider=shelf).cost == 2.0
    assert AlwaysAskReset.for_provider(skill_provider=shelf, cost=0.5).cost == 0.5


def test_without_a_domain_human_reset_the_price_must_be_given() -> None:
    shelf = SweepOffShelf3DSkillProvider(env=SweepOffShelf3DEnvironment(), offer_human_reset=False)
    with pytest.raises(ValueError, match="price"):
        AlwaysAskReset.for_provider(skill_provider=shelf)


# --- the planner's own choices are unchanged ------------------------------------------


def _symbolic_run(
    *, domain: _Domain, start: frozenset[GroundAtom], steps: int, monkeypatch
) -> list[str]:
    """Drive one practice episode through a symbolic world in which every skill
    succeeds, and a forced reset leaves the world as it was -- so a wrapped and an
    unwrapped run see identical observations, and any difference in what they do is the
    wrapper's."""
    world = {"atoms": start}
    monkeypatch.setattr(type(domain.method), "abstract_state", lambda self, **kw: world["atoms"])
    episode = _episode(domain=domain)
    out: list[str] = []
    while len([label for label in out if label != "RESET"]) < steps:
        try:
            out.append(episode.step(state=domain.state).label)
        except HumanStartResetRequested:
            out.append("RESET")
            domain.method.observe_help_granted(state=domain.state)
            continue
        except InteractionComplete:
            break
        done = episode._pending  # noqa: SLF001
        assert done is not None
        world["atoms"] = (world["atoms"] - done.delete_effects) | done.expected_add_effects(
            before_atoms=world["atoms"]
        )
    return out


def _shelf_start() -> frozenset[GroundAtom]:
    named = {("HandEmpty", ("robot",)), ("WiperHome", (O.wiper.name,))}
    named |= {("OnShelf", (cube.name,)) for cube in O.cubes}
    state = SweepOffShelf3DState(
        data={}, abstract_atoms=SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(named))
    )
    return _shelf().method.abstract_state(state=state)


def test_real_ees_selection_and_planning_are_unchanged_by_the_wrapper(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real selector, sampler and Fast Downward: wrapped and unwrapped EES take the
    same actions, so the wrapper's per-step stranded check perturbs nothing."""
    start = _shelf_start()
    plain = _symbolic_run(
        domain=_shelf(wrapped=False), start=start, steps=12, monkeypatch=monkeypatch
    )
    wrapped = _symbolic_run(
        domain=_shelf(wrapped=True), start=start, steps=12, monkeypatch=monkeypatch
    )
    assert plain
    assert [label for label in wrapped if label != "RESET"] == plain


def test_forced_resets_leave_the_planners_choices_and_draws_unchanged(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a selector that draws from the method's own RNG among the applicable skills
    (so the sweep is chosen often) and the real sampler, the wrapped run takes exactly
    the unwrapped run's actions, parameters included, with resets in between."""
    from hitl_pmp.planning.grounding import SkillGrounder

    def select(self, *, true_atoms):  # noqa: PLR0917 - stands in for a bound method
        applicable = SkillGrounder.applicable_ground_skills(
            skills=self.skills(), objects=self.objects(), true_atoms=true_atoms
        )
        return [self.random_choice(ground_skills=list(applicable))]

    monkeypatch.setattr(EesMethod, "select_skill_to_practice", select)
    start = _shelf_start()
    plain = _symbolic_run(
        domain=_shelf(wrapped=False), start=start, steps=25, monkeypatch=monkeypatch
    )
    wrapped_domain = _shelf(wrapped=True)
    wrapped = _symbolic_run(domain=wrapped_domain, start=start, steps=25, monkeypatch=monkeypatch)
    assert [label for label in wrapped if label != "RESET"] == plain
    sweeps = sum(label.startswith("SweepShelf") for label in plain)
    assert sweeps > 0
    assert wrapped.count("RESET") == _rule(domain=wrapped_domain).forced[PARAMETERIZED_SKILL]
    assert _rule(domain=wrapped_domain).skill_attempts == sweeps


def test_a_forced_resets_measured_duration_is_not_credited_to_the_last_robot_skill(
    *, world: _World
) -> None:
    """Under step accounting the harness reports the human's steps via
    `observe_execution_cost`; they belong to no robot skill's duration forecast."""
    domain = _pomdp(solver="expectimax")()
    method = domain.method
    assert isinstance(method, Tossing3DPomdpMethod)
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={"reset_to_sampled_start": HumanCharge(cost=ChargeFunction(value=1))}
            )
        )
    )
    world.candidates = [domain.parameterized]
    world.atoms = domain.parameterized.preconditions
    episode = _episode(domain=domain)
    episode.step(state=domain.state)
    method.observe_execution_cost(cost=1.0, steps=40, complete=True)
    with pytest.raises(HumanStartResetRequested):
        episode.step(state=domain.state)
    method.observe_execution_cost(cost=1.0, steps=999, complete=True)
    method.observe_help_granted(state=domain.state)
    assert method._execution_durations[domain.parameterized] == [40]  # noqa: SLF001


# --- the belief-space practice domain under the wrapper --------------------------------


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_a_shelf_belief_space_planner_builds_with_and_without_the_wrapper(
    *, solver: str, wrapped: bool
) -> None:
    """Under the wrapper the shelf's human skills are not offered, so the practice
    domain drops their beliefs; unwrapped, they stay and the completeness check still
    holds."""
    from hitl_pmp.methods.belief_space.practice_domain import SWEEP_OFF_SHELF_DOMAIN

    env = SweepOffShelf3DEnvironment()
    provider = SweepOffShelf3DSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        pomdp_solver=solver,
        pomdp_num_particles=16,
        pomdp_domain=SWEEP_OFF_SHELF_DOMAIN,
        always_ask_reset=_rule_or_none(wrapped=wrapped),
    )
    human = {reset.skill.name for reset in provider.movables_reset_skills()}
    beliefs = {name for name, _ in method.pomdp_domain.belief_sources}
    actions = {ground.skill.name for ground in method._pomdp_model.ground_skills}  # noqa: SLF001
    assert bool(beliefs & human) is not wrapped
    assert bool(actions & human) is not wrapped
    assert {skill.name for skill in method.skills()} <= actions
    # The shared module-level domain is never modified.
    assert human <= {name for name, _ in SWEEP_OFF_SHELF_DOMAIN.belief_sources}


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_the_tossing3d_belief_space_domain_is_unaffected_by_the_wrapper(
    *, solver: str, wrapped: bool
) -> None:
    """Tossing3D's beliefs are the robot's alone (its reset is a deterministic skill,
    not a belief), so the wrapper leaves its practice domain exactly as it was."""
    from hitl_pmp.methods.belief_space.practice_domain import TOSSING3D_DOMAIN

    domain = _pomdp(solver=solver)(wrapped=wrapped)
    assert domain.method.pomdp_domain == TOSSING3D_DOMAIN
