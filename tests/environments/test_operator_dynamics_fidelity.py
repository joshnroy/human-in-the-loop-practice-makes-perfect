"""Cross-domain property test: a domain's symbolic operator model must never permit
more than its raw dynamics actually allow.

WHY this exists (two bugs of exactly this shape have shipped already):

1. **Ball-Ring, fixed in "Add ignore_effects to the symbolic layer" (#27).** `Skill`
   had no `ignore_effects`, so reachability was monotone: navigating to table B never
   revoked `IsReachableSurface(robot, A)`. Fast Downward happily returned
   `NavigateToTable(A) -> NavigateToTable(B) -> ...(A)`, a plan the environment cannot
   execute (`BallRingEnvironment._handle_placing` early-returns unless the robot is
   `euclidean_reachable` to the table). Evaluation success went 67% -> 98% once fixed.
2. **Tossing Room, still open.** `TossingRoomSkills.PICKUP`'s preconditions are
   `{RobotInRoom(robot, ?room), HandEmpty(robot)}` -- *any* room -- but
   `TossingRoomEnvironment._apply_pickup` only acts when `robot_room == start_room`.
   A pickup planned anywhere else is a silent no-op. (The walk below finds two more
   Tossing Room instances of the same shape; see `_TOSSINGROOM_DEFECTS`.)

A manual operator-by-operator audit *passed* on Ball-Ring before #27: a field-by-field
diff of preconditions/add/delete effects cannot see an entire effect class missing from
the representation. Only behaviour can. Hence this test.

WHY the walk is driven by a *believed* symbolic state rather than by re-abstracting the
real state every step: `SkillGrounder.applicable_ground_skills` reads only
`Skill.preconditions`, and `SkillGrounder.abstract_state` recomputes every atom from the
real `State` via `Predicate.holds`. `ignore_effects` touches neither. So a test that
re-abstracts reality at every step produces bit-identical output with and without
Ball-Ring's `ignore_effects` declarations -- it provably could not have caught bug #1.
Progressing the operator model forward symbolically (predicators' `utils.apply_operator`
ordering: ignore effects dropped first, then deletes, then adds) is what reproduces what
a *planner* believes, and therefore what exposes a model that has drifted above reality.

THE INVARIANT
    Take any symbolic state the planner could actually reach (start from a sampled
    task's initial state, or from a state the domain's own privileged oracle passes
    through; advance only via ground skills whose add effects the real environment
    genuinely achieved). Every ground skill applicable in that believed state must,
    when executed from the corresponding real state, *change the real state* -- unless
    all of its add effects already hold in reality, in which case it is symbolically
    vacuous and a no-op is the correct outcome.

    A silent no-op otherwise means the symbolic model claimed applicability the
    dynamics deny: the planner can emit it, and it will do nothing.

WHAT THIS DELIBERATELY DOES NOT ASSERT: that a skill *achieves* its add effects. That
is competence, which EES exists to learn, and it legitimately fails -- Light Switch's
`JumpToLight` never achieves anything, Ball-Ring's `PlaceBallOnTable` always drops the
ball (`place_ball_fall_prob=1.0`, and where it lands is stochastic), and a Tossing Room
throw with a badly sampled force misses. All three still *change* the world (except
`JumpToLight`, exempted below), so a change-based invariant separates "the sampler
missed" from "the model lied".
"""

from collections.abc import Callable

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.skill_provider import BallRingOracle, BallRingSkillProvider
from hitl_pmp.environments.ballring.tasks import BallRingTasks
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import (
    LightSwitchOracle,
    LightSwitchSkillProvider,
)
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.skill_provider import (
    TossingRoomOracle,
    TossingRoomSkillProvider,
)
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks
from hitl_pmp.planning.grounding import SkillGrounder

# Draws per execution trial for a skill with continuous parameters. A sampler miss is
# NOT a model defect, so a skill only counts as silently ignored when *every* draw
# leaves the world unchanged. 30 is sized by the worst per-draw hit rate of the three
# domains: Tossing Room's Throw force is U(0, 1) against a target drawn U(0.5, 1.0)
# with throw_tolerance=0.1, i.e. ~0.2 per draw, so a legitimate in-the-right-room throw
# reads as a violation with probability ~0.8**30 ~= 0.004. (Light Switch's toggle is
# ~0.5 per draw -- only a dial move clipped at the [0, 1] boundary is a no-op -- and
# Ball-Ring's place-on-table skills change `held` before any fall logic runs, so any
# draw changes the state.)
_PARAM_DRAWS = 30

# Walks seeded from freshly sampled train tasks. These are the ones that find bug #1's
# shape: only an undirected ramble navigates somewhere, navigates somewhere else, and
# then tries to act on the first place.
_NUM_RANDOM_WALKS = 8
_RANDOM_WALK_LENGTH = 40

# Walks seeded from every state the domain's own privileged oracle passes through while
# solving a task. Random walks alone do not reliably reach the deep states (Ball-Ring's
# "cup on a table with the ball in it" is eight specific steps from any initial state),
# and the coverage floor below is what would otherwise silently rot.
_ORACLE_HORIZON = 12
_ORACLE_WALK_LENGTH = 12


class Exemption(BaseModel):
    """A skill whose no-op is legitimate, *scoped to the situation that makes it so*.

    Deliberately not a bare skill-name allowlist: exempting `NavigateToBall` outright
    would also blind this test to a broken navigation on the floor, where it genuinely
    must work. `applies` therefore takes the pre-execution state and returns True only
    in the exempt situation; everywhere else the skill is checked normally.

    An exemption that silently absorbs a real defect would be worse than no test at
    all, so `reason` must name the source file that makes the no-op deliberate --
    enforced by `test_every_exemption_documents_why_it_is_legitimate`.

    `situational=False` marks the rare exemption that really is total (Light Switch's
    JumpToLight can never do anything, in any state). Everything else must stay
    situational, and `test_no_situational_exemption_is_silently_total` proves it by
    requiring that the walk did execute those skills for real somewhere -- otherwise a
    scoped exemption could quietly widen until it covered every state the walk visits,
    which is the blanket allowlist this design exists to avoid."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill_names: frozenset[str]
    reason: str
    # (*, case, ground_skill, state) -> bool. A plain `Callable` because the guards are
    # static methods on the containers below, not a project interface with a fixed
    # signature.
    applies: Callable[..., bool]
    situational: bool = True


class DomainCase(BaseModel):
    """One domain under test: the environment, its `SkillProvider`, the `Tasks` that
    seed random walks, the privileged oracle plus the tasks it should solve (seeding
    the deep walks), and the situations in which a no-op is legitimate."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    env: Environment
    provider: SkillProvider
    tasks: Tasks
    oracle: OraclePolicyProvider
    oracle_tasks: tuple[Task, ...]
    exemptions: tuple[Exemption, ...] = ()


class WalkReport(BaseModel):
    """What one domain's walks found: every silently-ignored ground skill, every skill
    name that was ever enumerated as a non-vacuous candidate (`exercised`), and the
    subset of those that were actually executed rather than exempted (`checked`).

    The two coverage sets are deliberately distinct. `exercised` says the walk reached
    the skill at all; `checked` says the invariant was really applied to it. A skill
    that is only ever *exempted* is `exercised` but not `checked` -- indistinguishable
    from a blanket allowlist entry, and the reason both are recorded."""

    violations: list[str]
    exercised: set[str]
    checked: set[str]


class Guards:
    """Predicates deciding whether an `Exemption` applies to a given situation. Static
    method containers, never instantiated, same as every other business-logic class in
    this project."""

    @staticmethod
    def always(*, case: DomainCase, ground_skill: GroundSkill, state: State) -> bool:
        """For a skill that is unconditionally incapable of affecting the world."""
        del case, ground_skill, state
        return True

    @staticmethod
    def ball_ring_navigation_target_is_not_on_the_floor(
        *, case: DomainCase, ground_skill: GroundSkill, state: State
    ) -> bool:
        """True when a Ball-Ring `NavigateToBall`/`NavigateToCup` names an object that
        is not lying on the floor -- i.e. it is resting on a table, or in the robot's
        own gripper. Both are situations with no reachable pose at all; see the
        exemption's `reason`."""
        del case
        return not BallRingEnvironment.on_floor(state=state, obj=ground_skill.objects[1])


_JUMP_TO_LIGHT_IS_A_DELIBERATE_TRAP = Exemption(
    skill_names=frozenset({"JumpToLight"}),
    reason=(
        "JumpToLight is Light Switch's deliberately impossible skill: its NSRT claims "
        "the robot lands two cells away, but compute_action returns a hardcoded zero "
        "action, so it can never change anything. Ported verbatim from predicators' "
        "own JumpToLight -- see the 'impossible skill' comment in "
        "src/hitl_pmp/environments/lightswitch/skills.py and "
        "test_compute_action_for_jump_to_light_is_always_a_no_op in "
        "tests/environments/lightswitch/test_skills.py. EES is supposed to DISCOVER "
        "this by practicing it and failing, so it must stay broken, not be 'fixed'."
    ),
    applies=Guards.always,
    situational=False,  # total by design: there is no state in which it does anything
)

_BALL_RING_NAVIGATION_TO_AN_OBJECT_OFF_THE_FLOOR = Exemption(
    skill_names=frozenset({"NavigateToBall", "NavigateToCup"}),
    reason=(
        "NavigateToBall/NavigateToCup declare no preconditions -- faithful to "
        "predicators' ground_truth_models/ball_and_cup_sticky_table/nsrts.py, where "
        "both are literally `preconditions = set()` -- so they stay enumerable when "
        "the ball/cup is NOT on the floor, and in exactly those situations no "
        "reachable pose exists at all. On a table: reachable_thresh (0.1) is smaller "
        "than a table's radius (~0.098), so every pose within reach of the object lies "
        "inside the table circle and BallRingEnvironment.exists_robot_collision "
        "rejects it. Held: the object's stored (x, y) is still the surface it was "
        "picked from, so the same collision rejects it. Either way the navigation is a "
        "no-op -- see src/hitl_pmp/environments/ballring/skills.py's navigate_action "
        "and the collision branch of BallRingEnvironment._simulate. The reference "
        "implementation is no better: its navigate_to_obj_sampler is a `while True` "
        "rejection loop that would spin forever in these states rather than return. "
        "And it is provably harmless to a planner: IsReachableBall/IsReachableCup are "
        "consumed ONLY by operators that also require BallOnFloor/CupOnFloor and "
        "HandEmpty (PickBallFromFloor, PickCupWithBallFromFloor, "
        "PickCupWithoutBallFromFloor, PlaceBallInCupOnFloor), i.e. exactly the states "
        "where navigation does work. Scoped to the off-the-floor situation ONLY: "
        "navigating to an object lying on the floor is still fully checked, so a real "
        "navigation regression still fails this test."
    ),
    applies=Guards.ball_ring_navigation_target_is_not_on_the_floor,
)

# The three Tossing Room defects this walk found while they were still live, kept as
# the record of what it is for. All are fixed on main (#28), so the domain is a plain
# parametrization below rather than an xfail:
#   (a) Pickup's preconditions were {RobotInRoom(robot, ?room), HandEmpty(robot)} for
#       ANY room, while _apply_pickup only acts when robot_room == start_room.
#   (b) Throw bound ?bin to any bin in the robot's room, while _apply_throw routes by
#       the HELD item's kind, so a mismatched bin could never succeed at any force.
#   (c) MoveRoom's only spatial precondition was symmetric Adjacent, while _apply_move
#       blocks the one-way ledge rightward.
# All three are the same defect class as Ball-Ring's missing ignore_effects: an
# over-permissive operator model yielding plans that look valid and cannot execute.
# Measured cost of (a)-(c) together: EES solved 1/10 Tossing Room tasks before the fix
# and 10/10 after.


class DomainCases:
    """Builds a fresh `DomainCase` per domain. Static-method container, never
    instantiated, same as every other business-logic class in this project. Fresh
    instances (not module-level singletons) because a walk mutates
    `Environment.current_state` and consumes `Tasks`' RNG streams."""

    @staticmethod
    def lightswitch() -> DomainCase:
        # grid_size=6, not the 100 default: the robot starts in cell 0 and the light
        # sits in the last cell, so on a 100-cell grid no bounded walk ever reaches the
        # light and TurnOnLight/TurnOffLight/JumpToLight are never enumerated at all.
        # The operator models under test do not depend on grid_size.
        env = LightSwitchEnvironment(grid_size=6)
        tasks = LightSwitchTasks(env=env, seed=0)
        return DomainCase(
            name="lightswitch",
            env=env,
            provider=LightSwitchSkillProvider(env=env),
            tasks=tasks,
            oracle=LightSwitchOracle(env=env),
            oracle_tasks=(tasks.sample_test_task(),),
            exemptions=(_JUMP_TO_LIGHT_IS_A_DELIBERATE_TRAP,),
        )

    @staticmethod
    def ballring() -> DomainCase:
        env = BallRingEnvironment()
        tasks = BallRingTasks(env=env, seed=0)
        return DomainCase(
            name="ballring",
            env=env,
            provider=BallRingSkillProvider(env=env),
            tasks=tasks,
            oracle=BallRingOracle(env=env),
            oracle_tasks=(tasks.sample_test_task(), tasks.sample_test_task()),
            # Ball-Ring's own "impossible" skill, PlaceBallOnTable, needs no exemption:
            # the bare ball always falls (place_ball_fall_prob=1.0), so it never
            # achieves BallOnTable -- but the ball still leaves the hand and lands on
            # the floor, which is a state change. That the landing point is stochastic
            # (_sample_floor_point_around_table) is exactly why this invariant is
            # change-based and never asserts that add effects hold.
            exemptions=(_BALL_RING_NAVIGATION_TO_AN_OBJECT_OFF_THE_FLOOR,),
        )

    @staticmethod
    def tossingroom() -> DomainCase:
        env = TossingRoomEnvironment()
        tasks = TossingRoomTasks(env=env, seed=0)
        # One oracle task per goal family: the EMPTY family is what walks the oracle
        # into the button room, and it is only ~20% of the default goal_weights.
        oracle_tasks = tuple(
            TossingRoomTasks(env=env, seed=0, forced_goal_type=goal_type).sample_test_task()
            for goal_type in TossingRoomGoalType
        )
        return DomainCase(
            name="tossingroom",
            env=env,
            provider=TossingRoomSkillProvider(env=env),
            tasks=tasks,
            oracle=TossingRoomOracle(env=env),
            oracle_tasks=oracle_tasks,
            # No exemptions ON PURPOSE. Everything this domain reports is a real defect
            # (see _TOSSINGROOM_DEFECTS); exempting any of it would be the failure mode
            # this whole file exists to prevent.
        )

    @staticmethod
    def build(*, domain: str) -> DomainCase:
        builders = {
            "lightswitch": DomainCases.lightswitch,
            "ballring": DomainCases.ballring,
            "tossingroom": DomainCases.tossingroom,
        }
        return builders[domain]()


class OperatorDynamicsFidelity:
    """The walk itself. Static-method container, never instantiated.

    Each step: enumerate every ground skill applicable in the *believed* symbolic
    state, execute each one speculatively from the real state (restoring afterwards),
    and record any that leaves the world untouched. Then advance both worlds together
    using one candidate that really did achieve its add effects -- so `believed` can
    only drift away from reality through genuine model unsoundness, never through a
    merely unlucky continuous-parameter draw."""

    @staticmethod
    def report(*, domain: str) -> WalkReport:
        case = DomainCases.build(domain=domain)
        violations: list[str] = []
        exercised: set[str] = set()
        checked: set[str] = set()
        # Global across walks: the walk always advances with the least-executed skill it
        # can, which turns a random ramble into a coverage-seeking one (Ball-Ring's
        # three NavigateTo* have empty preconditions and would otherwise swamp every
        # candidate set). Ties are broken by a seeded draw, not lexicographically --
        # a lexicographic tie-break makes a Light Switch walk ping-pong between two
        # cells forever.
        executions: dict[str, int] = {skill.name: 0 for skill in case.provider.skills()}

        for index, start in enumerate(OperatorDynamicsFidelity._starts(case=case)):
            case.env.set_state(state=start.state.model_copy(deep=True))
            believed = OperatorDynamicsFidelity._abstract(case=case)
            rng = np.random.default_rng(index)
            for _ in range(start.length):
                advanced = OperatorDynamicsFidelity._step(
                    case=case,
                    believed=believed,
                    rng=rng,
                    executions=executions,
                    violations=violations,
                    exercised=exercised,
                    checked=checked,
                )
                if advanced is None:
                    break
                believed = advanced

        return WalkReport(violations=violations, exercised=exercised, checked=checked)

    @staticmethod
    def _starts(*, case: DomainCase) -> list["WalkStart"]:
        """Where the walks begin: freshly sampled train-task initial states, plus every
        state the privileged oracle passes through on its way to solving each
        `oracle_task`."""
        starts = [
            WalkStart(
                state=case.tasks.sample_train_task().initial_state, length=_RANDOM_WALK_LENGTH
            )
            for _ in range(_NUM_RANDOM_WALKS)
        ]
        for task in case.oracle_tasks:
            case.env.set_state(state=task.initial_state.model_copy(deep=True))
            for _ in range(_ORACLE_HORIZON):
                state = case.env.get_current_state()
                starts.append(
                    WalkStart(state=state.model_copy(deep=True), length=_ORACLE_WALK_LENGTH)
                )
                if task.goal.is_satisfied(state=state):
                    break
                case.env.take_action(
                    action=case.oracle.get_labeled_action(state=state, goal=task.goal).action
                )
        return starts

    @staticmethod
    def _step(
        *,
        case: DomainCase,
        believed: frozenset[GroundAtom],
        rng: np.random.Generator,
        executions: dict[str, int],
        violations: list[str],
        exercised: set[str],
        checked: set[str],
    ) -> frozenset[GroundAtom] | None:
        """Check every believed-applicable ground skill, then advance. Returns the next
        believed symbolic state, or None when nothing can carry the walk forward."""
        snapshot = case.env.get_current_state().model_copy(deep=True)
        actual = OperatorDynamicsFidelity._abstract(case=case)
        candidates = sorted(
            SkillGrounder.applicable_ground_skills(
                skills=case.provider.skills(),
                objects=case.provider.objects(),
                true_atoms=believed,
            ),
            key=OperatorDynamicsFidelity.describe,
        )

        advanceable: list[tuple[GroundSkill, np.ndarray]] = []
        for ground_skill in candidates:
            if ground_skill.add_effects <= actual:
                continue  # symbolically vacuous here: a no-op is the correct outcome
            exercised.add(ground_skill.skill.name)
            if OperatorDynamicsFidelity._exempt(
                case=case, ground_skill=ground_skill, state=snapshot
            ):
                continue
            checked.add(ground_skill.skill.name)
            changed, achieving, failure = OperatorDynamicsFidelity._trial(
                case=case, ground_skill=ground_skill, snapshot=snapshot, rng=rng
            )
            if failure is not None:
                violations.append(failure)
            elif not changed:
                violations.append(
                    f"{OperatorDynamicsFidelity.describe(ground_skill)} is applicable "
                    f"(its preconditions hold) but executing it left the environment "
                    f"state completely unchanged -- the symbolic model claims an "
                    f"applicability the dynamics deny"
                )
            if achieving is not None:
                advanceable.append((ground_skill, achieving))

        case.env.set_state(state=snapshot.model_copy(deep=True))
        if not advanceable:
            return None
        ground_skill, params = min(
            zip(advanceable, rng.random(len(advanceable)), strict=True),
            key=lambda pair: (executions[pair[0][0].skill.name], float(pair[1])),
        )[0]
        case.env.take_action(
            action=case.provider.compute_action(
                ground_skill=ground_skill, params=params, state=case.env.get_current_state()
            )
        )
        executions[ground_skill.skill.name] += 1
        return OperatorDynamicsFidelity.apply(atoms=believed, ground_skill=ground_skill)

    @staticmethod
    def _trial(
        *,
        case: DomainCase,
        ground_skill: GroundSkill,
        snapshot: State,
        rng: np.random.Generator,
    ) -> tuple[bool, np.ndarray | None, str | None]:
        """Execute `ground_skill` from `snapshot` up to `_PARAM_DRAWS` times (once when
        it has no continuous parameters -- execution is then deterministic). Returns
        (did any draw change the state, the first draw that achieved the add effects or
        None, a violation message if the dynamics raised)."""
        draws = _PARAM_DRAWS if ground_skill.skill.param_dim > 0 else 1
        changed = False
        for _ in range(draws):
            params = case.provider.sample_params(ground_skill=ground_skill, rng=rng)
            case.env.set_state(state=snapshot.model_copy(deep=True))
            try:
                after = case.env.take_action(
                    action=case.provider.compute_action(
                        ground_skill=ground_skill,
                        params=params,
                        state=case.env.get_current_state(),
                    )
                )
            except AssertionError as exc:
                # Ball-Ring's _handle_placing carries a deliberate tripwire assert for
                # "an inapplicable skill was executed". Reaching it from a believed
                # state is the same defect class as a silent no-op -- the model let the
                # planner ask for something the dynamics cannot do -- so it is recorded,
                # never swallowed.
                message = (
                    f"{OperatorDynamicsFidelity.describe(ground_skill)} is applicable "
                    f"(its preconditions hold) but executing it raised from the "
                    f"dynamics: {exc!r}"
                )
                return changed, None, message
            changed = changed or OperatorDynamicsFidelity._changed(before=snapshot, after=after)
            if ground_skill.add_effects <= OperatorDynamicsFidelity._abstract(case=case):
                return True, params, None
        return changed, None, None

    @staticmethod
    def apply(*, atoms: frozenset[GroundAtom], ground_skill: GroundSkill) -> frozenset[GroundAtom]:
        """The symbolic successor, ported from predicators' `utils.apply_operator`.
        Ignore effects are dropped FIRST, so a predicate that is both ignored and added
        (every Ball-Ring NavigateTo*) ends up true, not false."""
        survivors = {atom for atom in atoms if atom.predicate not in ground_skill.ignore_effects}
        return frozenset((survivors - ground_skill.delete_effects) | ground_skill.add_effects)

    @staticmethod
    def describe(ground_skill: GroundSkill) -> str:  # noqa: PLR0917 (sort key: positional)
        objects = ", ".join(obj.name for obj in ground_skill.objects)
        return f"{ground_skill.skill.name}({objects})"

    @staticmethod
    def _exempt(*, case: DomainCase, ground_skill: GroundSkill, state: State) -> bool:
        return any(
            ground_skill.skill.name in exemption.skill_names
            and exemption.applies(case=case, ground_skill=ground_skill, state=state)
            for exemption in case.exemptions
        )

    @staticmethod
    def _abstract(*, case: DomainCase) -> frozenset[GroundAtom]:
        return SkillGrounder.abstract_state(
            state=case.env.get_current_state(),
            objects=case.provider.objects(),
            predicates=case.provider.predicates(),
        )

    @staticmethod
    def _changed(*, before: State, after: State) -> bool:
        if set(before.data) != set(after.data):
            return True
        return any(not np.array_equal(before.data[obj], after.data[obj]) for obj in before.data)


class WalkStart(BaseModel):
    """One walk's starting point and how many steps it gets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: State
    length: int


_REPORTS: dict[str, WalkReport] = {}


def _report(*, domain: str) -> WalkReport:
    """One set of walks per domain, shared by the tests below (the walks are the
    expensive part and are fully deterministic, so recomputing them per test would only
    cost time)."""
    if domain not in _REPORTS:
        _REPORTS[domain] = OperatorDynamicsFidelity.report(domain=domain)
    return _REPORTS[domain]


_DOMAINS = ["lightswitch", "ballring", "tossingroom"]


@pytest.mark.parametrize(
    "domain",
    [
        "lightswitch",
        "ballring",
        "tossingroom",
    ],
)
def test_applicable_ground_skills_are_never_silently_ignored(*, domain: str) -> None:
    """THE property. See this module's docstring for the two shipped bugs it guards."""
    report = _report(domain=domain)
    assert not report.violations, (
        f"{domain}: {len(report.violations)} ground-skill execution(s) were applicable "
        f"in a symbolic state a planner can reach, yet the dynamics silently ignored "
        f"them:\n" + "\n".join(f"  - {violation}" for violation in sorted(set(report.violations)))
    )


@pytest.mark.parametrize("domain", _DOMAINS)
def test_the_walk_exercises_every_declared_skill(*, domain: str) -> None:
    """A coverage floor, so the property test above cannot pass vacuously. Without it,
    shortening the walk (or a change that strands it in a corner of the state space)
    would quietly reduce the test to "nothing was enumerated, nothing was violated".
    Deliberately NOT xfailed for tossingroom: coverage must stay enforced there even
    while the defects above are outstanding."""
    report = _report(domain=domain)
    declared = {skill.name for skill in DomainCases.build(domain=domain).provider.skills()}
    assert declared <= report.exercised, (
        f"{domain}: the walk never reached a state where "
        f"{sorted(declared - report.exercised)} was applicable, so this file proves "
        f"nothing about those skills"
    )


@pytest.mark.parametrize("domain", _DOMAINS)
def test_no_situational_exemption_is_silently_total(*, domain: str) -> None:
    """The second half of the coverage floor. `exercised` only records that the walk
    *enumerated* a skill; a skill whose every enumeration was exempted is enumerated but
    never actually tested, which is operationally identical to a blanket allowlist entry
    -- the exact failure mode the situation-scoped `Exemption` design exists to prevent.
    So every skill named by a `situational=True` exemption must also have been executed
    for real somewhere (Ball-Ring's navigations to an object lying on the floor). Only an
    exemption that declares itself total (`situational=False`, i.e. JumpToLight) is
    allowed to have been exempted every single time."""
    report = _report(domain=domain)
    for exemption in DomainCases.build(domain=domain).exemptions:
        if not exemption.situational:
            continue
        never_checked = exemption.skill_names - report.checked
        assert not never_checked, (
            f"{domain}: {sorted(never_checked)} was exempted every time it came up, so "
            f"its situation-scoped exemption is behaving as a blanket allowlist. Either "
            f"the walk no longer reaches the situations where the skill must work, or "
            f"the guard has widened -- neither is safe to leave in place"
        )


@pytest.mark.parametrize("domain", _DOMAINS)
def test_every_exemption_documents_why_it_is_legitimate(*, domain: str) -> None:
    """An exemption that silently absorbs a real bug is worse than no test. Each one
    must name the source file that makes its no-op deliberate, and must cover skills the
    domain actually declares (so a renamed skill drops its exemption loudly rather than
    leaving a dead entry behind)."""
    case = DomainCases.build(domain=domain)
    declared = {skill.name for skill in case.provider.skills()}
    for exemption in case.exemptions:
        assert "src/hitl_pmp/environments/" in exemption.reason, (
            f"{domain}: exempted {sorted(exemption.skill_names)} without pointing at "
            f"the code that makes the no-op deliberate"
        )
        assert exemption.skill_names <= declared, (
            f"{domain}: exemption names {sorted(exemption.skill_names - declared)}, "
            f"which this domain does not declare"
        )
