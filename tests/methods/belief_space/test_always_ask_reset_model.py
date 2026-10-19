"""The always-ask belief-space planners know the forced reset is coming (Josh,
2026-10-19): in their model a skill with continuous parameters is followed by the
human's full reset to the start state, priced like the real one. The unwrapped arms'
model is untouched.

Every sampled start scene of Tossing3D abstracts to one symbolic state (`START`), so
the reset's destination is a single state, not a distribution."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_BIN
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import mean_competence
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    apply_success_effects,
    make_tossing3d_search_state,
    transition_outcomes,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import AlwaysAskReset
from hitl_pmp.planning.grounding import SkillGrounder

PRICE = 5.0

# What every sampled start scene abstracts to (measured on 30 reset-start draws at
# seeds 0-2, their train tasks and the canonical scene 125: one state).
START = frozenset({
    "BinAtSide(bin_0, cuboid_barrier, opposite_side)",
    "BinOnGround(bin_0)",
    "CubeAtSide(cube_0, cuboid_barrier, robot_side)",
    "GraspClear(cube_0, bin_0)",
    "HandEmpty(robot)",
    "NotHolding(robot, cube_0)",
    "OnGround(cube_0)",
    "PickPlannable(robot, cube_0)",
    "PickupUnblocked(cube_0)",
    "RobotAtSide(robot, cuboid_barrier, robot_side)",
})

# EXP-22c's belief-space settings, as the always-ask pilot ran them.
EXP22C: dict[str, object] = {
    "pomdp_competence_model": "local_trend",
    "pomdp_inference_engine": "grid",
    "pomdp_competence_process_noise_std": 0.03,
    "pomdp_learning_rate_decay": 0.9,
    "pomdp_learning_rate_max": 0.15,
    "pomdp_learning_rate_process_noise_std": 0.005,
    "pomdp_linear_cost_lambda": 3e-6,
    "pomdp_max_search_iterations": 100,
    "pomdp_num_samples": 100,
    "pomdp_observation_probability_weight": 0.0,
    "pomdp_search_depth": 6,
    "exploration_epsilon": 0.5,
}


def _render(*, atom: GroundAtom) -> str:
    return f"{atom.predicate.name}({', '.join(o.name for o in atom.objects)})"


def _method(
    *, solver: str, wrapped: bool, price: float = PRICE, seed: int = 0
) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=seed,
        pomdp_solver=solver,
        always_ask_reset=AlwaysAskReset(cost=price) if wrapped else None,
        **EXP22C,  # type: ignore[arg-type]
    )


def _atoms(*, method: Tossing3DPomdpMethod, names: frozenset[str]) -> frozenset[GroundAtom]:
    found = frozenset(
        atom
        for atom in SkillGrounder.all_possible_ground_atoms(
            objects=method.objects(), predicates=method.predicates()
        )
        if _render(atom=atom) in names
    )
    assert {_render(atom=atom) for atom in found} == names
    return found


def _ground(
    *, method: Tossing3DPomdpMethod, name: str, atoms: frozenset[GroundAtom]
) -> GroundSkill:
    return next(
        skill
        for skill in method._pomdp_model.ground_skills  # noqa: SLF001
        if skill.skill.name == name and skill.preconditions <= atoms
    )


def _start_session(*, method: Tossing3DPomdpMethod, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a practice session in a start scene, as the harness does."""
    start = _atoms(method=method, names=START)
    monkeypatch.setattr(type(method), "abstract_state", lambda self, **kwargs: start)
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    goal = frozenset({GroundAtom(predicate=IN_BIN, objects=(method.env.cube, method.env.bin))})
    method.get_practice_policy(task=Task(initial_state=state, goal=Goal(atoms=goal)))
    method.observe_practice_action_budget(remaining_actions=20)


def _holding(*, method: Tossing3DPomdpMethod) -> frozenset[GroundAtom]:
    start = _atoms(method=method, names=START)
    pick = _ground(method=method, name=PICK_SKILL, atoms=start)
    return apply_success_effects(
        true_atoms=start,
        ground_skill=pick,
        effects=method._pomdp_model._effects,  # noqa: SLF001
    )


def _with_toss_competence(*, state: Tossing3DBeliefState, rate: float) -> Tossing3DBeliefState:
    """Ten practice cycles of five tosses at `rate` and five clean picks: a toss belief
    centred near `rate` whose learning rate has had time to settle, as late in a run."""
    beliefs = dict(state.skill_beliefs)
    toss, pick = beliefs[TOSS_SKILL], beliefs[PICK_SKILL]
    hits = 0.0
    for _cycle in range(10):
        for _attempt in range(5):
            hits += rate
            success = hits >= 1.0
            hits -= float(success)
            toss = toss.condition_outcome(success=success)
            pick = pick.condition_outcome(success=True)
        toss = toss.advance_cycle(training_examples=5)
        pick = pick.advance_cycle(training_examples=0)
    beliefs[TOSS_SKILL], beliefs[PICK_SKILL] = toss, pick
    return state.model_copy(update={"skill_beliefs": beliefs})


@pytest.mark.parametrize("rate", [0.2, 0.5, 0.7])
def test_wrapped_astar_no_longer_stops_while_holding_the_cube(
    *, rate: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot's failure: a toss looked like a dead end, so determinized sampling
    noise flipped a margin of about +0.01 to STOP. Knowing the reset follows, a toss is
    worth "learn, then practise again", and A* tosses every time."""
    method = _method(solver="determinized_astar", wrapped=True)
    _start_session(method=method, monkeypatch=monkeypatch)
    method._pomdp_state = _with_toss_competence(state=method.pomdp_state, rate=rate)  # noqa: SLF001
    competence = mean_competence(belief=method.pomdp_state.skill_beliefs[TOSS_SKILL])
    assert abs(competence - rate) < 0.1
    holding = _holding(method=method)
    choices = [method.select_skill_to_practice(true_atoms=holding)[0].skill.name for _ in range(6)]
    assert choices == [TOSS_SKILL] * 6, (competence, choices)


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
def test_a_wrapped_toss_ends_in_the_start_state_and_pays_for_the_reset(
    *, solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every branch, success and failure alike, keeps the toss's own belief update and
    cost, then lands in the start state with the reset's price added to the plan cost."""
    wrapped = _method(solver=solver, wrapped=True)
    _start_session(method=wrapped, monkeypatch=monkeypatch)
    holding = _holding(method=wrapped)
    toss = _ground(method=wrapped, name=TOSS_SKILL, atoms=holding)
    state = wrapped.pomdp_state
    search_state = make_tossing3d_search_state(state=state, true_atoms=holding)
    model = wrapped._pomdp_model  # noqa: SLF001
    reset_aware = model.outcomes(environment_state=search_state, state=state, action=toss)
    naive = transition_outcomes(
        environment_state=search_state,
        state=state,
        action=toss,
        ground_skills=model.ground_skills,
        effects=model._effects,  # noqa: SLF001
        exploration_epsilon=model.exploration_epsilon,
        random_toss_competence=model.random_toss_competence,
        competence_evidence=model.competence_evidence,
    )
    assert len(reset_aware) == len(naive)
    start = _atoms(method=wrapped, names=START)
    for (p, belief, atoms), (q, naive_belief, naive_atoms) in zip(reset_aware, naive, strict=True):
        assert p == q
        assert atoms == start
        assert naive_atoms != start
        assert belief.skill_beliefs == naive_belief.skill_beliefs
        assert belief.accumulated_cost == pytest.approx(naive_belief.accumulated_cost + PRICE)


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
def test_the_reset_price_decides_whether_practice_is_worth_it(
    *, solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same belief, the same planner: at the domain's price the robot tosses; at a
    price that outweighs everything a toss could teach, it stops. The price is in the
    plan cost, not only in the bookkeeping."""
    choices = {}
    for price in (PRICE, 1e6):
        method = _method(solver=solver, wrapped=True, price=price)
        _start_session(method=method, monkeypatch=monkeypatch)
        method._pomdp_state = _with_toss_competence(state=method.pomdp_state, rate=0.5)  # noqa: SLF001
        choices[price] = method.select_skill_to_practice(true_atoms=_holding(method=method))
    assert choices[PRICE][0].skill.name == TOSS_SKILL
    assert choices[1e6][0].skill.name == "STOP"


def test_a_skill_without_continuous_parameters_is_not_followed_by_a_reset(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _method(solver="expectimax", wrapped=True)
    _start_session(method=method, monkeypatch=monkeypatch)
    start = _atoms(method=method, names=START)
    pick = _ground(method=method, name=PICK_SKILL, atoms=start)
    state = method.pomdp_state
    outcomes = method._pomdp_model.outcomes(  # noqa: SLF001
        environment_state=make_tossing3d_search_state(state=state, true_atoms=start),
        state=state,
        action=pick,
    )
    model = method._pomdp_model  # noqa: SLF001
    assert outcomes == transition_outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=start),
        state=state,
        action=pick,
        ground_skills=model.ground_skills,
        effects=model._effects,  # noqa: SLF001
        exploration_epsilon=model.exploration_epsilon,
        random_toss_competence=model.random_toss_competence,
        competence_evidence=model.competence_evidence,
    )
    assert any(atoms != start for _, _, atoms in outcomes)


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
def test_the_unwrapped_planner_is_unchanged(
    *, solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values pinned from the code before the reset-aware model existed: the canonical
    arm's decision and root values on a fixed belief are exactly what they were."""
    method = _method(solver=solver, wrapped=False)
    _start_session(method=method, monkeypatch=monkeypatch)
    method._pomdp_state = _with_toss_competence(state=method.pomdp_state, rate=0.5)  # noqa: SLF001
    action = method.select_skill_to_practice(true_atoms=_holding(method=method))
    values = method.practice_action_values()
    pinned_action, pinned_values = UNWRAPPED_PINNED[solver]
    assert action[0].skill.name == pinned_action
    assert values == pytest.approx(pinned_values, rel=1e-12)


_HUMAN = "ask_for_reset_cube_bin_only(robot, cube_0, bin_0, cuboid_barrier, robot_side, "
UNWRAPPED_PINNED: dict[str, tuple[str, dict[str, float]]] = {
    "expectimax": (
        TOSS_SKILL,
        {
            "STOP": 0.5677313363747786,
            TOSS_SKILL: 0.5804767038664549,
            _HUMAN + "robot_side)": 0.5773679034277341,
            _HUMAN + "opposite_side)": 0.5763621794761442,
        },
    ),
    "determinized_astar": (
        "ask_for_reset_cube_bin_only",
        {
            "STOP": 0.5677313363747786,
            TOSS_SKILL: 0.5448128406697347,
            _HUMAN + "robot_side)": 0.6804955577713723,
            _HUMAN + "opposite_side)": 0.5677253364050446,
        },
    ),
}


def test_without_a_session_the_wrapped_model_has_no_reset_destination() -> None:
    """The start state is read from the first practice session's scene; until then the
    wrapped model is the naive one, and the method says so rather than guessing."""
    method = _method(solver="expectimax", wrapped=True)
    assert method._pomdp_model.forced_start_reset is None  # noqa: SLF001


def test_a_forced_reset_to_a_different_symbolic_start_is_logged(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The model assumes every start scene abstracts alike; a reset that lands
    elsewhere is reported in the decision log, never silently adopted."""
    method = _method(solver="expectimax", wrapped=True)
    method.decision_log = tmp_path / "decisions.jsonl"
    _start_session(method=method, monkeypatch=monkeypatch)
    expected = method._pomdp_model.forced_start_reset  # noqa: SLF001
    assert expected is not None
    method.record_forced_reset(reason="parameterized_skill", cost=PRICE, consecutive_stranded=0)
    elsewhere = _holding(method=method)
    monkeypatch.setattr(type(method), "abstract_state", lambda self, **kwargs: elsewhere)
    method.observe_help_granted(state=None)  # type: ignore[arg-type]
    log = method.decision_log.read_text()
    assert "forced_reset_start_mismatch" in log
    assert method._pomdp_model.forced_start_reset == expected  # noqa: SLF001
