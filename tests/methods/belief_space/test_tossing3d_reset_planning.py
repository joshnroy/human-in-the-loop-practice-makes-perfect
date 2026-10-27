"""Known reset mechanics and symbolic self-loop dominance."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    NON_HUMAN_RESET_SKILL,
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    RESET_SKILLS,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    mean_competence,
    mean_cost,
    refit_belief_state,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    create_fixed_performance_cost_prior,
)
from hitl_pmp.methods.belief_space.types.search_state import Tossing3DSearchState
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief
from hitl_pmp.planning.grounding import SkillGrounder


def _method(*, offer_non_human_reset: bool = False) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, offer_non_human_reset=offer_non_human_reset),
        seed=0,
        pomdp_num_particles=32,
    )


def _offered_resets(*, method: Tossing3DPomdpMethod) -> set[str]:
    return {skill.name for skill in method.human_skills()}


def _atom(*, method: Tossing3DPomdpMethod, name: str) -> GroundAtom:
    return next(
        atom
        for atom in SkillGrounder.all_possible_ground_atoms(
            objects=method.objects(), predicates=method.predicates()
        )
        if atom.predicate.name == name
        and (
            not name.endswith("AtSide")
            or atom.objects[-1].name == ("opposite_side" if name == "BinAtSide" else "robot_side")
        )
    )


def _reset(*, method: Tossing3DPomdpMethod, name: str) -> GroundSkill:
    return next(
        skill
        for skill in method._pomdp_model.ground_skills  # noqa: SLF001
        if skill.skill.name == name
        and skill.objects[-2].name == "robot_side"
        and skill.objects[-1].name == "robot_side"
    )


@pytest.mark.parametrize("reset_name", [NON_HUMAN_RESET_SKILL])
def test_non_human_reset_success_remains_known_even_with_zero_performance_posterior(
    *, reset_name: str
) -> None:
    method = _method(offer_non_human_reset=True)
    model = method._pomdp_model  # noqa: SLF001
    reset = _reset(method=method, name=reset_name)
    beliefs = dict(method.pomdp_state.skill_beliefs)
    beliefs[reset_name] = create_fixed_performance_cost_prior(
        num_particles=32, seed=10, competence=0.0, learning_rate=0.0
    )
    state = method.pomdp_state.model_copy(update={"skill_beliefs": beliefs})
    before = reset.preconditions | frozenset({_atom(method=method, name="Holding")})
    search_state = make_tossing3d_search_state(state=state, true_atoms=before)
    outcomes = model.outcomes(environment_state=search_state, state=state, action=reset)

    assert len(outcomes) == 1
    probability, next_state, after = outcomes[0]
    assert probability == 1.0
    assert _atom(method=method, name="Holding") not in after
    assert _atom(method=method, name="ClosedEmpty") in after
    assert reset.add_effects <= after
    assert next_state.accumulated_cost == pytest.approx(mean_cost(belief=beliefs[reset_name]))
    assert next_state.skill_beliefs == state.skill_beliefs
    assert next_state.pending_examples == {**state.pending_examples, reset_name: 1}
    assert (
        model.transition_outcomes(
            environment_state=search_state, belief_state=state, practice_action=reset
        )[0][2]
        == 1.0
    )


@pytest.mark.parametrize("competence", [0.0, 0.2, 1.0])
def test_human_reset_forecasts_success_and_failure_from_competence(*, competence: float) -> None:
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    reset = _reset(method=method, name=RESET_SKILL)
    belief = WeightedHypothesisBelief(
        hypotheses=(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=competence, learning_rate=0.0),
                probability=1.0,
            ),
        )
    )
    state = method.pomdp_state.model_copy(
        update={"skill_beliefs": {**method.pomdp_state.skill_beliefs, RESET_SKILL: belief}}
    )
    before = reset.preconditions | frozenset({_atom(method=method, name="Holding")})
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=before),
        state=state,
        action=reset,
    )
    assert [p for p, _, _ in outcomes] == pytest.approx([
        p for p in (competence, 1 - competence) if p > 0
    ])
    for _, next_state, after in outcomes:
        if after != before:
            assert reset.add_effects <= after
            assert _atom(method=method, name="Holding") not in after
        else:
            assert competence < 1
        assert next_state.pending_examples == {**state.pending_examples, RESET_SKILL: 1}
        assert next_state.accumulated_cost == pytest.approx(reset.evaluate_practice_cost())
    assert (outcomes[0][2] != before) == (competence > 0)
    assert state.skill_beliefs[RESET_SKILL] == belief


def test_human_reset_conditions_only_its_own_hypothetical_belief() -> None:
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    reset = _reset(method=method, name=RESET_SKILL)
    belief = WeightedHypothesisBelief(
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=k, learning_rate=0.0), probability=0.5
            )
            for k in (0.2, 0.8)
        )
    )
    state = method.pomdp_state.model_copy(
        update={"skill_beliefs": {**method.pomdp_state.skill_beliefs, RESET_SKILL: belief}}
    )
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=reset.preconditions),
        state=state,
        action=reset,
    )
    assert [p for p, _, _ in outcomes] == pytest.approx([0.5, 0.5])
    assert [
        mean_competence(belief=s.skill_beliefs[RESET_SKILL]) for _, s, _ in outcomes
    ] == pytest.approx([0.68, 0.32])
    for _, next_state, _ in outcomes:
        assert {k: v for k, v in next_state.skill_beliefs.items() if k != RESET_SKILL} == {
            k: v for k, v in state.skill_beliefs.items() if k != RESET_SKILL
        }
    assert state.skill_beliefs[RESET_SKILL] == belief


@pytest.mark.parametrize("learning_rate", [0.0, 0.1])
def test_human_learning_changes_next_reset_forecast_after_refit(*, learning_rate: float) -> None:
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    reset = _reset(method=method, name=RESET_SKILL)
    belief = WeightedHypothesisBelief(
        hypotheses=(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=0.2, learning_rate=learning_rate),
                probability=1.0,
            ),
        )
    )
    state = method.pomdp_state.model_copy(
        update={"skill_beliefs": {**method.pomdp_state.skill_beliefs, RESET_SKILL: belief}}
    )
    _, after, atoms = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=reset.preconditions),
        state=state,
        action=reset,
    )[0]
    refitted = refit_belief_state(state=after)
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=refitted, true_atoms=atoms),
        state=refitted,
        action=reset,
    )
    assert outcomes[0][0] == pytest.approx(0.2 + learning_rate)


@pytest.mark.parametrize("reset_name", sorted(RESET_SKILLS))
def test_real_reset_still_updates_identical_cost_and_performance_filters(
    *, reset_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset that delivers what it promised -- read here through a patched
    `abstract_state` -- conditions its own belief exactly as a robot success would."""
    method = _method(offer_non_human_reset=True)
    reset = _reset(method=method, name=reset_name)
    before_scene, after_scene = (
        Tossing3DState(data={obj: np.zeros(obj.type.dim) for obj in method.objects()})
        for _ in range(2)
    )
    readings = {
        id(before_scene): frozenset(),
        id(after_scene): reset.expected_add_effects(before_atoms=frozenset()),
    }
    monkeypatch.setattr(
        Tossing3DPomdpMethod, "abstract_state", lambda _self, *, state: readings[id(state)]
    )
    method.env.current_state = before_scene
    before = method.pomdp_state.skill_beliefs[reset_name]
    method.record_action_cost(ground_skill=reset)
    method.observe_help_granted(state=after_scene)
    expected = before.condition_execution(success=True, observed_cost=5.0)
    assert method.pomdp_state.skill_beliefs[reset_name] == expected
    assert method.pomdp_state.pending_examples.get(reset_name, 0) == 1
    assert method.pomdp_state.accumulated_cost == 5.0
    refitted = refit_belief_state(state=method.pomdp_state)
    assert refitted.skill_beliefs[reset_name] == expected.refit(training_examples=1)


@pytest.mark.parametrize("gripper", ["HandEmpty", "ClosedEmpty"])
def test_a_reset_that_changes_no_atom_is_still_offered(*, gripper: str) -> None:
    """A reset re-randomizes the geometry even when no symbolic atom moves, so whether
    that is worth its cost is the planner's decision, not a pruning rule's."""
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    ready = frozenset(
        _atom(method=method, name=name)
        for name in (
            gripper,
            "OnGround",
            "NotHolding",
            "RobotAtSide",
            "CubeAtSide",
            "BinAtSide",
            "GraspClear",
            "PickPlannable",
            "PickupUnblocked",
            "BinOnGround",
        )
    )
    if gripper == "ClosedEmpty":
        ready |= {_atom(method=method, name="GripperClosed")}
    search_state = make_tossing3d_search_state(state=method.pomdp_state, true_atoms=ready)
    self_loop = next(
        reset
        for reset in model.ground_skills
        if reset.skill.name in RESET_SKILLS
        and reset.objects[-2].name == "robot_side"
        and reset.objects[-1].name == "opposite_side"
    )
    assert self_loop.preconditions <= ready
    assert (
        model.outcomes(environment_state=search_state, state=method.pomdp_state, action=self_loop)[
            0
        ][2]
        == ready
    )
    actions = model.get_valid_actions(environment_state=search_state)
    assert self_loop in actions
    assert {reset.objects[-1].name for reset in actions if reset.skill.name in RESET_SKILLS} == {
        "robot_side",
        "opposite_side",
    }
    assert {action.skill.name for action in actions if action.skill.name not in RESET_SKILLS} == {
        PICK_SKILL if gripper == "HandEmpty" else OPEN_GRIPPER_SKILL
    }


@pytest.mark.parametrize("condition", ["unreachable", "in_bin", "holding"])
def test_reset_recoveries_remain_available(*, condition: str) -> None:
    method = _method()
    names = {
        "unreachable": {"HandEmpty", "OnGround"},
        "in_bin": {"HandEmpty", "OnGround", "InBin"},
        "holding": {"Holding", "CubeAtSide"},
    }[condition]
    atoms = frozenset(
        _atom(method=method, name=name) for name in (names | {"RobotAtSide", "BinAtSide"})
    )
    actions = method._pomdp_model.get_valid_actions(  # noqa: SLF001
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=atoms)
    )
    assert {action.skill.name for action in actions} >= _offered_resets(method=method)


@pytest.mark.parametrize("cost_lambda", [0.0, 0.0003])
def test_deleting_a_reset_self_loop_preserves_the_continuation_and_saves_its_cost(
    *, cost_lambda: float
) -> None:
    method = _method()
    model = Tossing3DPracticeModel(
        ground_skills=method._pomdp_model.ground_skills,  # noqa: SLF001
        linear_cost_lambda=cost_lambda,
    )
    pick = next(skill for skill in model.ground_skills if skill.skill.name == PICK_SKILL)
    reset = _reset(method=method, name=sorted(RESET_SKILLS)[0])
    point = WeightedHypothesisBelief(
        hypotheses=(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=1.0, learning_rate=0.0), probability=1.0
            ),
        )
    )
    state = Tossing3DBeliefState(
        skill_beliefs={
            PICK_SKILL: point,
            TOSS_SKILL: point,
            OPEN_GRIPPER_SKILL: point,
            reset.skill.name: create_fixed_performance_cost_prior(
                num_particles=32, seed=10, competence=0.0, learning_rate=0.0
            ),
        }
    )
    ready = make_tossing3d_search_state(
        state=state, true_atoms=pick.preconditions | reset.preconditions | reset.add_effects
    )
    direct_probability, direct_state, direct_atoms = model.outcomes(
        environment_state=ready, state=state, action=pick
    )[0]
    reset_probability, reset_state, reset_atoms = model.outcomes(
        environment_state=ready, state=state, action=reset
    )[0]
    long_probability, long_state, long_atoms = model.outcomes(
        environment_state=make_tossing3d_search_state(state=reset_state, true_atoms=reset_atoms),
        state=reset_state,
        action=pick,
    )[0]

    assert direct_probability == reset_probability == long_probability == 1.0
    assert reset_atoms == ready.true_atoms
    assert long_atoms == direct_atoms
    assert long_state.skill_beliefs == direct_state.skill_beliefs
    # The detour's only belief difference is the reset's own clock tick.
    assert long_state.pending_examples == {
        **direct_state.pending_examples,
        reset.skill.name: 1,
    }
    saved_cost = mean_cost(belief=state.skill_beliefs[reset.skill.name])
    assert long_state.accumulated_cost - direct_state.accumulated_cost == pytest.approx(saved_cost)
    assert model.J(
        belief_state=direct_state, summed_cost=direct_state.accumulated_cost, num_samples=1
    ) - model.J(
        belief_state=long_state, summed_cost=long_state.accumulated_cost, num_samples=1
    ) == pytest.approx(cost_lambda * saved_cost)


def test_an_observed_pickup_refusal_leaves_the_pick_and_every_reset_available() -> None:
    """With PickupUnblocked absent (an observed refusal), the pick is still offered
    alongside every reset: whether to retry it is the planner's choice, and a refused
    retry is one more failed attempt for the pick's competence belief."""
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    refused = frozenset(
        _atom(method=method, name=name)
        for name in (
            "HandEmpty",
            "OnGround",
            "NotHolding",
            "RobotAtSide",
            "CubeAtSide",
            "BinAtSide",
            "GraspClear",
            "PickPlannable",
            "BinOnGround",
            # Deliberately NOT PickupUnblocked: the refusal was observed.
        )
    )
    actions = model.get_valid_actions(
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=refused)
    )
    names = {action.skill.name for action in actions}
    assert PICK_SKILL in names
    assert _offered_resets(method=method) <= names


def _ready_atoms(*, method: Tossing3DPomdpMethod) -> frozenset[GroundAtom]:
    return frozenset(
        _atom(method=method, name=name)
        for name in (
            "HandEmpty",
            "OnGround",
            "NotHolding",
            "RobotAtSide",
            "CubeAtSide",
            "BinAtSide",
            "GraspClear",
            "PickPlannable",
            "PickupUnblocked",
            "BinOnGround",
        )
    )


@pytest.mark.parametrize("scope", ["state", "skill"])
def test_a_pick_after_a_reset_reuses_the_roots_draw_within_its_scope(*, scope: str) -> None:
    """Determinized A*'s shared draws key a sampled outcome by what the outcome
    distribution reads. A reset advances the cost, its own belief and its own training
    clock, so the search node is new, but the pick's distribution is unchanged. Under
    "state" the pick reuses the root pick's draw when the reset left every atom as it
    was; under "skill" it does whatever the reset did to the atoms, because the pick's
    success probability reads only the pick's own belief."""
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    ready = _ready_atoms(method=method)
    root = make_tossing3d_search_state(state=method.pomdp_state, true_atoms=ready)
    actions = model.get_valid_actions(environment_state=root)
    pick = next(action for action in actions if action.skill.name == PICK_SKILL)
    resets = {
        reset.objects[-1].name: reset
        for reset in actions
        if reset.skill.name == RESET_SKILL and reset.objects[-2].name == "robot_side"
    }

    def after(*, reset: GroundSkill) -> Tossing3DSearchState:
        _, state, atoms = model.outcomes(
            environment_state=root, state=method.pomdp_state, action=reset
        )[0]
        return make_tossing3d_search_state(state=state, true_atoms=atoms)

    def key(*, node: Tossing3DSearchState, action: GroundSkill) -> object:
        return model.outcome_draw_key(
            environment_state=node, belief_state=node.state, action=action, scope=scope
        )

    unchanged, moved = after(reset=resets["opposite_side"]), after(reset=resets["robot_side"])
    assert unchanged.true_atoms == ready
    assert moved.true_atoms != ready
    for node in (unchanged, moved):
        assert model.search_cache_key(
            environment_state=node,
            summed_cost=node.state.accumulated_cost,
            belief_state=node.state,
            horizon=None,
        ) != model.search_cache_key(
            environment_state=root,
            summed_cost=root.state.accumulated_cost,
            belief_state=root.state,
            horizon=None,
        )
    assert key(node=unchanged, action=pick) == key(node=root, action=pick)
    assert (key(node=moved, action=pick) == key(node=root, action=pick)) == (scope == "skill")
    # The reset's own outcome reads its own belief, which the reset just conditioned.
    reset = resets["opposite_side"]
    assert key(node=unchanged, action=reset) != key(node=root, action=reset)
    # A second pick after an imagined pick outcome is a new trial in either scope.
    for _, picked_state, picked_atoms in model.outcomes(
        environment_state=root, state=method.pomdp_state, action=pick
    ):
        picked = make_tossing3d_search_state(state=picked_state, true_atoms=picked_atoms)
        assert key(node=picked, action=pick) != key(node=root, action=pick)


def test_the_belief_key_is_the_search_key_without_the_accumulated_cost() -> None:
    """Curtis et al.'s PO_DAStar keys a node by the belief alone. A reset that leaves
    every atom as it was still changes its own belief and training clock, so the node
    after it stays new; only the cost is dropped."""
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    ready = _ready_atoms(method=method)
    root = make_tossing3d_search_state(state=method.pomdp_state, true_atoms=ready)
    dearer_state = method.pomdp_state.model_copy(update={"accumulated_cost": 7.5})
    dearer = make_tossing3d_search_state(state=dearer_state, true_atoms=ready)

    def belief_key(*, node: Tossing3DSearchState) -> object:
        return model.belief_cache_key(environment_state=node, belief_state=node.state, horizon=4)

    def search_key(*, node: Tossing3DSearchState) -> object:
        return model.search_cache_key(
            environment_state=node,
            summed_cost=node.state.accumulated_cost,
            belief_state=node.state,
            horizon=4,
        )

    assert search_key(node=dearer) != search_key(node=root)
    assert belief_key(node=dearer) == belief_key(node=root)
    full = search_key(node=root)
    assert belief_key(node=root) == (*full[:-2], full[-1])
    reset = next(
        action
        for action in model.get_valid_actions(environment_state=root)
        if action.skill.name == RESET_SKILL and action.objects[-1].name == "opposite_side"
    )
    _, after_state, after_atoms = model.outcomes(
        environment_state=root, state=method.pomdp_state, action=reset
    )[0]
    after = make_tossing3d_search_state(state=after_state, true_atoms=after_atoms)
    assert after_atoms == ready
    assert belief_key(node=after) != belief_key(node=root)
