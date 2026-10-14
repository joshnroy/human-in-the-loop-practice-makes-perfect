"""Exact symmetry reduction over interchangeable objects in belief-space expectimax.

SweepOffShelf3D's five cubes are the case: every belief is per lifted skill, the
deployment value counts cubes rather than naming them, and the operators treat every
cube alike -- the `CubeOrder`-pinned sweep and human reset included, since their
effects are symmetric in their slots. What can tell two cubes apart is learned
per *ground* skill: a cost forecast from that grounding's own executions, a failure
effect observed in one exact state, or a starved parameter pool. The reduction must
keep exactly the symmetry those leave, and the tests below pin that it does, by
comparing every root action value against the unreduced search.
"""

import functools
from collections.abc import Mapping

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    SweepOffShelf3DAtoms,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.object_symmetry import InterchangeableObjects
from hitl_pmp.methods.belief_space.practice_domain import (
    SHELF_ROBOT_SKILLS,
    SWEEP_OFF_SHELF_DOMAIN,
    TOSSING3D_DOMAIN,
    SweepOffShelfDeployment,
    SweepOffShelfPracticeDomain,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    condition_skill_belief,
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    make_tossing3d_search_state,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief
from hitl_pmp.planning.grounding import SkillGrounder

CUBES = O.cubes
STATIC = SweepOffShelf3DPredicates.static_atoms()
LINEAR_COST_LAMBDA = 3e-6
SMALL_GRID = InferenceConfig(competence_bins=5, learning_rate_bins=2)
PRODUCTION_GRID = InferenceConfig(
    competence_bins=25,
    learning_rate_bins=16,
    sigma_competence=0.03,
    sigma_eta=0.005,
    learning_rate_decay=0.9,
    eta_max=0.15,
)


def _atoms(
    *,
    shelf=(),
    box=(),
    floor=(),
    held_cube=None,
    wiper="home",
) -> frozenset[GroundAtom]:
    """A symbolic shelf state with its summary atoms, as the backend classifies it."""
    named: set[tuple[str, tuple[str, ...]]] = set()
    named |= {("OnShelf", (CUBES[i].name,)) for i in shelf}
    named |= {("InBox", (CUBES[i].name, O.box.name)) for i in box}
    named |= {("OnFloor", (CUBES[i].name,)) for i in floor}
    if held_cube is not None:
        named.add(("HoldingCube", (O.robot.name, CUBES[held_cube].name)))
    elif wiper != "held":
        named.add(("HandEmpty", (O.robot.name,)))
    named.add(
        {
            "home": ("WiperHome", (O.wiper.name,)),
            "held": ("HoldingWiper", (O.robot.name, O.wiper.name)),
            "floor": ("WiperOnFloor", (O.wiper.name,)),
        }[wiper]
    )
    named = set(SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(named)))
    predicates = {p.name: p for p in SweepOffShelf3DPredicates.all()}
    objects = {o.name: o for o in O.all_objects()}
    return (
        frozenset(
            GroundAtom(predicate=predicates[name], objects=tuple(objects[n] for n in names))
            for name, names in named
        )
        | STATIC
    )


STATES = {
    "start": _atoms(shelf=range(5)),
    "swept": _atoms(box=range(5), wiper="held"),
    "scattered": _atoms(floor=(0, 1), box=(2, 3), shelf=(4,), wiper="floor"),
    "holding": _atoms(shelf=(0, 1), box=(3,), floor=(4,), held_cube=2),
}


class _CheapShelfDomain(SweepOffShelfPracticeDomain):
    """The shelf domain with a cheap deterministic J -- the deployment value at each
    skill's posterior-mean competence -- so the exactness tests can go deep. Symmetry
    is a property of the operators and the belief routing, not of J's estimator; the
    production J is checked separately at a shallower depth."""

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        competences = {
            name: np.array([beliefs[name].mean_competence()]) for name in SHELF_ROBOT_SKILLS
        }
        return float(
            SweepOffShelfDeployment.values(
                competences=competences, horizon=self.deployment_horizon
            )[0]
        )


CHEAP_DOMAIN = _CheapShelfDomain()


@functools.cache
def _ground_skills() -> tuple[GroundSkill, ...]:
    """Grounded as `Tossing3DPomdpMethod` grounds them: against every possible atom.
    Cached per process rather than a module fixture, so the module's slow searches
    still spread across xdist workers."""
    objects = O.all_objects()
    return tuple(
        SkillGrounder.applicable_ground_skills(
            skills=(*S.robot_skills(), *S.human_skills()),
            objects=objects,
            true_atoms=SkillGrounder.all_possible_ground_atoms(
                objects=objects, predicates=SweepOffShelf3DPredicates.all()
            ),
        )
    )


@pytest.fixture
def ground_skills() -> tuple[GroundSkill, ...]:
    return _ground_skills()


def _ground(*, ground_skills, name, cube) -> GroundSkill:
    return next(g for g in ground_skills if g.skill.name == name and CUBES[cube] in g.objects)


def _belief(*, domain, config=SMALL_GRID, costs=None) -> Tossing3DBeliefState:
    belief = make_default_tossing3d_belief(
        num_particles=64,
        seed=0,
        model="local_trend",
        engine="grid",
        inference_config=config,
        additional_skill_names=tuple(skill.name for skill in S.human_skills()),
        domain=domain,
    )
    if costs is not None:
        belief = belief.model_copy(update={"expected_execution_costs": costs})
    return belief


# Real outcomes observed before the search: the carry route (pick, place) has worked
# and the sweep route has not, so the cube-specific skills carry value -- at the prior
# the sweep route dominates and no cube-specific outcome can move J at all.
CARRY_ROUTE_EVIDENCE = {
    "PickCubeFromShelf": (True,) * 8 + (False,) * 2,
    "PlaceCubeInBox": (True,) * 9 + (False,),
    "PickCubeFromFloor": (True,) * 6 + (False,) * 2,
    "PickWiper": (False,) * 10,
    "SweepShelf": (False,) * 10,
}


def _trained(*, belief: Tossing3DBeliefState) -> Tossing3DBeliefState:
    beliefs = dict(belief.skill_beliefs)
    for name, outcomes in CARRY_ROUTE_EVIDENCE.items():
        for success in outcomes:
            beliefs[name] = condition_skill_belief(belief=beliefs[name], success=success)
    return belief.model_copy(update={"skill_beliefs": beliefs})


BELIEFS = {
    "prior": lambda domain, config=SMALL_GRID, costs=None: _belief(
        domain=domain, config=config, costs=costs
    ),
    "carry_trained": lambda domain, config=SMALL_GRID, costs=None: _trained(
        belief=_belief(domain=domain, config=config, costs=costs)
    ),
}


def _uniform_costs(*, ground_skills, steps=40.0) -> dict[str, float]:
    return {str(g): steps * 1e-3 for g in ground_skills}


def _detect(*, ground_skills, belief, **records) -> InterchangeableObjects:
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills, domain=CHEAP_DOMAIN, object_symmetry=True, **records
    )
    return model.object_symmetry_for(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=STATES["start"])
    )


def test_the_five_cubes_are_one_interchangeable_class(*, ground_skills):
    found = _detect(ground_skills=ground_skills, belief=_belief(domain=CHEAP_DOMAIN))
    assert [set(c) for c in found.classes] == [set(CUBES)]


def test_an_asymmetric_operator_set_breaks_the_class(*, ground_skills):
    """PickCubeFromShelf grounded for every cube but cube_4: cube_4 is distinguishable."""
    asymmetric = tuple(
        g
        for g in ground_skills
        if not (g.skill.name == "PickCubeFromShelf" and CUBES[4] in g.objects)
    )
    found = _detect(ground_skills=asymmetric, belief=_belief(domain=CHEAP_DOMAIN))
    assert [set(c) for c in found.classes] == [set(CUBES[:4])]


def test_a_per_grounding_cost_forecast_splits_the_class(*, ground_skills):
    """Step accounting forecasts each grounding's cost from its own executions, so a
    cube whose pick has run slower is no longer interchangeable with the rest."""
    costs = _uniform_costs(ground_skills=ground_skills)
    costs[str(_ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=0))] *= 2
    found = _detect(ground_skills=ground_skills, belief=_belief(domain=CHEAP_DOMAIN, costs=costs))
    assert [set(c) for c in found.classes] == [set(CUBES[1:])]


def test_equal_per_grounding_costs_keep_the_class(*, ground_skills):
    costs = _uniform_costs(ground_skills=ground_skills)
    found = _detect(ground_skills=ground_skills, belief=_belief(domain=CHEAP_DOMAIN, costs=costs))
    assert [set(c) for c in found.classes] == [set(CUBES)]


def _knocked_off_record(*, ground_skills, cube, before) -> FailureEffectCount:
    """A failed shelf pick of `cube` that knocked it to the floor."""
    on_shelf = next(a for a in before if a.predicate.name == "OnShelf" and CUBES[cube] in a.objects)
    on_floor = next(a for a in _atoms(floor=(cube,)) if a.predicate.name == "OnFloor")
    return FailureEffectCount(
        ground_skill=_ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=cube),
        before_atoms=before,
        was_random_exploration=False,
        add_effects=frozenset({on_floor}),
        delete_effects=frozenset({on_shelf}),
    )


def test_a_failure_record_splits_off_only_what_it_distinguishes(*, ground_skills):
    """At the start state every cube looks alike, so a record of cube_0's failed pick
    there distinguishes cube_0 alone; the other four stay interchangeable."""
    record = _knocked_off_record(ground_skills=ground_skills, cube=0, before=STATES["start"])
    found = _detect(
        ground_skills=ground_skills,
        belief=_belief(domain=CHEAP_DOMAIN),
        failure_effect_counts=(record,),
    )
    assert [set(c) for c in found.classes] == [set(CUBES[1:])]


def test_a_failure_record_in_an_asymmetric_state_pins_its_distinctions(*, ground_skills):
    """cube_3 is on the floor in the record's state, so no swap may move it; cube_0 is
    the picked cube. cube_1, cube_2 and cube_4 sit alike on the shelf and stay a class."""
    before = _atoms(shelf=(0, 1, 2, 4), floor=(3,))
    record = _knocked_off_record(ground_skills=ground_skills, cube=0, before=before)
    found = _detect(
        ground_skills=ground_skills,
        belief=_belief(domain=CHEAP_DOMAIN),
        failure_effect_counts=(record,),
    )
    assert [set(c) for c in found.classes] == [{CUBES[1], CUBES[2], CUBES[4]}]


def test_a_starved_pool_splits_off_its_cube(*, ground_skills):
    pool = (STATES["start"], _ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=2))
    found = _detect(
        ground_skills=ground_skills, belief=_belief(domain=CHEAP_DOMAIN), starved_pools=(pool,)
    )
    assert [set(c) for c in found.classes] == [{CUBES[0], CUBES[1], CUBES[3], CUBES[4]}]


def test_the_symmetry_follows_records_added_by_model_copy(*, ground_skills):
    """The method adds failure records with `model_copy`, which skips post-init: the
    symmetry must be derived from the model's current records, not cached at build."""
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills, domain=CHEAP_DOMAIN, object_symmetry=True
    )
    state = make_tossing3d_search_state(
        state=_belief(domain=CHEAP_DOMAIN), true_atoms=STATES["start"]
    )
    assert [set(c) for c in model.object_symmetry_for(environment_state=state).classes] == [
        set(CUBES)
    ]
    record = _knocked_off_record(ground_skills=ground_skills, cube=0, before=STATES["start"])
    updated = model.model_copy(update={"failure_effect_counts": (record,)})
    assert [set(c) for c in updated.object_symmetry_for(environment_state=state).classes] == [
        set(CUBES[1:])
    ]


def test_permuted_states_share_one_canonical_form(*, ground_skills):
    symmetry = _detect(ground_skills=ground_skills, belief=_belief(domain=CHEAP_DOMAIN))
    first = _atoms(floor=(0, 1), box=(2, 3), shelf=(4,), wiper="floor")
    second = _atoms(floor=(3, 4), box=(0, 2), shelf=(1,), wiper="floor")
    different = _atoms(floor=(0,), box=(1, 2, 3), shelf=(4,), wiper="floor")
    assert symmetry.canonical_atoms(atoms=first) == symmetry.canonical_atoms(atoms=second)
    assert symmetry.canonical_atoms(atoms=first) != symmetry.canonical_atoms(atoms=different)
    assert symmetry.canonical_atoms(atoms=first) >= STATIC


def test_one_representative_per_equivalent_action(*, ground_skills):
    model = Tossing3DPracticeModel(ground_skills=ground_skills, domain=CHEAP_DOMAIN)
    reduced = model.model_copy(update={"object_symmetry": True})
    state = make_tossing3d_search_state(
        state=_belief(domain=CHEAP_DOMAIN), true_atoms=STATES["scattered"]
    )
    valid = model.get_valid_actions(environment_state=state)
    kept = reduced.get_valid_actions(environment_state=state)
    assert len(valid) == 2 + 1 + 1 + 1  # floor picks, shelf pick, both human resets
    assert sorted(g.skill.name for g in kept) == sorted([
        "PickCubeFromFloor",
        "PickCubeFromShelf",
        "reset_cubes",
        "reset_wiper",
    ])


def _solve(*, model, atoms, depth, belief):
    trace = SearchTrace()
    planner = ExpectimaxPlanner(use_model_j=True, observation_probability_weight=0.0)
    value, action = planner.solve(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=atoms),
        summed_cost=belief.accumulated_cost,
        belief_state=belief,
        horizon=depth,
        model=model,
        trace=trace,
    )
    trace.close()
    by_key = {(g.skill.name, tuple(o.name for o in g.objects)): g for g in model.ground_skills}
    values = [
        (
            by_key[
                event["action"]["skill"]["name"],
                tuple(o["name"] for o in event["action"]["objects"]),
            ],
            event["value"],
        )
        for event in trace.events
        if event["node"] == 0 and event["event"] == "action_value"
    ]
    return value, action, values, planner.next_node


def _assert_exact(*, ground_skills, model, atoms, depth, belief):
    """The reduced search's root value, choice and every root action's value equal
    the unreduced search's, each unreduced action matched to its orbit representative."""
    full_value, full_action, full_values, full_nodes = _solve(
        model=model, atoms=atoms, depth=depth, belief=belief
    )
    reduced_model = model.model_copy(update={"object_symmetry": True})
    reduced_value, reduced_action, reduced_values, reduced_nodes = _solve(
        model=reduced_model, atoms=atoms, depth=depth, belief=belief
    )
    symmetry = reduced_model.object_symmetry_for(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=atoms)
    )
    assert reduced_value == pytest.approx(full_value, rel=1e-12, abs=1e-12)
    assert symmetry.equivalent(atoms=atoms, first=full_action, second=reduced_action)
    assert len(reduced_values) <= len(full_values)
    for action, value in full_values:
        matches = [
            v
            for representative, v in reduced_values
            if symmetry.equivalent(atoms=atoms, first=action, second=representative)
        ]
        assert len(matches) == 1, action
        assert matches[0] == pytest.approx(value, rel=1e-12, abs=1e-12), action
    assert reduced_nodes <= full_nodes
    return full_nodes, reduced_nodes


@pytest.mark.parametrize("belief", sorted(BELIEFS))
@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("state", sorted(STATES))
def test_the_reduced_search_returns_the_unreduced_action_values(
    *, ground_skills, state, depth, belief
):
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills, domain=CHEAP_DOMAIN, linear_cost_lambda=LINEAR_COST_LAMBDA
    )
    full_nodes, reduced_nodes = _assert_exact(
        ground_skills=ground_skills,
        model=model,
        atoms=STATES[state],
        depth=depth,
        belief=BELIEFS[belief](CHEAP_DOMAIN),
    )
    if depth >= 3:
        assert reduced_nodes < full_nodes


@pytest.mark.parametrize("state", ["start", "scattered"])
def test_exact_with_per_grounding_costs_and_failure_records(*, ground_skills, state):
    """The production run's sources of asymmetry, all at once: a failed pick of cube_1
    at the start state that knocked it to the floor, a starved floor pick of cube_0 in
    the scattered state, and a slower shelf pick for cube_2 -- the one cube the other
    two leave interchangeable (with cube_3), so the cost alone must split it off."""
    costs = _uniform_costs(ground_skills=ground_skills)
    costs[str(_ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=2))] *= 2
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills,
        domain=CHEAP_DOMAIN,
        linear_cost_lambda=LINEAR_COST_LAMBDA,
        failure_effect_counts=(
            _knocked_off_record(ground_skills=ground_skills, cube=1, before=STATES["start"]),
        ),
        starved_pools=(
            (
                STATES["scattered"],
                _ground(ground_skills=ground_skills, name="PickCubeFromFloor", cube=0),
            ),
        ),
    )
    _assert_exact(
        ground_skills=ground_skills,
        model=model,
        atoms=STATES[state],
        depth=3,
        belief=BELIEFS["carry_trained"](CHEAP_DOMAIN, costs=costs),
    )


@pytest.mark.parametrize("source", ["failure_record", "starved_pool", "cost"])
def test_exact_with_each_source_of_asymmetry_alone(*, ground_skills, source):
    """Each per-grounding record on its own, from a root where the cubes it concerns
    otherwise look alike, so a reduction that ignored it would merge unequal nodes."""
    records: dict[str, object] = {}
    costs = None
    root = STATES["start"]
    depth = 3
    if source == "failure_record":
        # A failed pick of cube_1 that left the gripper jammed (HandEmpty no longer
        # holds): nothing can be practised after it, so a wrong merge with another
        # cube's identity failure shows in the root values.
        start = STATES["start"]
        hand_empty = next(a for a in start if a.predicate.name == "HandEmpty")
        records["failure_effect_counts"] = (
            FailureEffectCount(
                ground_skill=_ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=1),
                before_atoms=start,
                was_random_exploration=False,
                add_effects=frozenset(),
                delete_effects=frozenset({hand_empty}),
            ),
        )
    elif source == "starved_pool":
        # cube_3's pick starved once it was the last cube on the shelf. From a root
        # with cube_2 and cube_3 alike on the shelf, carrying either one away reaches
        # a state with the other alone there, and only one of those is masked.
        root = _atoms(shelf=(2, 3), box=(0, 1, 4))
        depth = 4
        records["starved_pools"] = (
            (
                _atoms(shelf=(3,), box=(0, 1, 2, 4)),
                _ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=3),
            ),
        )
    else:
        costs = _uniform_costs(ground_skills=ground_skills)
        costs[str(_ground(ground_skills=ground_skills, name="PickCubeFromShelf", cube=4))] *= 2
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills,
        domain=CHEAP_DOMAIN,
        linear_cost_lambda=LINEAR_COST_LAMBDA,
        **records,
    )
    _assert_exact(
        ground_skills=ground_skills,
        model=model,
        atoms=root,
        depth=depth,
        belief=BELIEFS["carry_trained"](CHEAP_DOMAIN, costs=costs),
    )


@pytest.mark.parametrize("belief", sorted(BELIEFS))
@pytest.mark.parametrize("state", sorted(STATES))
def test_exact_under_the_production_model(*, ground_skills, state, belief):
    """The shelf domain's own J (4096 fixed-seed draws) and the production grid."""
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills,
        domain=SWEEP_OFF_SHELF_DOMAIN,
        linear_cost_lambda=LINEAR_COST_LAMBDA,
    )
    _assert_exact(
        ground_skills=ground_skills,
        model=model,
        atoms=STATES[state],
        depth=2,
        belief=BELIEFS[belief](SWEEP_OFF_SHELF_DOMAIN, config=PRODUCTION_GRID),
    )


@pytest.mark.parametrize("belief", sorted(BELIEFS))
@pytest.mark.parametrize("state", sorted(STATES))
def test_merging_beliefs_by_counts_matches_to_rounding(*, ground_skills, state, belief):
    """In-search conditioning is a product of likelihoods, so within one search a belief
    is a function of its S/F counts; only floating-point order separates two
    equal-count posteriors."""
    model = Tossing3DPracticeModel(
        ground_skills=ground_skills,
        domain=CHEAP_DOMAIN,
        linear_cost_lambda=LINEAR_COST_LAMBDA,
        object_symmetry=True,
    )
    root = BELIEFS[belief](CHEAP_DOMAIN)
    reduced_value, _, reduced_values, reduced_nodes = _solve(
        model=model, atoms=STATES[state], depth=3, belief=root
    )
    merged_model = model.model_copy(update={"merge_beliefs_by_counts": True})
    merged_value, _, merged_values, merged_nodes = _solve(
        model=merged_model, atoms=STATES[state], depth=3, belief=root
    )
    assert merged_value == pytest.approx(reduced_value, rel=1e-12, abs=1e-12)
    assert [a for a, _ in merged_values] == [a for a, _ in reduced_values]
    for (_, merged), (_, reduced) in zip(merged_values, reduced_values, strict=True):
        assert merged == pytest.approx(reduced, rel=1e-12, abs=1e-12)
    assert merged_nodes < reduced_nodes


def test_only_the_shelf_domain_reduces_its_search():
    assert SWEEP_OFF_SHELF_DOMAIN.reduce_expectimax_search
    assert not TOSSING3D_DOMAIN.reduce_expectimax_search


def test_the_shelf_searches_seven_skills_deep_and_tossing3d_three():
    assert SWEEP_OFF_SHELF_DOMAIN.default_search_depth == 7
    assert TOSSING3D_DOMAIN.default_search_depth == 3


def test_a_model_without_symmetry_reports_no_classes(*, ground_skills):
    model = Tossing3DPracticeModel(ground_skills=ground_skills, domain=CHEAP_DOMAIN)
    state = make_tossing3d_search_state(
        state=_belief(domain=CHEAP_DOMAIN), true_atoms=STATES["start"]
    )
    assert model.object_symmetry_for(environment_state=state).classes == ()
