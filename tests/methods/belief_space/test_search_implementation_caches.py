"""Expectimax's implementation caches change how fast a search runs, never what it finds.

Each cache memoises a pure function and is keyed on everything that function reads, so
the tests below check two things per cache: a hit returns exactly what the uncached
function returns, and any change to an input the function reads is a miss. The last
group replays searches recorded on the uncached code, before any of these caches
existed: root value, choice, node and hit counts, and every memo entry.
"""

import io
import json
import math
import pickle

import pytest

from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.sweepoffshelf3d.predicates import SweepOffShelf3DPredicates
from hitl_pmp.methods.belief_space import tossing3d_model
from hitl_pmp.methods.belief_space.belief_digest import BeliefDigest
from hitl_pmp.methods.belief_space.bounded_memo import BoundedMemo
from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    InferenceConfig,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.tossing3d_model import AtomMasks, Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    SearchForecastCache,
    condition_skill_belief,
    refit_belief_state,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    FrozenSkillEffects,
    SuccessEffectsCache,
    apply_success_effects,
    compute_success_effects,
    make_tossing3d_search_state,
    render_atoms,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    ParticleFilterBelief,
    create_broad_particle_prior,
)
from hitl_pmp.methods.belief_space.types.search_state import Tossing3DSearchState
from hitl_pmp.methods.belief_space.types.skill_belief import (
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)

from . import record_search_fixture
from . import test_object_symmetry as shelf
from .record_search_fixture import CASES, FIXTURE, RecordedSearch

SMALL = InferenceConfig(competence_bins=5, learning_rate_bins=3)


def _bayesian(*, seed: int = 0, engine: str = "grid") -> BayesianSkillBelief:
    return create_bayesian_prior(
        model="local_trend", engine=engine, seed=seed, num_particles=16, config=SMALL
    )


def _copy(*, belief: SkillBelief) -> SkillBelief:
    """Equal content in a fresh object, so identity cannot stand in for equality."""
    return type(belief).model_validate_json(belief.model_dump_json())


def _bayesian_variants(*, belief: BayesianSkillBelief) -> dict[str, BayesianSkillBelief]:
    """One belief per field of `belief`, each differing from it in that field alone."""
    weights = bytearray(belief.state_weights)
    weights[0] ^= 1
    values = bytearray(belief.latent_values)
    values[0] ^= 1
    variants = {
        "model_name": belief.model_copy(
            update={"model_name": "global_curve" if belief.model_name == "local_trend" else "x"}
        ),
        "engine": belief.model_copy(
            update={"engine": "grid" if belief.engine == "particle" else "particle"}
        ),
        "config": belief.model_copy(
            update={"config": SMALL.model_copy(update={"sigma_eta": 0.01})}
        ),
        "seed": belief.model_copy(update={"seed": belief.seed + 1}),
        "state_count": belief.model_copy(update={"state_count": belief.state_count + 1}),
        "latent_values": belief.model_copy(update={"latent_values": bytes(values)}),
        "state_weights": belief.model_copy(update={"state_weights": bytes(weights)}),
        "parent_indices": belief.model_copy(update={"parent_indices": b"\x00" * 8}),
        "cost_belief": belief.model_copy(
            update={"cost_belief": belief.cost_belief.condition_cost(observed_cost=1.0)}
        ),
    }
    for field in (
        "cycle_index",
        "total_training_examples",
        "incoming_training_examples",
        "cycle_successes",
        "cycle_failures",
        "resampling_count",
        "process_transition_count",
    ):
        variants[field] = belief.model_copy(update={field: getattr(belief, field) + 1})
    assert set(variants) == set(BayesianSkillBelief.model_fields), "a field has no variant"
    return variants


def _particle_variants(*, belief: ParticleFilterBelief) -> dict[str, ParticleFilterBelief]:
    weights = bytearray(belief.particle_weights)
    weights[0] ^= 1
    parameters = bytearray(belief.particle_parameters)
    parameters[0] ^= 1
    variants = {
        "particle_parameters": belief.model_copy(update={"particle_parameters": bytes(parameters)}),
        "particle_weights": belief.model_copy(update={"particle_weights": bytes(weights)}),
        "num_particles": belief.model_copy(update={"num_particles": belief.num_particles + 1}),
    }
    for field in ("resampling_count", "process_transition_count", "resampling_seed"):
        variants[field] = belief.model_copy(update={field: getattr(belief, field) + 1})
    assert set(variants) == set(ParticleFilterBelief.model_fields), "a field has no variant"
    return variants


# --- BoundedMemo -----------------------------------------------------------------------


def test_a_bounded_memo_empties_whole_when_it_reaches_its_entry_ceiling() -> None:
    memo: BoundedMemo[int, str] = BoundedMemo(max_entries=2)
    memo.put(key=1, value="one")
    memo.put(key=2, value="two")
    assert memo.get(key=1) == "one"
    memo.put(key=3, value="three")
    assert memo.get(key=1) is None
    assert memo.get(key=3) == "three"
    assert len(memo) == 1


def test_a_bounded_memo_empties_whole_when_it_reaches_its_byte_ceiling() -> None:
    memo: BoundedMemo[int, str] = BoundedMemo(max_entries=100, max_bytes=10)
    memo.put(key=1, value="one", nbytes=6)
    memo.put(key=2, value="two", nbytes=6)
    assert memo.get(key=1) is None
    assert memo.get(key=2) == "two"


# --- Signature completeness: the premise every digest-keyed cache rests on -----------------


def test_exactly_the_beliefs_whose_signature_covers_every_field_are_marked_complete() -> None:
    def subclasses(*, cls: type) -> set[type]:
        found = set(cls.__subclasses__())
        for sub in tuple(found):
            found |= subclasses(cls=sub)
        return found

    marked = {cls for cls in subclasses(cls=SkillBelief) if cls.SIGNATURE_COVERS_EVERY_FIELD}
    assert marked == {BayesianSkillBelief, ParticleFilterBelief}


@pytest.mark.parametrize("engine", ["grid", "particle"])
def test_every_bayesian_field_moves_its_digest(*, engine: str) -> None:
    belief = _bayesian(engine=engine)
    digest = BeliefDigest.of(belief=belief)
    assert BeliefDigest.of(belief=_copy(belief=belief)) == digest
    for field, variant in _bayesian_variants(belief=belief).items():
        if field == "state_count":
            # Fixed by the weight buffer's length, which the digest does read.
            continue
        assert BeliefDigest.of(belief=variant) != digest, field


def test_every_particle_field_moves_its_digest() -> None:
    belief = create_broad_particle_prior(num_particles=8, seed=0)
    digest = BeliefDigest.of(belief=belief)
    for field, variant in _particle_variants(belief=belief).items():
        if field == "num_particles":
            continue
        assert BeliefDigest.of(belief=variant) != digest, field


# --- refit_belief_state's per-belief refit ---------------------------------------------


@pytest.mark.parametrize("engine", ["grid", "particle"])
def test_the_cached_refit_is_the_uncached_refit(*, engine: str) -> None:
    SearchForecastCache.clear()
    belief = _bayesian(engine=engine).condition_outcome(success=False)
    for count in (0, 1, 3):
        cached = SearchForecastCache.refit(belief=belief, training_examples=count)
        assert cached == belief.refit(training_examples=count)
        assert SearchForecastCache.refit(belief=_copy(belief=belief), training_examples=count) is (
            cached
        )


@pytest.mark.parametrize("engine", ["grid", "particle"])
def test_the_refit_cache_misses_on_every_changed_input(*, engine: str) -> None:
    SearchForecastCache.clear()
    belief = _bayesian(engine=engine)
    first = SearchForecastCache.refit(belief=belief, training_examples=2)
    assert SearchForecastCache.refit(belief=belief, training_examples=3) == belief.refit(
        training_examples=3
    )
    assert SearchForecastCache.refit(belief=belief, training_examples=3) is not first
    for field, variant in _bayesian_variants(belief=belief).items():
        if field in ("state_count", "engine", "model_name", "parent_indices"):
            continue  # not a valid belief on its own; the digest test covers the field
        refitted = SearchForecastCache.refit(belief=variant, training_examples=2)
        assert refitted is not first, field
        assert refitted == variant.refit(training_examples=2), field


def test_refit_belief_state_is_unchanged_by_the_cache() -> None:
    beliefs = {
        "a": _bayesian(seed=1).condition_outcome(success=True),
        "b": _bayesian(seed=2, engine="particle"),
        "c": create_broad_particle_prior(num_particles=8, seed=3),
    }
    state = Tossing3DBeliefState(skill_beliefs=beliefs, pending_examples={"a": 2, "c": 1})
    expected = {
        "a": beliefs["a"].refit(training_examples=2),
        "b": beliefs["b"].refit(training_examples=0),
        "c": beliefs["c"].refit(training_examples=1).advance_learning_rate(process_noise_std=0.0),
    }
    for _ in range(2):
        assert refit_belief_state(state=state).skill_beliefs == expected


# --- condition_skill_belief ----------------------------------------------------------


@pytest.mark.parametrize("engine", ["grid", "particle"])
def test_the_cached_conditioning_is_the_uncached_conditioning(*, engine: str) -> None:
    SearchForecastCache.clear()
    belief = _bayesian(engine=engine)
    for success in (True, False):
        cached = condition_skill_belief(belief=belief, success=success)
        assert cached == belief.condition_outcome(success=success)
        assert condition_skill_belief(belief=_copy(belief=belief), success=success) is cached
    assert condition_skill_belief(belief=belief, success=True) != condition_skill_belief(
        belief=belief, success=False
    )


def test_the_conditioning_cache_misses_on_every_changed_input() -> None:
    SearchForecastCache.clear()
    belief = _bayesian()
    first = condition_skill_belief(belief=belief, success=True)
    for field, variant in _bayesian_variants(belief=belief).items():
        if field in ("state_count", "engine", "model_name", "parent_indices"):
            continue
        conditioned = condition_skill_belief(belief=variant, success=True)
        assert conditioned is not first, field
        assert conditioned == variant.condition_outcome(success=True), field


def test_a_belief_whose_signature_omits_a_field_is_never_served_from_the_cache() -> None:
    """A weighted belief's signature leaves out each hypothesis's cost, so two beliefs
    differing only there share a digest; each must still get its own posterior."""

    def weighted(*, cost: float) -> WeightedHypothesisBelief:
        return WeightedHypothesisBelief(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(competence=c, learning_rate=0.1, cost=cost),
                    probability=0.5,
                )
                for c in (0.2, 0.8)
            )
        )

    cheap, dear = weighted(cost=1.0), weighted(cost=5.0)
    assert BeliefDigest.of(belief=cheap) == BeliefDigest.of(belief=dear)
    assert condition_skill_belief(belief=cheap, success=True) == cheap.condition_outcome(
        success=True
    )
    assert condition_skill_belief(belief=dear, success=True) == dear.condition_outcome(success=True)


# --- apply_success_effects ------------------------------------------------------------


def _effects(*, ground_skills):
    return FrozenSkillEffects(
        table={g: (g.add_effects, g.delete_effects, g.ignore_effects) for g in ground_skills}
    )


def test_cached_success_effects_are_the_uncached_ones_and_the_same_object() -> None:
    SuccessEffectsCache.clear()
    ground_skills = shelf._ground_skills()
    effects = _effects(ground_skills=ground_skills)
    atoms = shelf.STATES["start"]
    applicable = [g for g in ground_skills if g.preconditions <= atoms]
    assert applicable
    for ground_skill in applicable:
        cached = apply_success_effects(true_atoms=atoms, ground_skill=ground_skill, effects=effects)
        assert cached == compute_success_effects(
            true_atoms=atoms, ground_skill=ground_skill, effects=effects
        )
        again = apply_success_effects(
            true_atoms=frozenset(atoms), ground_skill=ground_skill, effects=effects
        )
        assert again is cached


def test_the_success_effects_cache_misses_on_every_changed_input() -> None:
    SuccessEffectsCache.clear()
    ground_skills = shelf._ground_skills()
    effects = _effects(ground_skills=ground_skills)
    first_skill, second_skill = [
        g for g in ground_skills if g.preconditions <= shelf.STATES["start"]
    ][:2]
    start = apply_success_effects(
        true_atoms=shelf.STATES["start"], ground_skill=first_skill, effects=effects
    )
    # Another skill, the same atoms.
    assert apply_success_effects(
        true_atoms=shelf.STATES["start"], ground_skill=second_skill, effects=effects
    ) == compute_success_effects(
        true_atoms=shelf.STATES["start"], ground_skill=second_skill, effects=effects
    )
    # The same skill, other atoms.
    other_atoms = shelf.STATES["start"] | shelf.STATES["scattered"]
    assert apply_success_effects(
        true_atoms=other_atoms, ground_skill=first_skill, effects=effects
    ) == compute_success_effects(true_atoms=other_atoms, ground_skill=first_skill, effects=effects)
    # Another effects table, in which this skill adds nothing and deletes nothing.
    inert = FrozenSkillEffects(
        table={**effects, first_skill: (frozenset(), frozenset(), frozenset())}
    )
    assert apply_success_effects(
        true_atoms=shelf.STATES["start"], ground_skill=first_skill, effects=inert
    ) == compute_success_effects(
        true_atoms=shelf.STATES["start"], ground_skill=first_skill, effects=inert
    )
    assert (
        compute_success_effects(
            true_atoms=shelf.STATES["start"], ground_skill=first_skill, effects=inert
        )
        != start
    )


def test_a_mutable_effects_table_is_never_served_from_the_cache() -> None:
    SuccessEffectsCache.clear()
    ground_skills = shelf._ground_skills()
    ground_skill = next(g for g in ground_skills if g.preconditions <= shelf.STATES["start"])
    table = dict(_effects(ground_skills=ground_skills))
    atoms = shelf.STATES["start"]
    before = apply_success_effects(true_atoms=atoms, ground_skill=ground_skill, effects=table)
    table[ground_skill] = (frozenset(), frozenset(), frozenset())
    after = apply_success_effects(true_atoms=atoms, ground_skill=ground_skill, effects=table)
    assert after != before
    assert after == compute_success_effects(
        true_atoms=atoms, ground_skill=ground_skill, effects=table
    )


def test_the_model_hands_the_transition_a_read_only_effects_table() -> None:
    model = Tossing3DPracticeModel(ground_skills=shelf._ground_skills())
    assert isinstance(model._effects, FrozenSkillEffects)  # noqa: SLF001


def test_the_model_and_its_caches_survive_a_checkpoint_pickle() -> None:
    """Method checkpoints pickle the model, referring to its predicates (whose `holds`
    is a lambda) by name; every cache the model holds must pickle too."""
    predicates = {p.name: p for p in SweepOffShelf3DPredicates.all()}

    class ByName(pickle.Pickler):
        def persistent_id(self, obj: object) -> str | None:  # noqa: PLR0917 -- pickle API
            return obj.name if isinstance(obj, Predicate) else None

    class FromName(pickle.Unpickler):
        def persistent_load(self, pid: str) -> Predicate:  # noqa: PLR0917 -- pickle API
            return predicates[pid]

    model = Tossing3DPracticeModel(ground_skills=shelf._ground_skills())
    model._atoms_mask(atoms=shelf.STATES["start"])  # noqa: SLF001 -- warm its mask table
    buffer = io.BytesIO()
    ByName(buffer).dump(model)
    buffer.seek(0)
    restored = FromName(buffer).load()
    assert dict(restored._effects) == dict(model._effects)  # noqa: SLF001
    for atoms in shelf.STATES.values():
        assert restored._atoms_mask(atoms=atoms) == model._atoms_mask(atoms=atoms)  # noqa: SLF001


# --- _atoms_mask ---------------------------------------------------------------------


def test_the_cached_atoms_mask_is_the_uncached_one() -> None:
    model = Tossing3DPracticeModel(ground_skills=shelf._ground_skills())
    masks = model._atom_masks  # noqa: SLF001
    for atoms in shelf.STATES.values():
        expected = masks.compute(atoms=atoms)
        assert model._atoms_mask(atoms=atoms) == expected  # noqa: SLF001
        assert model._atoms_mask(atoms=frozenset(atoms)) == expected  # noqa: SLF001
        assert model._atoms_mask(atoms=set(atoms)) == expected  # noqa: SLF001
    assert len({model._atoms_mask(atoms=a) for a in shelf.STATES.values()}) == len(shelf.STATES)  # noqa: SLF001


def test_the_atoms_mask_cache_belongs_to_its_atom_index() -> None:
    atoms = shelf.STATES["start"]
    ordered = sorted(atoms, key=str)
    forward = AtomMasks(indexes={atom: i for i, atom in enumerate(ordered)})
    backward = AtomMasks(indexes={atom: i for i, atom in enumerate(reversed(ordered))})
    partial = AtomMasks(indexes={ordered[0]: 0})
    assert forward.mask(atoms=atoms) == backward.mask(atoms=atoms)
    first = frozenset(ordered[:1])
    assert forward.mask(atoms=first) != backward.mask(atoms=first)
    assert partial.mask(atoms=atoms) == 1
    assert forward.mask(atoms=atoms) == forward.compute(atoms=atoms)


# --- transition_key -------------------------------------------------------------------


def _pydantic_transition_key(*, environment_state, cost: float) -> object:
    """The key before belief digests: pydantic equality and hashing of every belief."""
    state = environment_state.state
    return (
        environment_state.atoms,
        tuple(sorted(state.skill_beliefs.items())),
        tuple(sorted(state.pending_examples.items())),
        tuple(sorted(state.sampler_training.items())),
        state.accumulated_cost,
        cost,
    )


def _beliefs_for_merging() -> list[SkillBelief]:
    grid = _bayesian()
    particle = _bayesian(engine="particle")
    filtered = create_broad_particle_prior(num_particles=8, seed=0)
    weighted = WeightedHypothesisBelief.broad_prior(points_per_dimension=3)
    found: list[SkillBelief] = [grid, particle, filtered, weighted]
    found += [_copy(belief=b) for b in found]
    # Every single-field variant but the two counts no valid belief can set apart from
    # its buffers' length (the validators pin state_count and num_particles to it).
    found += [v for f, v in _bayesian_variants(belief=grid).items() if f != "state_count"]
    found += [v for f, v in _particle_variants(belief=filtered).items() if f != "num_particles"]
    found += [grid.condition_outcome(success=True), grid.condition_outcome(success=False)]
    found += [
        weighted.model_copy(
            update={
                "hypotheses": tuple(
                    item.model_copy(
                        update={"hypothesis": item.hypothesis.model_copy(update={"cost": 2.0})}
                    )
                    for item in weighted.hypotheses
                )
            }
        )
    ]
    return found


def test_the_transition_key_merges_exactly_what_pydantic_equality_merged() -> None:
    beliefs = _beliefs_for_merging()
    atoms = shelf.STATES["start"]
    states = [
        make_tossing3d_search_state(
            state=Tossing3DBeliefState.model_construct(skill_beliefs={"skill": belief}),
            true_atoms=atoms,
        )
        for belief in beliefs
    ]
    for first in states:
        for second in states:
            old = _pydantic_transition_key(environment_state=first, cost=1.0) == (
                _pydantic_transition_key(environment_state=second, cost=1.0)
            )
            new = Tossing3DPracticeModel.transition_key(
                environment_state=first, cost=1.0
            ) == Tossing3DPracticeModel.transition_key(environment_state=second, cost=1.0)
            assert old == new, (first.state.skill_beliefs, second.state.skill_beliefs)


def test_the_transition_key_still_reads_every_other_field() -> None:
    belief = _bayesian()
    atoms = shelf.STATES["start"]
    base = Tossing3DBeliefState(skill_beliefs={"skill": belief})
    changed = [
        base.model_copy(update={"pending_examples": {"skill": 1}}),
        base.model_copy(update={"accumulated_cost": 1.0}),
        base.model_copy(update={"skill_beliefs": {"other": belief}}),
    ]
    key = Tossing3DPracticeModel.transition_key(
        environment_state=make_tossing3d_search_state(state=base, true_atoms=atoms), cost=1.0
    )
    for state in changed:
        assert key != Tossing3DPracticeModel.transition_key(
            environment_state=make_tossing3d_search_state(state=state, true_atoms=atoms), cost=1.0
        )
    assert key != Tossing3DPracticeModel.transition_key(
        environment_state=make_tossing3d_search_state(
            state=base, true_atoms=shelf.STATES["scattered"]
        ),
        cost=1.0,
    )
    assert key != Tossing3DPracticeModel.transition_key(
        environment_state=make_tossing3d_search_state(state=base, true_atoms=atoms), cost=2.0
    )


# --- make_tossing3d_search_state ------------------------------------------------------


def test_the_search_state_is_built_without_revalidating_its_atoms() -> None:
    state = Tossing3DBeliefState(skill_beliefs={"skill": _bayesian()})
    atoms = shelf.STATES["start"]
    built = make_tossing3d_search_state(state=state, true_atoms=atoms)
    # The same frozenset object, so its cached hash is reused by every key it enters.
    assert built.true_atoms is atoms
    assert built.state is state
    validated = Tossing3DSearchState(state=state, true_atoms=atoms, atoms=built.atoms)
    assert built == validated
    assert built.model_fields_set == validated.model_fields_set
    assert built.model_dump() == validated.model_dump()


# --- Recorded decisions --------------------------------------------------------------


def _uncached(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every cached function back to its uncached original, as on main."""

    def build(*, state, true_atoms):
        return Tossing3DSearchState(
            state=state, true_atoms=true_atoms, atoms=render_atoms(true_atoms=true_atoms)
        )

    monkeypatch.setattr(
        SearchForecastCache,
        "refit",
        staticmethod(
            lambda *, belief, training_examples: belief.refit(training_examples=training_examples)
        ),
    )
    monkeypatch.setattr(
        SearchForecastCache,
        "condition",
        staticmethod(lambda *, belief, success: belief.condition_outcome(success=success)),
    )
    monkeypatch.setattr(SuccessEffectsCache, "of", staticmethod(compute_success_effects))
    monkeypatch.setattr(AtomMasks, "mask", AtomMasks.compute)
    monkeypatch.setattr(
        Tossing3DPracticeModel, "transition_key", staticmethod(_pydantic_transition_key)
    )
    for module in (tossing3d_model, record_search_fixture):
        monkeypatch.setattr(module, "make_tossing3d_search_state", build)


@pytest.mark.parametrize(("state", "reduce"), CASES)
def test_the_caches_change_no_bit_of_a_search(
    *, monkeypatch: pytest.MonkeyPatch, state: str, reduce: bool
) -> None:
    """The same search, with every cache routed back to its uncached original and then
    with the caches, cold and warm, in one process: every memo entry -- each node's key,
    value to the bit and best action -- and the root's value, choice, node and hit
    counts must match. One process, because the low bits of a posterior differ between
    CPUs even for the uncached code (see the recorded test below)."""
    with monkeypatch.context() as patched:
        _uncached(monkeypatch=patched)
        uncached = RecordedSearch.solve(state=state, reduce=reduce)
    SearchForecastCache.clear()
    SuccessEffectsCache.clear()
    for _ in range(2):
        assert RecordedSearch.solve(state=state, reduce=reduce) == uncached


@pytest.fixture(scope="module")
def recorded() -> dict[str, dict]:
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize(("state", "reduce"), CASES)
def test_a_recorded_decision_is_reproduced(
    *, recorded: dict[str, dict], state: str, reduce: bool
) -> None:
    """SweepOffShelf3D's production J and grid, a carry-trained belief, depth 3, with and
    without the symmetry and count reductions; recorded on agni from main @ 746f77732,
    before any of these caches existed. Choice and node and hit counts are exact; the
    root value is compared to 1e-12, not to the bit, because main itself does not
    reproduce agni's last bit on every CPU (on a Della Sapphire Rapids node it gives
    ...e99bap-2 for agni's ...e99bcp-2). Bit-identity is the test above's job."""
    expected = recorded[RecordedSearch.name(state=state, reduce=reduce)]
    observed = RecordedSearch.solve(state=state, reduce=reduce)
    for field in ("action", "unique_nodes", "cache_hits"):
        assert observed[field] == expected[field], field
    assert math.isclose(
        float.fromhex(observed["value"]), float.fromhex(expected["value"]), rel_tol=1e-12
    )
