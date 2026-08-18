"""KINDER's one-shot state abstractor, exposed as `core.Predicate`s.

Every KINDER environment ships a `<Env>StateAbstractor` whose `state_abstractor(state)`
returns a `RelationalAbstractState` -- *all* of that environment's ground atoms, computed
together. `core.Predicate.holds` asks one predicate at a time. Bridging the two is this
module's whole job, and it is a lookup rather than a reimplementation: a predicate here
tests membership in the set upstream computed, so there is no second classifier to keep
in agreement with upstream's by test.

## Two things the shape forces, neither of them optional

**The abstractor is expensive.** It does a `PyBulletSim.set_state(state)` -- a full
forward-kinematics pass -- and then classifies. `Goal.is_satisfied` walks every atom in a
goal, and a planner evaluates every predicate over every state it considers, so a wrapper
that called through per predicate would pay for that pass N times per state. Hence a
cache, keyed on the state's contents.

**The abstractor is not pure, so the state is not a sufficient key.**
`Tossing3DStateAbstractor.state_abstractor` says it outright: *"poses come from `state`,
the goal region from the live simulator"*. `MovableInGoalRegion` is read off
`sim._ground_fixture` against the live compiled model, not off anything in the state. So
one `core.State` maps to different atom sets either side of a scene rebuild, and a cache
keyed on the state alone would serve the pre-rebuild answer forever.

The key is therefore `(generation, state contents)`, where `generation` is a counter this
object owns and the *simulator's owner* bumps through `invalidate()`. That places the
responsibility where the knowledge is: nothing about a `core.State` can reveal that
`env.reset(seed=...)` rebuilt the scene, but the code that called `reset` knows exactly.

## Why an undeclared predicate raises

If the abstractor emits an atom over a predicate the domain did not declare, this raises
rather than dropping it. Dropping would leave the operator model quietly incomplete, and
a silently incomplete symbolic model is precisely the drift that consuming upstream's
abstractor exists to eliminate -- a pin bump that adds a predicate should be a failing
test, not a subtly different plan.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.adapters.kinder.state_translation import KinderStateTranslator
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

# How many (generation, state) entries to keep. One would serve `Goal.is_satisfied`, which
# asks about a single state in a row; a planner interleaves a handful of successor states,
# and each entry is a small set of atoms, so a few cost nothing and avoid thrashing.
DEFAULT_CACHE_ENTRIES = 8


class KinderAbstraction(BaseModel):
    """A KINDER state abstractor plus the `core.Predicate`s that read its output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    translator: KinderStateTranslator
    # `<Env>StateAbstractor(sim).state_abstractor` -- a bound method, held as an opaque
    # callable so this class is generic over every KINDER environment.
    state_abstractor: Any
    cache_entries: int = DEFAULT_CACHE_ENTRIES

    _core_predicates: dict[str, Predicate] = PrivateAttr(default_factory=dict)
    _predicate_order: tuple[str, ...] = PrivateAttr(default=())
    _generation: int = PrivateAttr(default=0)
    _cache: dict[Any, frozenset[GroundAtom]] = PrivateAttr(default_factory=dict)

    @staticmethod
    def build(
        *,
        translator: KinderStateTranslator,
        state_abstractor: Any,
        kinder_predicates: tuple[Any, ...],
        cache_entries: int = DEFAULT_CACHE_ENTRIES,
    ) -> "KinderAbstraction":
        """Wrap one environment's abstractor, declaring which predicates it may report.

        `kinder_predicates` are `relational_structs.Predicate`s straight out of the
        environment's own `state_abstractions` module -- the domain names them, this
        translates them.
        """
        abstraction = KinderAbstraction(
            translator=translator,
            state_abstractor=state_abstractor,
            cache_entries=cache_entries,
        )
        for kinder_predicate in kinder_predicates:
            abstraction._register(kinder_predicate=kinder_predicate)
        return abstraction

    def predicates(self) -> tuple[Predicate, ...]:
        """Every declared predicate, in declaration order (what a `SkillProvider` returns)."""
        return tuple(self._core_predicates[name] for name in self._predicate_order)

    def predicate(self, *, name: str) -> Predicate:
        if name not in self._core_predicates:
            raise KeyError(
                f"no predicate named {name!r} was declared; known: {sorted(self._core_predicates)}"
            )
        return self._core_predicates[name]

    def invalidate(self) -> None:
        """Declare that the live simulator moved under every cached answer.

        Called by whatever owns the simulator, on any rebuild -- `env.reset(seed=...)`,
        a scene swap, a restore. Cheap: it bumps a counter and drops the cache, so
        calling it more often than strictly necessary costs one abstractor call.
        """
        self._generation += 1
        self._cache.clear()

    def atoms(self, *, state: State) -> frozenset[GroundAtom]:
        """Every ground atom that holds in `state`, as `core` atoms.

        One abstractor call per (generation, state), served from the cache afterwards.
        """
        key = (self._generation, self._digest(state=state))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kinder_state = self.translator.to_kinder_state(state=state)
        abstract_state = self.state_abstractor(kinder_state)
        atoms = frozenset(
            self._to_core_atom(kinder_atom=kinder_atom) for kinder_atom in abstract_state.atoms
        )
        if len(self._cache) >= self.cache_entries:
            # Plain FIFO rather than an LRU: the access pattern is a walk over successor
            # states, where recency and insertion order coincide, so an LRU's bookkeeping
            # would buy nothing.
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = atoms
        return atoms

    def _register(self, *, kinder_predicate: Any) -> None:
        name = kinder_predicate.name

        # `Predicate.holds` is a positional `(state, objects)` callable by its own
        # interface contract (`Goal.is_satisfied` calls it that way), so this adapts into
        # the keyword-only membership test below. A closure rather than the lambda the
        # domains use, because this one is built in a loop and needs to be annotated for
        # mypy; `name` is a fresh local per call, so there is no late binding to guard.
        def holds(state: State, objects: tuple[Object, ...]) -> bool:  # noqa: PLR0917
            return self._holds(state=state, name=name, objects=objects)

        typed_holds: Callable[[State, tuple[Object, ...]], bool] = holds
        core_predicate = Predicate(
            name=name,
            types=tuple(
                self.translator.core_types[kinder_type.name]
                for kinder_type in kinder_predicate.types
            ),
            holds=typed_holds,
        )
        self._core_predicates[name] = core_predicate
        self._predicate_order = (*self._predicate_order, name)

    def _holds(self, *, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        return GroundAtom(predicate=self._core_predicates[name], objects=objects) in self.atoms(
            state=state
        )

    def _to_core_atom(self, *, kinder_atom: Any) -> GroundAtom:
        name = kinder_atom.predicate.name
        if name not in self._core_predicates:
            raise ValueError(
                f"the abstractor reported {name!r}, which this domain did not declare; "
                f"declared: {sorted(self._core_predicates)}. Upstream has added a "
                "predicate: decide whether the operator model needs it rather than "
                "letting the symbolic layer silently drop it."
            )
        return GroundAtom(
            predicate=self._core_predicates[name],
            objects=tuple(
                self.translator.core_objects[kinder_object.name]
                for kinder_object in kinder_atom.objects
            ),
        )

    @staticmethod
    def _digest(*, state: State) -> tuple[tuple[str, bytes], ...]:
        """A hashable, exact fingerprint of a `core.State`'s contents.

        Exact bytes rather than a rounded or hashed summary: two states that differ in the
        last bit of a joint angle can differ in an atom, and a cache that conflated them
        would be a wrong answer rather than a stale one. Sorted by name, because
        `State.data` is a dict and its order is not part of its identity.
        """
        return tuple(
            sorted(
                (obj.name, np.ascontiguousarray(vector, dtype=np.float64).tobytes())
                for obj, vector in state.data.items()
            )
        )
