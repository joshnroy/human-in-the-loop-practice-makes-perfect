"""Exact symmetry reduction over interchangeable objects for belief-space expectimax.

Two objects are interchangeable when swapping them is an automorphism of everything
the search reads:

1. the operators -- the swap maps the set of statically applicable ground skills onto
   itself, each onto one with the same lifted name, forecast cost and dynamic
   preconditions and effects (a cost forecast is per *grounding* under step
   accounting, so a cube whose pick has run slower is told apart by it);
2. the beliefs -- competence is per lifted skill, so no belief names an object;
3. the stopping value -- J reads only the beliefs, never the atoms;
4. what is learned per exact ground skill and state -- failure-effect records and
   starved parameter pools: the swap must map the set of them onto itself;
5. any fixed state a transition can jump to (the always-ask wrapper's reset
   destination): the swap must map it onto itself.

(1), (4) and (5) are checked here; (2) and (3) hold for every `PracticeDomain`, whose J
and deployment value are functions of `skill_beliefs` alone. Swaps that are
automorphisms compose, and conjugating one by another gives a third, so "swapping
a and b is an automorphism" is an equivalence relation: its classes are found by
testing one swap per object against each class found so far, and every permutation
within a class is then an automorphism too.

Given that, a node's expectimax value is invariant under renaming the objects of a
class, so the memo key can use a canonical renaming of the atoms, and each node needs
one ground action per stabiliser orbit. Static atoms (predicates no skill changes,
such as SweepOffShelf3D's `CubeOrder` chain) are never renamed: they decide which
groundings exist, and (1) is checked over the groundings they admit. That is what
lets the chain-pinned `reset_cubes`, whose effects are symmetric in its slots, count
as symmetric even though the chain is not. A one-cube skill (`SweepShelf`, the picks)
grounds once per cube, and the swap maps those groundings onto each other.

Grouping actions into orbits by per-object atom profile is exact only when no dynamic
atom mentions two class members; `detect` drops a class otherwise.
"""

from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .types.failure_effects import FailureEffectCount

Signature = tuple[object, ...]
Mapping = dict[Object, Object]


class InterchangeableObjects(BaseModel):
    """Classes of mutually interchangeable objects, each in a fixed canonical order."""

    model_config = ConfigDict(frozen=True)

    classes: tuple[tuple[Object, ...], ...] = ()
    static_predicates: frozenset[str] = frozenset()

    @staticmethod
    def detect(
        *,
        ground_skills: Iterable[GroundSkill],
        dynamic_predicates: frozenset[str],
        static_atoms: frozenset[GroundAtom],
        cost_of: Callable[[GroundSkill], float],
        failure_effect_counts: Iterable[FailureEffectCount] = (),
        starved_pools: Iterable[tuple[frozenset[GroundAtom], GroundSkill]] = (),
        invariant_atoms: frozenset[GroundAtom] = frozenset(),
    ) -> "InterchangeableObjects":
        """The interchangeable classes given the root's static atoms and cost forecasts,
        which no search transition changes."""
        none = InterchangeableObjects(
            static_predicates=frozenset(
                atom.predicate.name
                for atom in static_atoms
                if atom.predicate.name not in dynamic_predicates
            )
        )
        renamer = _Renamer(dynamic_predicates=dynamic_predicates, cost_of=cost_of)
        applicable = [
            skill
            for skill in dict.fromkeys(ground_skills)
            if all(
                atom in static_atoms
                for atom in skill.preconditions
                if atom.predicate.name not in dynamic_predicates
            )
        ]
        if any(
            atom.predicate.name not in dynamic_predicates
            for skill in applicable
            for effect in skill.conditional_add_effects
            for atom in effect.conditions
        ):
            return none
        signatures = {renamer.signature(skill=skill) for skill in applicable}
        if len(signatures) < len(applicable):
            # Two groundings indistinguishable to the search: a renamed record could
            # not be matched to the one ground skill it names.
            return none
        records = {renamer.record(record=record, mapping={}) for record in failure_effect_counts}
        pools = {renamer.pool(pool=pool, mapping={}) for pool in starved_pools}

        def is_automorphism(*, first: Object, second: Object) -> bool:
            mapping = {first: second, second: first}
            return (
                all(
                    renamer.rename_signature(signature=s, mapping=mapping) in signatures
                    for s in signatures
                )
                and all(renamer.rename_record(key=r, mapping=mapping) in records for r in records)
                and all(renamer.rename_pool(key=p, mapping=mapping) in pools for p in pools)
                and renamer.rename_atoms(atoms=invariant_atoms, mapping=mapping) == invariant_atoms
            )

        by_type: dict[str, list[Object]] = {}
        for obj in sorted({o for skill in applicable for o in skill.objects}, key=lambda o: o.name):
            by_type.setdefault(obj.type.name, []).append(obj)
        candidates: list[list[Object]] = []
        for members in by_type.values():
            found: list[list[Object]] = []
            for obj in members:
                for group in found:
                    if is_automorphism(first=group[0], second=obj):
                        group.append(obj)
                        break
                else:
                    found.append([obj])
            candidates.extend(group for group in found if len(group) > 1)

        dynamic_atoms = [
            *(atom for s in signatures for atom in renamer.signature_atoms(signature=s)),
            *(atom for r in records for atom in renamer.record_atoms(key=r)),
            *(atom for p in pools for atom in p[0] if atom.predicate.name in dynamic_predicates),
            *(atom for atom in invariant_atoms if atom.predicate.name in dynamic_predicates),
        ]
        classes: list[tuple[Object, ...]] = []
        claimed: set[Object] = set()
        for group in candidates:
            union = claimed | set(group)
            if any(sum(o in union for o in atom.objects) > 1 for atom in dynamic_atoms):
                continue
            classes.append(tuple(group))
            claimed = union
        return none.model_copy(update={"classes": tuple(classes)})

    def _profiles(self, *, atoms: frozenset[GroundAtom]) -> dict[Object, tuple[object, ...]]:
        members = {o for group in self.classes for o in group}
        found: dict[Object, list[tuple[str, tuple[str, ...]]]] = {o: [] for o in members}
        for atom in atoms:
            if atom.predicate.name in self.static_predicates:
                continue
            for obj in atom.objects:
                if obj in found:
                    found[obj].append((
                        atom.predicate.name,
                        tuple("*" if o == obj else o.name for o in atom.objects),
                    ))
        return {o: tuple(sorted(entries)) for o, entries in found.items()}

    def canonical_atoms(self, *, atoms: frozenset[GroundAtom]) -> frozenset[GroundAtom]:
        """The orbit's representative: each class's members renamed in profile order."""
        if not self.classes:
            return atoms
        profiles = self._profiles(atoms=atoms)
        mapping: Mapping = {}
        for group in self.classes:
            ordered = sorted(group, key=lambda o: (profiles[o], o.name))
            mapping.update(zip(ordered, group, strict=True))
        return frozenset(
            atom
            if atom.predicate.name in self.static_predicates
            else _Renamer.rename_atom(atom=atom, mapping=mapping)
            for atom in atoms
        )

    def _orbit_key(
        self, *, profiles: dict[Object, tuple[object, ...]], action: GroundSkill
    ) -> tuple[object, ...]:
        first_seen: dict[Object, int] = {}
        arguments: list[object] = []
        for obj in action.objects:
            if obj in profiles:
                first_seen.setdefault(obj, len(first_seen))
                arguments.append(("class", profiles[obj], first_seen[obj]))
            else:
                arguments.append(("object", obj.name))
        return (action.skill.name, tuple(arguments))

    def representatives(
        self, *, atoms: frozenset[GroundAtom], actions: list[GroundSkill]
    ) -> list[GroundSkill]:
        """The first action of each stabiliser orbit, in the input order."""
        if not self.classes:
            return actions
        profiles = self._profiles(atoms=atoms)
        seen: set[tuple[object, ...]] = set()
        kept: list[GroundSkill] = []
        for action in actions:
            key = self._orbit_key(profiles=profiles, action=action)
            if key not in seen:
                seen.add(key)
                kept.append(action)
        return kept

    def equivalent(self, *, atoms: frozenset[GroundAtom], first: object, second: object) -> bool:
        """Whether two actions lie in one stabiliser orbit of `atoms`."""
        if not isinstance(first, GroundSkill) or not isinstance(second, GroundSkill):
            return first == second
        profiles = self._profiles(atoms=atoms)
        return self._orbit_key(profiles=profiles, action=first) == self._orbit_key(
            profiles=profiles, action=second
        )


class _Renamer(BaseModel):
    """Object renamings of ground skills, failure records and starved pools, applied
    to dynamic atoms only. A ground skill is represented by its signature -- what the
    search reads of it -- so a chain-pinned skill renames to itself."""

    model_config = ConfigDict(frozen=True)

    dynamic_predicates: frozenset[str]
    cost_of: Callable[[GroundSkill], float]

    @staticmethod
    def rename_atom(*, atom: GroundAtom, mapping: Mapping) -> GroundAtom:
        if not any(o in mapping for o in atom.objects):
            return atom
        return GroundAtom(
            predicate=atom.predicate, objects=tuple(mapping.get(o, o) for o in atom.objects)
        )

    def rename_atoms(
        self, *, atoms: Iterable[GroundAtom], mapping: Mapping
    ) -> frozenset[GroundAtom]:
        return frozenset(
            self.rename_atom(atom=atom, mapping=mapping)
            if atom.predicate.name in self.dynamic_predicates
            else atom
            for atom in atoms
        )

    def signature(self, *, skill: GroundSkill) -> Signature:
        return (
            skill.skill.name,
            self.cost_of(skill),
            frozenset(
                a for a in skill.preconditions if a.predicate.name in self.dynamic_predicates
            ),
            frozenset(skill.add_effects),
            frozenset(skill.delete_effects),
            frozenset(
                (frozenset(effect.conditions), frozenset(effect.add_effects))
                for effect in skill.conditional_add_effects
            ),
            frozenset(p.name for p in skill.ignore_effects),
        )

    def rename_signature(self, *, signature: Signature, mapping: Mapping) -> Signature:
        name, cost, pre, add, delete, conditional, ignore = signature
        return (
            name,
            cost,
            self.rename_atoms(atoms=pre, mapping=mapping),  # type: ignore[arg-type]
            self.rename_atoms(atoms=add, mapping=mapping),  # type: ignore[arg-type]
            self.rename_atoms(atoms=delete, mapping=mapping),  # type: ignore[arg-type]
            frozenset(
                (
                    self.rename_atoms(atoms=conditions, mapping=mapping),
                    self.rename_atoms(atoms=adds, mapping=mapping),
                )
                for conditions, adds in conditional  # type: ignore[attr-defined]
            ),
            ignore,
        )

    @staticmethod
    def signature_atoms(*, signature: Signature) -> Iterable[GroundAtom]:
        _, _, pre, add, delete, conditional, _ = signature
        yield from pre  # type: ignore[misc]
        yield from add  # type: ignore[misc]
        yield from delete  # type: ignore[misc]
        for conditions, adds in conditional:  # type: ignore[attr-defined]
            yield from conditions
            yield from adds

    def record(self, *, record: FailureEffectCount, mapping: Mapping) -> Signature:
        return self.rename_record(
            key=(
                self.signature(skill=record.ground_skill),
                record.before_atoms,
                record.was_random_exploration,
                record.add_effects,
                record.delete_effects,
                record.count,
            ),
            mapping=mapping,
        )

    def rename_record(self, *, key: Signature, mapping: Mapping) -> Signature:
        signature, before, was_random, add, delete, count = key
        return (
            self.rename_signature(signature=signature, mapping=mapping),  # type: ignore[arg-type]
            self.rename_atoms(atoms=before, mapping=mapping),  # type: ignore[arg-type]
            was_random,
            self.rename_atoms(atoms=add, mapping=mapping),  # type: ignore[arg-type]
            self.rename_atoms(atoms=delete, mapping=mapping),  # type: ignore[arg-type]
            count,
        )

    def record_atoms(self, *, key: Signature) -> Iterable[GroundAtom]:
        signature, before, _, add, delete, _ = key
        yield from self.signature_atoms(signature=signature)  # type: ignore[arg-type]
        for atoms in (before, add, delete):
            yield from (
                a
                for a in atoms  # type: ignore[attr-defined]
                if a.predicate.name in self.dynamic_predicates
            )

    def pool(
        self, *, pool: tuple[frozenset[GroundAtom], GroundSkill], mapping: Mapping
    ) -> tuple[frozenset[GroundAtom], Signature]:
        atoms, skill = pool
        return self.rename_pool(key=(atoms, self.signature(skill=skill)), mapping=mapping)

    def rename_pool(
        self, *, key: tuple[frozenset[GroundAtom], Signature], mapping: Mapping
    ) -> tuple[frozenset[GroundAtom], Signature]:
        atoms, signature = key
        return (
            self.rename_atoms(atoms=atoms, mapping=mapping),
            self.rename_signature(signature=signature, mapping=mapping),
        )
