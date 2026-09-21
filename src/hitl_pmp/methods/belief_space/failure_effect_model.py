"""Online empirical symbolic effects conditional on a failed execution."""

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .types.failure_effects import FailureEffectCount


class EmpiricalFailureEffects:
    """Maximum-likelihood effects within an exact precontext and sampler mode.

    Unseen contexts retain the legacy identity approximation, which can remain
    optimistic until a failure is observed in that context. Exact contexts avoid
    transferring a release or reachability change into an incompatible state.
    Counts come only from real failed practice executions and remain fixed
    throughout each search; hypothetical observations never train this model.
    """

    @staticmethod
    def observe(
        *,
        counts: tuple[FailureEffectCount, ...],
        ground_skill: GroundSkill,
        before_atoms: frozenset[GroundAtom],
        after_atoms: frozenset[GroundAtom],
        was_random_exploration: bool,
    ) -> tuple[FailureEffectCount, ...]:
        observation = FailureEffectCount(
            ground_skill=ground_skill,
            before_atoms=before_atoms,
            was_random_exploration=was_random_exploration,
            add_effects=after_atoms - before_atoms,
            delete_effects=before_atoms - after_atoms,
        )
        for index, existing in enumerate(counts):
            if existing.model_copy(update={"count": 1}) == observation:
                return (
                    *counts[:index],
                    existing.model_copy(update={"count": existing.count + 1}),
                    *counts[index + 1 :],
                )
        return (*counts, observation)

    @staticmethod
    def outcomes(
        *,
        counts: tuple[FailureEffectCount, ...],
        ground_skill: GroundSkill,
        true_atoms: frozenset[GroundAtom],
        was_random_exploration: bool,
    ) -> tuple[tuple[float, frozenset[GroundAtom]], ...]:
        matching = tuple(
            record
            for record in counts
            if record.ground_skill == ground_skill
            and record.before_atoms == true_atoms
            and record.was_random_exploration == was_random_exploration
        )
        total = sum(record.count for record in matching)
        if total == 0:
            return ((1.0, true_atoms),)
        return tuple(
            (
                record.count / total,
                (true_atoms - record.delete_effects) | record.add_effects,
            )
            for record in matching
        )

    @staticmethod
    def diagnostics(*, counts: tuple[FailureEffectCount, ...]) -> list[dict[str, object]]:
        return [
            {
                "skill": record.ground_skill.skill.name,
                "objects": [obj.name for obj in record.ground_skill.objects],
                "before_atoms": sorted(str(atom) for atom in record.before_atoms),
                "after_atoms": sorted(
                    str(atom)
                    for atom in (record.before_atoms - record.delete_effects) | record.add_effects
                ),
                "was_random_exploration": record.was_random_exploration,
                "add_effects": sorted(str(atom) for atom in record.add_effects),
                "delete_effects": sorted(str(atom) for atom in record.delete_effects),
                "count": record.count,
                "context_total": sum(
                    other.count
                    for other in counts
                    if other.ground_skill == record.ground_skill
                    and other.before_atoms == record.before_atoms
                    and other.was_random_exploration == record.was_random_exploration
                ),
            }
            for record in counts
        ]
