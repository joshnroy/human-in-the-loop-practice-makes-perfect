from typing import Any

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom


class ObservationBuilder:
    """Turns one decision point into the plain-JSON dict an authored policy is called
    with. A static-method container, never instantiated, same as every other
    business-logic class in this project.

    **Plain builtins only -- no `State`, no `Object`, no numpy.** The authored file may
    import nothing but the standard library and numpy, and it must be readable by an
    agent that has never seen this codebase, so everything crossing that boundary is a
    dict, a list, a string or a float. That also makes the observation trivially
    loggable: a decision that went wrong can be reproduced from the transcript alone.

    **Everything is sorted or in provider order**, never in a set's iteration order. Two
    processes must build the same observation from the same state, or an authored policy
    that breaks a tie by position becomes nondeterministic and the whole record-then-replay
    design collapses. `atoms` and `goal` come out of `frozenset`s and are sorted by their
    rendered string; `objects` is sorted by name; `skills` keeps the order
    `SkillGrounder.applicable_ground_skills` produced, which is itself a deterministic
    function of the provider's own fixed skill order."""

    @staticmethod
    def build(
        *,
        state: State,
        goal: Goal,
        atoms: frozenset[GroundAtom],
        objects: tuple[Object, ...],
        ground_skills: list[GroundSkill],
    ) -> dict[str, Any]:
        return {
            "goal": ObservationBuilder.render_atoms(atoms=goal.atoms),
            "objects": [
                {
                    "name": obj.name,
                    "type": obj.type.name,
                    "features": {
                        feature_name: float(state.get(obj=obj, feature_name=feature_name))
                        for feature_name in obj.type.feature_names
                    },
                }
                for obj in sorted(objects, key=lambda obj: obj.name)
            ],
            "atoms": ObservationBuilder.render_atoms(atoms=atoms),
            "skills": [
                {
                    "index": index,
                    "name": ground_skill.skill.name,
                    "objects": [obj.name for obj in ground_skill.objects],
                    "param_dim": ground_skill.skill.param_dim,
                }
                for index, ground_skill in enumerate(ground_skills)
            ],
        }

    @staticmethod
    def render_atoms(*, atoms: frozenset[GroundAtom]) -> list[str]:
        """`Pred(a, b)` strings, sorted -- the same rendering `Goal.describe` uses, so a
        goal atom and the identical atom appearing in `atoms` are the same string and an
        authored policy can compare them directly."""
        return sorted(
            f"{atom.predicate.name}({', '.join(obj.name for obj in atom.objects)})"
            for atom in atoms
        )
