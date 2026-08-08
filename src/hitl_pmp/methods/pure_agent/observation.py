import hashlib
import json
from typing import Any

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom


class ObservationBuilder:
    """Turns one decision point into the plain-JSON dict the agent is handed. A
    static-method container, never instantiated, same as every other business-logic class
    in this project.

    **Plain builtins only -- no `State`, no `Object`, no numpy.** Everything crossing the
    boundary into a prompt is a dict, a list, a string or a float, because it has to be
    readable by an agent that has never seen this codebase and it has to survive
    `json.dumps`. That also makes the observation trivially loggable: a decision that went
    wrong can be reproduced from the ledger alone.

    **Everything is sorted or in provider order**, never in a set's iteration order. Two
    processes must build the same observation from the same state, or the digest a replay
    checks itself against would differ between the recording run and the replay for
    reasons that have nothing to do with the decisions. `atoms` and `goal` come out of
    `frozenset`s and are sorted by their rendered string; `objects` is sorted by name;
    `skills` keeps the order `SkillGrounder.applicable_ground_skills` produced, which is
    itself a deterministic function of the provider's own fixed skill order."""

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
        goal atom and the identical atom appearing in `atoms` are the same string and the
        agent can compare them directly."""
        return sorted(
            f"{atom.predicate.name}({', '.join(obj.name for obj in atom.objects)})"
            for atom in atoms
        )

    @staticmethod
    def render(*, observation: dict[str, Any]) -> str:
        """The exact bytes that go into a prompt and into the digest.

        One function for both, deliberately: a replay verifies that it is being asked the
        same question the recording was asked, and it can only do that if the thing
        hashed is the thing shown. `sort_keys` because a dict literal's insertion order is
        a property of this file rather than of the decision point, and a refactor that
        reorders a key must not invalidate every recorded ledger."""
        return json.dumps(observation, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def digest(*, observation: dict[str, Any]) -> str:
        """SHA-256 over `render`, hex. Short enough to sit on every ledger line, exact
        enough that a replay knows immediately when the state sequence it is replaying
        into has diverged from the one that was recorded."""
        return hashlib.sha256(
            ObservationBuilder.render(observation=observation).encode("utf-8")
        ).hexdigest()
