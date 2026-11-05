"""Record `data/recorded_searches.json` for `test_search_implementation_caches.py`.

Run against the code the searches should be pinned to (it was run on main @ 746f77732,
before the implementation caches existed):

    scripts/with_env.sh python -m tests.methods.belief_space.record_search_fixture
"""

import hashlib
import json
from pathlib import Path

from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.practice_domain import SWEEP_OFF_SHELF_DOMAIN
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.particle_filter_belief import ParticleFilterBelief

from . import test_object_symmetry as shelf

FIXTURE = Path(__file__).parent / "data" / "recorded_searches.json"
CASES = (("start", False), ("start", True), ("scattered", True))
DEPTH = 3


class RecordedSearch:
    """One recorded search, solved and summarised the same way by recorder and test."""

    @staticmethod
    def name(*, state: str, reduce: bool) -> str:
        return f"{state}-{'reduced' if reduce else 'full'}"

    @staticmethod
    def action_text(*, action: object) -> str:
        skill = getattr(action, "skill", None)
        if skill is None:
            return str(action)
        return f"{skill.name}({','.join(o.name for o in action.objects)})"  # type: ignore[attr-defined]

    @staticmethod
    def solve(*, state: str, reduce: bool) -> dict[str, object]:
        model = Tossing3DPracticeModel(
            ground_skills=shelf._ground_skills(),
            domain=SWEEP_OFF_SHELF_DOMAIN,
            linear_cost_lambda=shelf.LINEAR_COST_LAMBDA,
            object_symmetry=reduce,
            merge_beliefs_by_counts=reduce,
        )
        belief = shelf.BELIEFS["carry_trained"](
            SWEEP_OFF_SHELF_DOMAIN, config=shelf.PRODUCTION_GRID
        )
        # The cost lookups the searches were recorded with: a skill with a learned
        # belief priced at its posterior mean, every other skill at its old unit
        # practice cost. A forecast for every grounding would override the means.
        learned = (BayesianSkillBelief, ParticleFilterBelief)
        belief = belief.model_copy(
            update={
                "expected_execution_costs": {
                    str(ground): 1.0
                    for ground in shelf._ground_skills()
                    if not isinstance(belief.skill_beliefs.get(ground.skill.name), learned)
                }
            }
        )
        planner = ExpectimaxPlanner(use_model_j=True)
        value, action = planner.solve(
            environment_state=make_tossing3d_search_state(
                state=belief, true_atoms=shelf.STATES[state]
            ),
            belief_state=belief,
            summed_cost=0.0,
            horizon=DEPTH,
            model=model,
        )
        # Keyed by a hash of the memo key's text: the key holds only bytes, ints, floats
        # and strings, so its repr is the same in every process.
        return {
            "value": value.hex(),
            "action": RecordedSearch.action_text(action=action),
            "unique_nodes": planner.next_node,
            "cache_hits": planner.cache_hits,
            "memo": {
                hashlib.sha256(repr(key).encode()).hexdigest()[:20]: [
                    float(v).hex(),
                    RecordedSearch.action_text(action=a),
                ]
                for key, (v, a) in planner.memo.items()
            },
        }


if __name__ == "__main__":
    recorded = {
        RecordedSearch.name(state=state, reduce=reduce): RecordedSearch.solve(
            state=state, reduce=reduce
        )
        for state, reduce in CASES
    }
    FIXTURE.write_text(json.dumps(recorded, indent=0, sort_keys=True) + "\n")
    print(FIXTURE, {name: len(r["memo"]) for name, r in recorded.items()})
