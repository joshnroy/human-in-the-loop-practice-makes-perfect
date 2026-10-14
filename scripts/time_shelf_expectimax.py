"""Time one SweepOffShelf3D expectimax decision at one depth from one symbolic state.

Builds the belief-space model as `Tossing3DPomdpMethod.model_post_init` does for the
shelf (Model B `local_trend`, grid 25x16, the CLI's process-noise defaults, linear cost
lambda 3e-6, observation weight 0.1, `use_model_j=True`), with or without the exact
search reductions, and prints one JSON line.

    scripts/with_env.sh python -m scripts.time_shelf_expectimax \\
        --state start --depth 7 --reduction full

`--pinned-cube` adds one real failure record -- a failed pick of cube_0 at the start
state that left the hand as it was -- which splits cube_0 off the class, as the
first observed cube-specific failure of a run would.
"""

import argparse
import json
import resource
import time
from collections.abc import Callable, Iterable
from typing import Any

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    SweepOffShelf3DAtoms,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.practice_domain import (
    SWEEP_OFF_SHELF_DOMAIN,
    PracticeDomain,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    make_tossing3d_search_state,
)
from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace
from hitl_pmp.planning.grounding import SkillGrounder

LINEAR_COST_LAMBDA = 3e-6
OBSERVATION_WEIGHT = 0.1


class ShelfTiming:
    """A static-method container."""

    STATES: dict[str, dict[str, Any]] = {
        "start": {"shelf": range(5)},
        # Just swept: wiper in hand, every cube in the box.
        "swept": {"box": range(5), "wiper": "held"},
        # Wiper on the floor; two cubes on the floor, two in the box, one on the shelf.
        "scattered": {"floor": (0, 1), "box": (2, 3), "shelf": (4,), "wiper": "floor"},
    }

    @staticmethod
    def atoms(
        *,
        shelf: Iterable[int] = (),
        box: Iterable[int] = (),
        floor: Iterable[int] = (),
        wiper: str = "home",
    ) -> frozenset[GroundAtom]:
        cubes = O.cubes
        named: set[tuple[str, tuple[str, ...]]] = set()
        named |= {("OnShelf", (cubes[i].name,)) for i in shelf}
        named |= {("InBox", (cubes[i].name, O.box.name)) for i in box}
        named |= {("OnFloor", (cubes[i].name,)) for i in floor}
        if wiper != "held":
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
            | SweepOffShelf3DPredicates.static_atoms()
        )

    # Overridable, so a harness can time a variant skill set on the same states.
    skills: Callable[[], tuple[Skill, ...]] = staticmethod(
        lambda: (*S.robot_skills(), *S.human_skills())
    )
    domain: Callable[[], PracticeDomain] = staticmethod(lambda: SWEEP_OFF_SHELF_DOMAIN)

    @staticmethod
    def ground_skills() -> tuple[GroundSkill, ...]:
        objects = O.all_objects()
        return tuple(
            SkillGrounder.applicable_ground_skills(
                skills=ShelfTiming.skills(),
                objects=objects,
                true_atoms=SkillGrounder.all_possible_ground_atoms(
                    objects=objects, predicates=SweepOffShelf3DPredicates.all()
                ),
            )
        )

    @staticmethod
    def pinning_record(*, ground_skills: tuple[GroundSkill, ...]) -> FailureEffectCount:
        pick = next(
            g
            for g in ground_skills
            if g.skill.name == "PickCubeFromShelf" and O.cubes[0] in g.objects
        )
        return FailureEffectCount(
            ground_skill=pick,
            before_atoms=ShelfTiming.atoms(**ShelfTiming.STATES["start"]),
            was_random_exploration=False,
            add_effects=frozenset(),
            delete_effects=frozenset(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=sorted(ShelfTiming.STATES), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument(
        "--reduction", choices=("none", "symmetry", "counts", "full"), required=True
    )
    parser.add_argument("--pinned-cube", action="store_true")
    args = parser.parse_args()

    built_at = time.perf_counter()
    ground_skills = ShelfTiming.ground_skills()
    model = Tossing3DPracticeModel(
        seed=0,
        domain=ShelfTiming.domain(),
        exploration_epsilon=0.5,
        ground_skills=ground_skills,
        linear_cost_lambda=LINEAR_COST_LAMBDA,
        object_symmetry=args.reduction in {"symmetry", "full"},
        merge_beliefs_by_counts=args.reduction in {"counts", "full"},
        failure_effect_counts=(
            (ShelfTiming.pinning_record(ground_skills=ground_skills),) if args.pinned_cube else ()
        ),
    )
    belief = make_default_tossing3d_belief(
        num_particles=1024,
        seed=0,
        model="local_trend",
        engine="grid",
        inference_config=InferenceConfig(
            competence_bins=25,
            learning_rate_bins=16,
            sigma_competence=0.03,
            sigma_eta=0.005,
            learning_rate_decay=0.9,
            eta_max=0.15,
        ),
        additional_skill_names=tuple(skill.name for skill in S.human_skills()),
        domain=ShelfTiming.domain(),
    )
    environment_state = make_tossing3d_search_state(
        state=belief, true_atoms=ShelfTiming.atoms(**ShelfTiming.STATES[args.state])
    )
    classes = [
        [o.name for o in group]
        for group in model.object_symmetry_for(environment_state=environment_state).classes
    ]
    setup_seconds = time.perf_counter() - built_at

    trace = SearchTrace()
    planner = ExpectimaxPlanner(use_model_j=True, observation_probability_weight=OBSERVATION_WEIGHT)
    started = time.perf_counter()
    value, action = planner.solve(
        environment_state=environment_state,
        summed_cost=belief.accumulated_cost,
        belief_state=belief,
        horizon=args.depth,
        model=model,
        trace=trace,
        num_samples=100,
    )
    wall = time.perf_counter() - started
    trace.close()
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    print(
        json.dumps({
            "state": args.state,
            "depth": args.depth,
            "reduction": args.reduction,
            "pinned_cube": args.pinned_cube,
            "classes": classes,
            "wall_seconds": wall,
            "setup_seconds": setup_seconds,
            "value": value,
            "action": "STOP"
            if not isinstance(action, GroundSkill)
            else f"{action.skill.name}({','.join(o.name for o in action.objects)})",
            "root_action_values": {
                event["action"]["skill"]["name"]
                + "("
                + ",".join(o["name"] for o in event["action"]["objects"] if "cube" in o["name"])
                + ")": event["value"]
                for event in trace.events
                if event["event"] == "action_value" and event["node"] == 0
            },
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **{
                key: summary[key]
                for key in ("expanded_nodes", "unique_nodes", "cache_requests", "cache_hits")
            },
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
