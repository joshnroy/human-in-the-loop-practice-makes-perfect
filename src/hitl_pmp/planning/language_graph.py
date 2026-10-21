"""Compile learned state graphs for the shared classical deployment planner."""

import math
from collections.abc import Mapping, Sequence

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.types import TranslationCache


def _not_a_classifier(state, objects):  # noqa: PLR0917 -- Predicate callback protocol
    raise RuntimeError("Graph predicates must come from the external state classifier")


class LanguageGraphPlanner:
    """Use Fast Downward's existing seq-opt-lmcut and -log success objective.

    Success branches are determinized into separate operators. Their probability
    is frozen skill competence times the declared conditional destination mass.
    This is maximum-likelihood path planning, not expected physical-cost planning
    or the practice expectimax algorithm. Replanning uses each observed region.
    """

    def __init__(
        self,
        *,
        clusters: Sequence[str],
        goals: Sequence[str],
        edges: Sequence[Mapping],
        competences: Mapping[str, float],
        human_actions: frozenset[str],
    ):
        self.goals = frozenset(goals)
        self.predicates = {
            name: Predicate(name=f"region_{i}", types=(), holds=_not_a_classifier)
            for i, name in enumerate(clusters)
        }
        self.goal = Predicate(name="graph_goal", types=(), holds=_not_a_classifier)
        self.skills = []
        self.costs = {}
        self.identifiers = {}
        self.cache = TranslationCache()
        self.plans = {}
        for edge in edges:
            action = edge["action"]
            if action in human_actions:
                continue
            competence = competences[action]
            if not math.isfinite(competence) or not 0 <= competence <= 1:
                raise ValueError(f"Invalid frozen competence for {action}: {competence}")
            destinations = edge["success"]
            if isinstance(destinations, str):
                destinations = {destinations: 1.0}
            for target, conditional in destinations.items():
                if not math.isfinite(conditional) or not 0 < conditional <= 1:
                    raise ValueError(f"Invalid success destination mass: {conditional}")
                source = edge["source"]
                pre = LiftedAtom(predicate=self.predicates[source], variables=())
                effects = {LiftedAtom(predicate=self.predicates[target], variables=())}
                if target in self.goals:
                    effects.add(LiftedAtom(predicate=self.goal, variables=()))
                skill = Skill(
                    name=f"edge_{len(self.skills):04d}",
                    parameters=(),
                    preconditions=frozenset({pre}),
                    add_effects=frozenset(effects),
                    delete_effects=frozenset() if source == target else frozenset({pre}),
                    param_dim=0,
                )
                ground = GroundSkill(skill=skill, objects=())
                self.skills.append(skill)
                self.costs[ground] = -math.log(max(competence * conditional, 1e-12))
                self.identifiers[skill.name] = action

    def plan(self, *, cluster: str | None) -> tuple[str, ...]:
        if cluster is None or cluster in self.goals:
            return ()
        if cluster not in self.predicates:
            raise ValueError(f"Unknown language region: {cluster}")
        if cluster not in self.plans:
            try:
                plan = FastDownwardPlanner.plan(
                    skills=tuple(self.skills),
                    predicates=(*self.predicates.values(), self.goal),
                    types=(),
                    objects=(),
                    init_atoms=frozenset({
                        GroundAtom(predicate=self.predicates[cluster], objects=())
                    }),
                    goal=frozenset({GroundAtom(predicate=self.goal, objects=())}),
                    ground_skill_costs=self.costs,
                    translation_cache=self.cache,
                )
            except PlanningFailure:
                self.plans[cluster] = ()
            else:
                self.plans[cluster] = tuple(self.identifiers[g.skill.name] for g in plan)
        return self.plans[cluster]

    def action(self, *, cluster: str | None) -> str | None:
        plan = self.plan(cluster=cluster)
        return plan[0] if plan else None
