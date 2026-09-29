"""Exact Sweep expectimax with provably unproductive recovery suffixes removed."""

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .expectimax import ExpectimaxPlanner
from .sweep_model import SweepPracticeModel
from .types.belief_state import Tossing3DBeliefState
from .types.protocol import BeliefSpaceModel
from .types.search_state import Tossing3DSearchState
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, StopAction
from .types.sweep_theta import SweepTheta


class SweepExpectimaxPlanner(
    ExpectimaxPlanner[Tossing3DSearchState, Tossing3DBeliefState, SweepTheta, GroundSkill]
):
    """Preserve expectimax; collapse suffixes unable to affect deployment value.

    Delete relaxation is only an impossibility certificate. It ignores failures'
    contexts, conditional-effect conditions, action masking and all deletes, so
    it can overestimate reachability but never remove a truly reachable skill.
    Without a deployment skill attempt, every stock posterior and pending count
    stays unchanged. Nonnegative execution costs and surprise penalties make
    STOP at least as good as every such continuation.
    """

    def solve(
        self,
        *,
        environment_state: Tossing3DSearchState,
        summed_cost: float,
        belief_state: Tossing3DBeliefState,
        horizon: int,
        model: BeliefSpaceModel[
            Tossing3DSearchState, Tossing3DBeliefState, SweepTheta, GroundSkill
        ],
        num_samples: int = NUM_SAMPLES,
        trace: SearchTrace | None = None,
    ) -> tuple[float, GroundSkill | StopAction]:
        self.pruned_recovery_suffixes = 0
        self._reachability: dict[tuple[frozenset[GroundAtom], int], bool] = {}
        self._relaxed: list[tuple[frozenset[GroundAtom], frozenset[GroundAtom], bool]] = []
        self._can_prune = isinstance(model, SweepPracticeModel) and self.use_model_j
        if isinstance(model, SweepPracticeModel):
            relevant_names = {skill.skill.name for skill in model.deployment.ordered_skills}
            for skill in model.ground_skills:
                additions = (
                    skill.add_effects
                    | frozenset(
                        atom
                        for effect in skill.conditional_add_effects
                        for atom in effect.add_effects
                    )
                    | frozenset(
                        atom
                        for record in model.failure_effect_counts
                        if record.ground_skill == skill
                        for atom in record.add_effects
                    )
                )
                self._relaxed.append((
                    skill.preconditions,
                    additions,
                    skill.skill.name in relevant_names,
                ))
        result = super().solve(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
            model=model,
            num_samples=num_samples,
            trace=trace,
        )
        if trace is not None:
            trace.record(
                event="exact_recovery_suffix_pruning",
                node=0,
                pruned_suffixes=self.pruned_recovery_suffixes,
                requested_horizon=horizon,
            )
        return result

    def _deployment_attempt_reachable(self, *, atoms: frozenset[GroundAtom], horizon: int) -> bool:
        key = (atoms, horizon)
        cached = self._reachability.get(key)
        if cached is not None:
            return cached
        relaxed_atoms = atoms
        for _ in range(horizon):
            additions: set[GroundAtom] = set()
            for preconditions, effects, relevant in self._relaxed:
                if preconditions <= relaxed_atoms:
                    if relevant:
                        self._reachability[key] = True
                        return True
                    additions.update(effects)
            expanded = relaxed_atoms | additions
            if expanded == relaxed_atoms:
                break
            relaxed_atoms = expanded
        self._reachability[key] = False
        return False

    def cached_solve_belief_space_expectimax(
        self,
        *,
        environment_state: Tossing3DSearchState,
        summed_cost: float,
        belief_state: Tossing3DBeliefState,
        horizon: int,
    ) -> tuple[float, GroundSkill | StopAction]:
        # Leave the root intact to preserve its full action-value diagnostics.
        if (
            self._can_prune
            and self.next_node > 0
            and horizon > 0
            and not self._deployment_attempt_reachable(
                atoms=environment_state.true_atoms, horizon=horizon
            )
        ):
            self.pruned_recovery_suffixes += 1
            horizon = 0
        return super().cached_solve_belief_space_expectimax(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
