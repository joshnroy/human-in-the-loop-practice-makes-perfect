"""Experimental exact stock-only continuation bound, not wired into search."""

from pydantic import Field

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .sweep_expectimax import SweepExpectimaxPlanner
from .sweep_model import SweepPracticeModel
from .tossing3d_transition_model import make_tossing3d_search_state
from .types.belief_state import Tossing3DBeliefState
from .types.search_state import Tossing3DSearchState


class SweepRelaxedStockBound:
    """Solve the same stock learning problem with free, unconstrained access.

    Non-stock outcomes update only independent non-stock beliefs. Delete them
    from any actual policy, allow its stock actions at every state and remove
    nonnegative future charges/surprise. This embeds its stock-observation tree
    into this relaxation, with no smaller value. All stock inference, attempt
    counts, sampler evidence/modes and the deployment evaluator remain intact.
    """

    def __init__(self, *, model: SweepPracticeModel) -> None:
        if type(model) is not SweepPracticeModel:
            raise TypeError("bound proof requires the unmodified SweepPracticeModel contract")
        self.source = model
        self.names = frozenset(skill.skill.name for skill in model.deployment.ordered_skills)
        if self.names & set(model.human_skill_names):
            raise ValueError("deployment and human skill names must be disjoint")
        skills = tuple(skill for skill in model.ground_skills if skill.skill.name in self.names)
        settings = {name: getattr(model, name) for name in type(model).model_fields}
        settings.update(
            ground_skills=skills,
            trainable_skill_names=tuple(
                name for name in model.trainable_skill_names if name in self.names
            ),
            human_skill_names=(),
            linear_cost_lambda=0.0,
            starved_pools=(),
            relaxed_atoms=frozenset(atom for skill in skills for atom in skill.preconditions),
        )
        self.relaxed = _RelaxedStockModel(**settings)
        self.cache: dict[object, float] = {}
        self.last_planner: SweepExpectimaxPlanner | None = None

    def value(self, *, state: Tossing3DBeliefState, horizon: int) -> float:
        projected = state.model_copy(
            update={
                "skill_beliefs": {
                    name: belief
                    for name, belief in state.skill_beliefs.items()
                    if name in self.names
                },
                "pending_examples": {
                    name: count
                    for name, count in state.pending_examples.items()
                    if name in self.names
                },
                "sampler_training": {
                    name: training
                    for name, training in state.sampler_training.items()
                    if name in self.names
                },
                "accumulated_cost": 0.0,
            }
        )
        key = (self.relaxed._state_key(state=projected), horizon)
        value = self.cache.get(key)
        if value is None:
            self.last_planner = SweepExpectimaxPlanner(
                use_model_j=True, observation_probability_weight=0.0
            )
            value, _ = self.last_planner.solve(
                environment_state=make_tossing3d_search_state(
                    state=projected, true_atoms=self.relaxed.relaxed_atoms
                ),
                belief_state=projected,
                summed_cost=0,
                horizon=horizon,
                model=self.relaxed,
                num_samples=1,
            )
            self.cache[key] = value
        return self.source.G(policy_value=value, summed_cost=state.accumulated_cost)


class _RelaxedStockModel(SweepPracticeModel):
    relaxed_atoms: frozenset[GroundAtom] = Field(exclude=True)

    def get_valid_actions(self, *, environment_state: Tossing3DSearchState) -> list[GroundSkill]:
        return list(self.ground_skills)

    def outcomes(
        self,
        *,
        environment_state: Tossing3DSearchState,
        state: Tossing3DBeliefState,
        action: GroundSkill,
    ) -> tuple[tuple[float, Tossing3DBeliefState, frozenset[GroundAtom]], ...]:
        synthetic = environment_state.model_copy(update={"true_atoms": self.relaxed_atoms})
        # Symbolic failure subbranches carry no further stock evidence. Merging
        # them below preserves their total probability and drops only geometry.
        return tuple(
            (probability, successor, self.relaxed_atoms)
            for probability, successor, _ in super().outcomes(
                environment_state=synthetic, state=state, action=action
            )
        )
