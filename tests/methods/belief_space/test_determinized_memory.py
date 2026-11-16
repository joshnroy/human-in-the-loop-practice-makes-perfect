"""What the determinized search keeps alive while it runs.

A Two Routes search of 8,965 nodes once grew by ~2.5 GiB, most of it the posterior of
every child it had generated, kept for a reopening that never came. A child's belief
must live only as long as something can still read it: its open-list entry."""

import math
import weakref

import pytest
from pydantic import BaseModel, ConfigDict

from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner

LIVE: "weakref.WeakSet[Belief]" = weakref.WeakSet()


class Belief(BaseModel):
    model_config = ConfigDict(frozen=True)
    depth: int
    dead: bool


class Env(BaseModel):
    model_config = ConfigDict(frozen=True)
    depth: int
    dead: bool
    # As in the Tossing3D search state, the belief rides inside the environment state.
    state: Belief


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str


STEP = Action(name="step")
SIDE = Action(name="side")


class CorridorModel(BaseModel):
    """A corridor whose every node also offers a side step into a dead end. Each
    transition builds fresh beliefs, so the live ones are what the search holds."""

    max_live: int = 0

    @staticmethod
    def env(*, depth: int, dead: bool) -> Env:
        belief = Belief(depth=depth, dead=dead)
        LIVE.add(belief)
        return Env(depth=depth, dead=dead, state=belief)

    def J(self, *, belief_state: Belief, summed_cost: float, num_samples: int) -> float:
        del num_samples
        self.max_live = max(self.max_live, len(LIVE))
        if belief_state.dead:
            return -math.inf
        return belief_state.depth - summed_cost

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        return policy_value - summed_cost

    def get_valid_actions(self, *, environment_state: Env) -> list[Action]:
        return [] if environment_state.dead else [STEP, SIDE]

    def transition_outcomes(
        self, *, environment_state: Env, practice_action: Action, belief_state: Belief
    ) -> list[tuple[Env, float, float]]:
        del belief_state
        depth = environment_state.depth + 1
        return [(self.env(depth=depth, dead=practice_action == SIDE), 0.0, 1.0)]

    def compute_next_belief_state(
        self,
        *,
        belief_state: Belief,
        environment_state: Env,
        potential_next_environment_state: Env,
        practice_action: Action,
    ) -> Belief:
        del belief_state, environment_state, practice_action
        return potential_next_environment_state.state

    def search_cache_key(
        self,
        *,
        environment_state: Env,
        summed_cost: float,
        belief_state: Belief,
        horizon: int | None,
    ) -> object:
        del belief_state
        return environment_state.depth, environment_state.dead, summed_cost, horizon

    def belief_cache_key(
        self, *, environment_state: Env, belief_state: Belief, horizon: int | None
    ) -> object:
        del belief_state
        return environment_state.depth, environment_state.dead, horizon


@pytest.mark.parametrize("variant", ["algorithm3", "curtis"])
def test_a_long_search_keeps_no_belief_per_generated_child(*, variant: str) -> None:
    """The open list never holds more than one node here, while 2 children are
    generated per expansion; so the beliefs alive stay a handful however deep it goes."""
    nodes = 60
    model = CorridorModel()
    root = CorridorModel.env(depth=0, dead=False)
    value, action = DeterminizedAStarPlanner(
        max_iterations=nodes, seed=0, observation_probability_weight=0.0, variant=variant
    ).solve(
        environment_state=root,
        summed_cost=0.0,
        belief_state=root.state,
        model=model,  # type: ignore[arg-type]
        num_samples=1,
    )
    assert action == STEP
    assert value == nodes
    assert model.max_live <= 6, f"{model.max_live} beliefs alive during a {nodes}-node search"
