"""Generated topology with the existing classical cost/competence forecasts."""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel

from .graph import StateGraph


class ClusterState(BaseModel):
    model_config = ConfigDict(frozen=True)
    cluster_id: str
    outcome: Literal["success", "failure"] | None = None


class OptionAction(BaseModel):
    model_config = ConfigDict(frozen=True)
    option_id: str


class ChainModel:
    def __init__(self, *, manifest, method):
        self.graph = StateGraph(manifest=manifest)
        self.base = method._pomdp_model
        self.bindings = {}
        human_bindings = method.skill_provider.human_cube_bin_reset_skills()
        human_names = {g.skill.name for g in human_bindings}
        for ground in self.base.ground_skills:
            if ground.skill.name in human_names and ground not in human_bindings:
                continue
            if ground in method.skill_provider.human_cube_bin_reset_skills():
                side = method.skill_provider.movables_reset_destination(ground_skill=ground)
                key = "reset_cube_and_bin_near" if side == "robot_side" else "reset_cube_far"
            else:
                key = ground.skill.name
            if key in self.bindings:
                # Generated robot skills consume the live object observation, not
                # the old symbolic side parameter. One accounting key per code slot.
                continue
            self.bindings[key] = ground
        if set(e.action for e in manifest.edges) - self.bindings.keys():
            raise ValueError("Generated graph contains an unbound skill")
        self.counts = {}

    def get_valid_actions(self, *, environment_state):
        return [
            OptionAction(option_id=a)
            for a in self.graph.actions(cluster=environment_state.cluster_id)
        ]

    def transition_outcomes(self, *, environment_state, practice_action, belief_state):
        action = practice_action.option_id
        ground = self.bindings[action]
        k = belief_state.skill_beliefs[ground.skill.name].mean_competence()
        cost = belief_state.expected_execution_costs[str(ground)]
        edge = self.graph.edges[environment_state.cluster_id, action]
        result = []
        for success, prob in ((True, k), (False, 1 - k)):
            key = environment_state.cluster_id, action, success
            counts = self.counts.get(key)
            destinations = (
                {x: n / counts.total() for x, n in counts.items()}
                if counts
                else edge.destinations(outcome=success)
            )
            for target, conditional in destinations.items():
                if prob * conditional > 0:
                    result.append((
                        ClusterState(
                            cluster_id=target, outcome="success" if success else "failure"
                        ),
                        cost,
                        prob * conditional,
                    ))
        return result

    def compute_next_belief_state(
        self, *, belief_state, environment_state, potential_next_environment_state, practice_action
    ):
        return self.observe(
            belief=belief_state,
            action=practice_action.option_id,
            success=potential_next_environment_state.outcome == "success",
        )

    def observe(self, *, belief, action, success):
        ground = self.bindings[action]
        result = self.base.observe_outcome(
            state=belief, ground_skill=ground, success=success, was_random_exploration=False
        )
        if ground.skill.name == TOSS_SKILL:
            result = self.base.observe_training_example(
                state=result, skill_name=TOSS_SKILL, success=success
            )
        return result

    def record(self, *, source, action, destination, success):
        self.counts.setdefault((source, action, success), Counter())[destination] += 1

    def search_cache_key(self, *, environment_state, summed_cost, belief_state, horizon):
        # Counts/topology are fixed within a solve. Hash posterior buffers instead
        # of retaining megabytes of serialized grids at every hypothetical node.
        return (
            environment_state.cluster_id,
            summed_cost,
            horizon,
            tuple(
                (name, Tossing3DPracticeModel._belief_signature(belief=b))
                for name, b in sorted(belief_state.skill_beliefs.items())
            ),
            tuple(sorted(belief_state.pending_examples.items())),
        )

    def J(self, **kwargs):
        return self.base.J(**kwargs)

    def G(self, **kwargs):
        return self.base.G(**kwargs)
