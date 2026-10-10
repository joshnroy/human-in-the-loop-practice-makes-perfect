"""LLCC: learned language states route an unchanged classical expectimax search."""

from typing import Any

from pydantic import Field, PrivateAttr

from hitl_pmp.core.method.method import HumanCubeBinResetRequested, InteractionComplete
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.hybrid_skills.method import HybridMethod, skill_action
from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace
from hitl_pmp.methods.belief_space.types.stop_action import StopAction

from .graph import HUMANS, LanguageManifest
from .model import ChainModel, ClusterState


class SkillsStatesMethod(HybridMethod):
    manifest: LanguageManifest | None = None
    classifier: Any = Field(default=None, exclude=True)
    classifier_configuration: dict = Field(default_factory=dict)
    needs_representation_repair: bool = False
    _chain: Any = PrivateAttr(default=None)
    _pending_language: Any = PrivateAttr(default=None)

    @property
    def measurement_ready(self):
        return self.manifest is not None and self.env.bundle is not None

    def accept_manifest(self, *, manifest):
        chain = ChainModel(manifest=manifest, method=self)
        self.manifest, self._chain = manifest, chain
        self.needs_representation_repair = False

    def classify(self):
        observation = Tossing3DAgenticBridge(
            env=self.env, observation_mode="object_state"
        ).observe()
        judgment = self.classifier.classify(manifest=self.manifest, observation=observation)
        self.record_diagnostic(
            event="classification", revision=self.manifest.revision, **judgment.model_dump()
        )
        if judgment.cluster_id is None:
            self.needs_representation_repair = True
        return judgment.cluster_id

    def settle(self, *, cluster):
        pending, self._pending_language = self._pending_language, None
        if pending is None:
            return
        source, action = pending
        success = self._chain.graph.outcome(source=source, action=action, destination=cluster)
        self.record_diagnostic(
            event="language_outcome",
            source=source,
            action=action,
            destination=cluster,
            success=success,
            revision=self.manifest.revision,
        )
        if success is None:
            return
        self._pomdp_state = self._chain.observe(
            belief=self._pomdp_state, action=action, success=success
        )
        self._chain.record(source=source, action=action, destination=cluster, success=success)

    def get_practice_policy(self, *, task):
        self._remaining_practice_actions = None
        self._pomdp_state = self._pomdp_state.model_copy(update={"accumulated_cost": 0.0})

        def policy(state):  # noqa: PLR0917 -- Policy protocol is positional
            cluster = self.classify()
            self.settle(cluster=cluster)
            if cluster is None:
                raise InteractionComplete()
            self._refresh_execution_forecasts()
            # Accounting configuration may have replaced the model after init.
            self._chain.base = self._pomdp_model
            horizon = min(
                self.pomdp_search_depth, self._remaining_practice_actions or self.pomdp_search_depth
            )
            trace = SearchTrace()
            try:
                value, action = self.pomdp_planner.solve(
                    environment_state=ClusterState(cluster_id=cluster),
                    belief_state=self._pomdp_state,
                    summed_cost=self._pomdp_state.accumulated_cost,
                    horizon=horizon,
                    model=self._chain,
                    trace=trace,
                    num_samples=self.pomdp_num_samples,
                )
            finally:
                trace.close()
            self._decision_index += 1
            self.record_diagnostic(
                event="decision",
                cluster=cluster,
                representation="generated_language_states",
                revision=self.manifest.revision,
                solver="expectimax",
                horizon=horizon,
                value=value,
                action=action.model_dump(mode="json"),
                search=trace.events,
                competences=self.practice_skill_competences(),
                learning_rates=self.practice_skill_learning_rates(),
                estimated_costs=self.practice_skill_costs(),
            )
            if isinstance(action, StopAction):
                raise InteractionComplete(planner_stop=True)
            name = action.option_id
            ground = self._chain.bindings[name]
            self._execution_ground_skill = ground
            self._pending_language = (cluster, name)
            if name in HUMANS:
                raise HumanCubeBinResetRequested(
                    cost=self.practice_action_cost(ground_skill=ground),
                    destination=self.skill_provider.movables_reset_destination(ground_skill=ground),
                )
            return skill_action(ground_skill=ground)

        return policy

    def observe_environment_reset(self, *, state):
        # The step host calls this immediately BEFORE executing a requested human
        # reset. Its pending outcome must be judged only after the reset occurs.
        if self._pending_language is not None and self._pending_language[1] not in HUMANS:
            self.settle(cluster=self.classify())

    def observe_help_granted(self, *, state):
        self.settle(cluster=self.classify())

    def end_cycle(self):
        if self._pending_language is not None:
            self.settle(cluster=self.classify())
        self.revise_skills()
        self._pomdp_state = refit_belief_state(
            state=self._pomdp_state,
            learning_rate_process_noise_std=self.pomdp_learning_rate_process_noise_std,
            advance_cycle=True,
        )
        self._cycle_index += 1
        self.record_diagnostic(
            event="refit",
            beliefs=self.belief_diagnostics(),
            estimated_costs=self.practice_skill_costs(),
        )
