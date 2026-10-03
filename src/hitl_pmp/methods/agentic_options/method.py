"""Persistent option execution with visual checks and session-boundary learning."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from hitl_pmp.agentic_runtime.learner import PolicyLearner
from hitl_pmp.agentic_runtime.observations import ObservationEvidence, ObservationProvider
from hitl_pmp.agentic_runtime.types import OptionContract
from hitl_pmp.agentic_runtime.vision import VLMJudge
from hitl_pmp.core.method.method import HumanCubeBinResetRequested, InteractionComplete, Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossing3d.agentic_bridge import AgenticTossing3DEnvironment
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.stop_action import StopAction
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta

from .belief import LearnerBelief
from .chain_model import SkillChainModel
from .types import ClusterState, OptionAction, SearchConfig


class AgenticOptionsMethod(Method):
    """The same practice objective, over observed language-cluster identities.

    All execution evidence is settled before the session refit, including the last
    action at a step cap. Evaluation has its own simulator and never supplies
    training evidence. Unknown judgments end practice without inventing labels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    env: AgenticTossing3DEnvironment
    evaluation_env: AgenticTossing3DEnvironment
    practice_observer: ObservationProvider = Field(exclude=True)
    evaluation_observer: ObservationProvider = Field(exclude=True)
    evaluation_skill_order: tuple[str, ...]
    judge: VLMJudge
    learner: PolicyLearner = Field(exclude=True)
    learner_belief: LearnerBelief = Field(exclude=True)
    chain: SkillChainModel = Field(exclude=True)
    belief: Tossing3DBeliefState
    config: SearchConfig = Field(default_factory=SearchConfig)
    action_ids: dict[str, int]
    human_destinations: dict[str, str]
    audit_path: Path
    execution_records: dict[str, dict[str, Any]] = Field(default_factory=dict, exclude=True)
    evaluation_execution: dict[str, Any] = Field(default_factory=dict, exclude=True)
    _last_trajectory: str | None = PrivateAttr(default=None)
    _pending: tuple[str, OptionContract, dict[str, Any]] | None = PrivateAttr(default=None)
    _remaining: int = PrivateAttr(default=1)
    _session_active: bool = PrivateAttr(default=False)
    _unresolved: bool = PrivateAttr(default=False)
    _cost: float = PrivateAttr(default=0.0)

    def model_post_init(self, __context: object) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.chain, SkillChainModel):
            raise ValueError("Agentic practice requires a skill-chain model")

    def get_practice_policy(self, *, task: Task) -> Policy:
        del task
        self.learner.begin_session()
        self._session_active = True
        self._unresolved = False
        self._cost = 0.0
        self.belief = self.belief.model_copy(update={"accumulated_cost": 0.0})
        self.chain.blocked_actions.clear()
        return lambda state: self._practice_action(state=state)

    def _practice_action(self, *, state: State) -> LabeledAction:
        del state
        self._settle()
        if self._unresolved:
            raise InteractionComplete()
        observation = ObservationEvidence.with_history(
            observation=self.practice_observer.observe(), trajectory_path=self._last_trajectory
        )
        self.chain.blocked_actions.clear()
        classified = self.judge.classify(observation=observation, manifest=self.chain.manifest)
        self._audit(event="classification", judgment=classified.model_dump())
        if classified.cluster_id is None:
            self._unresolved = True
            raise InteractionComplete()
        current = ClusterState(cluster_id=classified.cluster_id)
        planner = ExpectimaxPlanner[
            ClusterState, Tossing3DBeliefState, Tossing3DTheta, OptionAction
        ](
            use_model_j=True,
            observation_probability_weight=self.config.observation_probability_weight,
        )
        while True:
            value, action = planner.solve(
                environment_state=current,
                summed_cost=self._cost,
                belief_state=self.belief,
                horizon=min(self.config.depth, self._remaining),
                model=self.chain,
            )
            self._audit(
                event="decision",
                cluster=current.cluster_id,
                value=value,
                action=action.model_dump(mode="json"),
                cost=self._cost,
            )
            if isinstance(action, StopAction):
                raise InteractionComplete(planner_stop=True)
            option = self.chain.options[action.option_id]
            initiation = self.judge.check(
                observation=observation, option=option, phase="initiation"
            )
            self._audit(
                event="initiation", option=option.option_id, judgment=initiation.model_dump()
            )
            if initiation.value is None:
                self._unresolved = True
                raise InteractionComplete()
            if not initiation.value:
                self.chain.blocked_actions.add((current.cluster_id, option.option_id))
                continue
            self._pending = current.cluster_id, option, observation
            if option.human_destination is not None:
                raise HumanCubeBinResetRequested(
                    cost=option.cost, destination=self.human_destinations[option.option_id]
                )
            return LabeledAction(
                action=np.array([self.action_ids[option.option_id], 0, 0, 0, 0], dtype=float),
                label=option.option_id,
            )

    def _settle(self) -> None:
        if self._pending is None:
            return
        source, option, before = self._pending
        self._pending = None
        self._cost += option.cost
        self.belief = self.belief.model_copy(update={"accumulated_cost": self._cost})
        after = self.practice_observer.observe()
        execution = self.execution_records.pop(option.option_id, {})
        self._last_trajectory = execution.get("trajectory_path")
        after["execution"] = execution
        self._audit(
            event="executed_unjudged",
            option=option.option_id,
            cost=option.cost,
            execution=execution,
        )
        try:
            after = ObservationEvidence.with_history(
                observation=after, trajectory_path=self._last_trajectory, before=before
            )
            termination = self.judge.check(observation=after, option=option, phase="termination")
            success = self.judge.check(observation=after, option=option, phase="success")
            classified = self.judge.classify(observation=after, manifest=self.chain.manifest)
        except Exception as exc:
            # A provider/parser failure does not undo the real robot action and
            # is not evidence that the robot's skill failed.
            self._unresolved = True
            self.learner.record(
                evidence={
                    "source": source,
                    "option_id": option.option_id,
                    "cost": option.cost,
                    "before": before,
                    "after": after,
                    "execution": execution,
                    "trajectory_path": execution.get("trajectory_path"),
                    "judgment_error": type(exc).__name__,
                    "resolved": False,
                }
            )
            self._audit(
                event="judgment_error", option=option.option_id, exception=type(exc).__name__
            )
            return
        evidence = {
            "source": source,
            "option_id": option.option_id,
            "termination": termination.model_dump(),
            "success": success.model_dump(),
            "destination": classified.model_dump(),
            "cost": option.cost,
            "before": before,
            "after": after,
            "execution": execution,
            "trajectory_path": execution.get("trajectory_path"),
        }
        # Trajectories and images are learner-owned real evidence, never search state.
        self.learner.record(evidence=evidence)
        self._audit(
            event="execution",
            **{key: value for key, value in evidence.items() if key not in ("before", "after")},
        )
        if termination.value is None or success.value is None or classified.cluster_id is None:
            self._unresolved = True
            return
        resolved_success = termination.value and success.value
        self.belief = self.learner_belief.observe(
            belief=self.belief,
            skill=option.belief_skill,
            success=resolved_success,
            observed_cost=option.cost,
        )
        self.chain.record_destination(
            source=source,
            option_id=option.option_id,
            outcome="success" if resolved_success else "failure",
            destination=classified.cluster_id,
        )

    def end_cycle(self) -> None:
        if not self._session_active:
            return
        self._settle()
        revision = self.learner.end_session()
        self.belief = self.learner_belief.advance_session(belief=self.belief)
        if revision.changed_files:
            self.chain.invalidate_code_revision()
        self._session_active = False
        self._audit(
            event="revision",
            revision=revision.model_dump(),
            belief=self.belief.model_dump(mode="json"),
        )

    def observe_practice_action_budget(self, *, remaining_actions: int) -> None:
        self._remaining = remaining_actions

    def observe_environment_reset(self, *, state: State) -> None:
        del state
        # PracticeLoop invokes this before dispatching the requested human reset.
        # Its outcome must be read after the reset, not labeled against the old world.
        if self._pending is not None and self._pending[1].human_destination is None:
            self._settle()

    def observe_help_granted(self, *, state: State) -> None:
        del state
        self._settle()

    def may_request_human_help(self) -> bool:
        return bool(self.human_destinations)

    def get_task_policy(self, *, task: Task) -> Policy:
        del task
        # Each evaluation task starts in its own reset state. Evidence from the
        # previous task must not be used to establish a new grasp.
        self.evaluation_execution.clear()
        return lambda state: self._evaluation_action(state=state)

    def _evaluation_action(self, *, state: State) -> LabeledAction:
        del state
        observation = ObservationEvidence.with_history(
            observation=self.evaluation_observer.observe(),
            trajectory_path=self.evaluation_execution.get("trajectory_path"),
        )
        classified = self.judge.classify(observation=observation, manifest=self.chain.manifest)
        if classified.cluster_id is not None:
            available_ids = {
                edge.option_id
                for edge in self.chain.manifest.edges
                if edge.source == classified.cluster_id
            }
            # The canonical deployment is Pick -> Toss with OpenGripper recovery.
            # No human help, exploration, learner record or belief update occurs here.
            for skill in self.evaluation_skill_order:
                for option_id in sorted(available_ids):
                    option = self.chain.options[option_id]
                    if option.human_destination is not None or option.belief_skill != skill:
                        continue
                    initiation = self.judge.check(
                        observation=observation, option=option, phase="initiation"
                    )
                    if initiation.value is True:
                        return LabeledAction(
                            action=np.array(
                                [self.action_ids[option.option_id], 0, 0, 0, 0], dtype=float
                            ),
                            label=option.option_id,
                        )
        return LabeledAction(action=self.evaluation_env.noop_action(), label="unresolved")

    def practice_skill_competences(self) -> dict[str, float]:
        return {name: value.mean_competence() for name, value in self.belief.skill_beliefs.items()}

    def practice_skill_learning_rates(self) -> dict[str, float]:
        return {
            name: value.mean_learning_rate() for name, value in self.belief.skill_beliefs.items()
        }

    def _audit(self, *, event: str, **payload: Any) -> None:
        with self.audit_path.open("a") as stream:
            stream.write(json.dumps({"event": event, **payload}, default=str) + "\n")

    def reset_environment(self, *, start_state: State) -> bool:
        return False

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError("PracticeLoop supplies the persistent-world task")

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError("Human setup is executed through PracticeLoop")

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError("Generated options execute through registered policy actions")

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError("Generated policy learning occurs only at end_cycle")
