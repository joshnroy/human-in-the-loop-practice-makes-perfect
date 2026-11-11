"""Frozen deployment policies corresponding to the methods' practice objectives.

These policies only execute learned samplers greedily. They never train, reset a
task, or read its configured human costs. The spec is data, independent of the
training method's planner, simulator and open log files.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.method import NoFeasibleParametersError
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILLS,
    TOSS_SKILL,
)
from hitl_pmp.methods.practice_makes_perfect.competence_models import (
    LinearSkillCompetenceModel,
    OptimisticSkillCompetenceModel,
)
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpTaskModel
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpSkillPrior
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.planning.grounding import SkillGrounder


class DeploymentPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["shared_ees", "dp_mdp", "tossing_canonical"] = "shared_ees"
    source_method: str = "ees"
    horizon: int = Field(default=4, ge=0)
    max_states: int = Field(default=4096, ge=1)
    dp_priors: dict[str, DpSkillPrior] = Field(default_factory=dict)

    def dp_competence(self, *, method: EesMethod, ground: GroundSkill) -> float:
        """DP values its cumulative posterior, rather than EES's recent window."""
        prior = self.dp_priors.get(ground.skill.name)
        history = method._competence_models.get(ground)
        return LinearSkillCompetenceModel(
            alpha=prior.alpha if prior is not None else method.prior_alpha,
            beta=prior.beta if prior is not None else method.prior_beta,
            cycle_observations=history.cycle_observations if history is not None else [[]],
        ).get_current_competence()

    def policy(self, *, method: EesMethod, task: Task) -> Policy:
        if self.kind == "shared_ees":
            return EesMethod.get_task_policy(method, task=task)
        episode = _DeploymentEpisode(method=method, task=task, spec=self)
        return lambda state: episode.step(state=state)


class SnapshotDeployment(EesMethod):
    """A deployment-only executor; no practice planner is rebuilt at evaluation."""

    deployment_policy: DeploymentPolicy

    def competence_model(self, *, ground_skill: GroundSkill) -> OptimisticSkillCompetenceModel:
        prior = self.deployment_policy.dp_priors.get(ground_skill.skill.name)
        if prior is not None and ground_skill not in self._competence_models:
            self._competence_models[ground_skill] = OptimisticSkillCompetenceModel(
                alpha=prior.alpha,
                beta=prior.beta,
                window_size=self.competence_window_size,
                recency_size=self.competence_recency_size,
            )
        return super().competence_model(ground_skill=ground_skill)

    def get_task_policy(self, *, task: Task) -> Policy:
        return self.deployment_policy.policy(method=self, task=task)


class _DeploymentEpisode:
    def __init__(self, *, method: EesMethod, task: Task, spec: DeploymentPolicy) -> None:
        self.method = method
        self.task = task
        self.spec = spec
        self.remaining = spec.horizon
        self.finished = False
        self.plan: list[GroundSkill] | None = None
        self.required: frozenset[GroundAtom] = frozenset()
        self.forbidden: frozenset[GroundAtom] = frozenset()

    def stop(self, *, reason: str) -> LabeledAction:
        self.finished = True
        return LabeledAction(action=self.method.env.noop_action(), label=f"no-op ({reason})")

    def step(self, *, state: State) -> LabeledAction:
        if self.finished:
            return self.stop(reason="deployment complete")
        atoms = self.method.abstract_state(state=state)
        if self.task.goal.atoms <= atoms:
            return self.stop(reason="goal")
        if self.spec.kind == "dp_mdp":
            if not self.required <= atoms or self.forbidden & atoms:
                return self.stop(reason="DP absorbing failure")
            if self.plan is None:
                mdp = DpTaskModel.build(
                    tasks=[(atoms, self.task.goal.atoms)],
                    skills=self.method.skills(),
                    objects=self.method.objects(),
                    max_states=self.spec.max_states,
                )
                _, policy = mdp.solve(
                    competences={
                        t.skill: self.spec.dp_competence(method=self.method, ground=t.skill)
                        for t in mdp.transitions
                    }
                )
                self.plan = []
                current = next(iter(mdp.initial_probabilities))
                visited = set()
                while current in policy and current not in visited:
                    visited.add(current)
                    transition = policy[current]
                    self.plan.append(transition.skill)
                    current = transition.successor
            if not self.plan:
                return self.stop(reason="DP policy complete")
            ground = self.plan.pop(0)
            ignored = ground.expected_ignore_effects(before_atoms=atoms)
            added = ground.expected_add_effects(before_atoms=atoms)
            self.required = (
                frozenset(a for a in atoms if a.predicate not in ignored) - ground.delete_effects
                | added
            )
            self.forbidden = ground.delete_effects - added
        else:
            if self.remaining <= 0:
                return self.stop(reason="deployment horizon")
            # J's policy: pick when ready, open after a failed pick, toss once
            # when holding. Both toss outcomes terminate, including a missed bin.
            names = {a.predicate.name for a in atoms}
            allowed = (
                {TOSS_SKILL}
                if "Holding" in names
                else {OPEN_GRIPPER_SKILL}
                if "ClosedEmpty" in names
                else PICK_SKILLS
                if "HandEmpty" in names
                else frozenset()
            )
            candidates = [
                g
                for g in SkillGrounder.applicable_ground_skills(
                    skills=self.method.skills(), objects=self.method.objects(), true_atoms=atoms
                )
                if g.skill.name in allowed and g.preconditions <= atoms
            ]
            if not candidates:
                return self.stop(reason="canonical policy unavailable")
            ground = min(
                candidates,
                key=lambda g: (
                    g.skill.name,
                    tuple(o.name for o in g.objects),
                ),
            )
            self.remaining -= 1
            if ground.skill.name == TOSS_SKILL:
                self.finished = True
        if not ground.preconditions <= atoms:
            return self.stop(reason="policy preconditions unavailable")
        try:
            action, _ = self.method.execute_ground_skill(
                ground_skill=ground, state=state, explore=False
            )
        except NoFeasibleParametersError:
            return self.stop(reason="no feasible parameters")
        return action
