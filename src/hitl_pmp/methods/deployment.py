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
from hitl_pmp.methods.belief_space.rack_serving_domain import RackServingDeployment
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

    kind: Literal[
        "shared_ees", "dp_mdp", "tossing_canonical", "reward_greedy", "rack_value_model"
    ] = "shared_ees"
    source_method: str = "ees"
    horizon: int = Field(default=4, ge=0)
    max_states: int = Field(default=4096, ge=1)
    dp_priors: dict[str, DpSkillPrior] = Field(default_factory=dict)
    # rack_value_model: `RackServingPracticeDomain.deployment_plan` for the planner's
    # belief when the snapshot was taken, keyed by `rack_key`; each entry is the skill
    # to attempt and the rack count its success reaches, or None to stop.
    rack_plan: dict[str, tuple[str, int] | None] = Field(default_factory=dict)

    @staticmethod
    def rack_key(*, state: tuple[int, str, bool], remaining: int) -> str:
        return f"{state[0]}|{state[1]}|{int(state[2])}|{remaining}"

    def dp_competence(self, *, method: EesMethod, ground: GroundSkill) -> float:
        """DP deploys on the estimator it practised with. A snapshot from before DP chose
        one (2026-11-12) pooled every outcome, rather than EES's recent window."""
        prior = self.dp_priors.get(ground.skill.name)
        history = method._competence_models.get(ground)
        return LinearSkillCompetenceModel(
            alpha=prior.alpha if prior is not None else method.prior_alpha,
            beta=prior.beta if prior is not None else method.prior_beta,
            estimator=getattr(method, "dp_competence_estimator", "cumulative"),
            cycle_observations=history.cycle_observations if history is not None else [[]],
        ).get_current_competence()

    def recent_competence(self, *, method: EesMethod, ground: GroundSkill) -> float:
        """The recent-window posterior mean of the snapshot's outcome counts, from this
        policy's own prior for the skill where it names one rather than EES's
        optimistic Beta(10, 1), under which a route never tried looks nearly perfect."""
        prior = self.dp_priors.get(ground.skill.name)
        history = method._competence_models.get(ground)
        return OptimisticSkillCompetenceModel(
            alpha=prior.alpha if prior is not None else method.prior_alpha,
            beta=prior.beta if prior is not None else method.prior_beta,
            window_size=method.competence_window_size,
            recency_size=method.competence_recency_size,
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
                    goal_rewards=self.method.skill_provider.goal_rewards(),
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
        elif self.spec.kind == "rack_value_model":
            if self.remaining <= 0:
                return self.stop(reason="deployment horizon")
            # The rack value model's own plan for the planner's belief: look the
            # measured abstract state up, with the skills still allowed.
            rack = RackServingDeployment.state_of(
                atoms=[(a.predicate.name, tuple(o.name for o in a.objects)) for a in atoms]
            )
            choice = self.spec.rack_plan.get(
                DeploymentPolicy.rack_key(state=rack, remaining=self.remaining)
            )
            if choice is None:
                return self.stop(reason="value model stops")
            name, rack_after = choice
            candidates = [
                g
                for g in SkillGrounder.applicable_ground_skills(
                    skills=self.method.skills(), objects=self.method.objects(), true_atoms=atoms
                )
                if g.skill.name == name and g.preconditions <= atoms
            ]
            if name == "PickFood":
                # The model has two picks: off the rack (its count drops) or off the plate.
                candidates = [
                    g
                    for g in candidates
                    if any(
                        a.predicate.name == "OnRack" and g.objects[1] in a.objects for a in atoms
                    )
                    == (rack_after < rack[0])
                ]
            if not candidates:
                return self.stop(reason="value model's skill unavailable")
            ground = min(candidates, key=lambda g: tuple(o.name for o in g.objects))
            self.remaining -= 1
        elif self.spec.kind == "reward_greedy":
            if self.remaining <= 0:
                return self.stop(reason="deployment horizon")
            # One attempt of the skill that reaches the goal with the largest reward
            # times competence (`RewardedRoutesPracticeDomain`'s deployed policy).
            rewards = self.method.skill_provider.goal_rewards()
            routes = [
                (
                    DpTaskModel.goal_reward(atoms=after, goal_rewards=rewards)
                    * self.spec.recent_competence(method=self.method, ground=g),
                    g,
                )
                for g in SkillGrounder.applicable_ground_skills(
                    skills=self.method.skills(), objects=self.method.objects(), true_atoms=atoms
                )
                if g.preconditions <= atoms
                and self.task.goal.atoms <= (after := DpTaskModel.successor(atoms=atoms, ground=g))
            ]
            if not routes:
                return self.stop(reason="no route reaches the goal")
            # Ties go to the first skill by name, so the choice never depends on order.
            _, ground = min(
                routes,
                key=lambda route: (
                    -route[0],
                    route[1].skill.name,
                    tuple(o.name for o in route[1].objects),
                ),
            )
            self.remaining -= 1
            self.finished = self.remaining <= 0
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
