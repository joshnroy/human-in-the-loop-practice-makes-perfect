"""Frozen deployment policies corresponding to the methods' practice objectives.

These policies only execute learned samplers greedily. They never train, reset a
task, or read its configured human costs. The spec is data, independent of the
training method's planner, simulator and open log files.
"""

from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta as beta_distribution

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


class CurrentCompetence(BaseModel):
    """A skill's competence now, learning included: competence after n tries is
    start + rate * n, with a Beta prior on the start and a normal prior on the rate
    (truncated at 0), conditioned on each practice outcome at the try it happened."""

    START_GRID: ClassVar[int] = 101
    RATE_GRID: ClassVar[int] = 41

    @staticmethod
    def estimate(
        *,
        prior: tuple[float, float, float, float],
        outcomes: list[bool],
        try_indices: list[int] | None = None,
        num_tries: int | None = None,
    ) -> float:
        """`try_indices` places each outcome at the try it happened (default: the i-th
        outcome at try i) and `num_tries` is how many tries "now" is after (default:
        one per outcome); they differ only when some tries were not observed."""
        c_mean, c_strength, r_mean, r_sd = prior
        indices = list(range(len(outcomes))) if try_indices is None else try_indices
        tries = len(outcomes) if num_tries is None else num_tries
        starts = (np.arange(CurrentCompetence.START_GRID) + 0.5) / CurrentCompetence.START_GRID
        rates = (
            np.array([r_mean])
            if r_sd <= 0
            else np.linspace(0.0, r_mean + 4 * r_sd, CurrentCompetence.RATE_GRID)
        )
        c, r = np.meshgrid(starts, rates, indexing="ij")
        log_w = beta_distribution.logpdf(c, c_mean * c_strength, (1 - c_mean) * c_strength)
        if r_sd > 0:
            log_w = log_w - 0.5 * ((r - r_mean) / r_sd) ** 2
        for i, success in zip(indices, outcomes, strict=True):
            p = np.clip(c + r * i, 1e-9, 1 - 1e-9)
            log_w = log_w + (np.log(p) if success else np.log(1 - p))
        w = np.exp(log_w - log_w.max())
        now = np.minimum(1.0, c + r * tries)
        return float((w * now).sum() / w.sum())


class DeploymentPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["shared_ees", "dp_mdp", "tossing_canonical", "reward_greedy"] = "shared_ees"
    source_method: str = "ees"
    horizon: int = Field(default=4, ge=0)
    max_states: int = Field(default=4096, ge=1)
    dp_priors: dict[str, DpSkillPrior] = Field(default_factory=dict)
    # Skill name -> (start, rate): perfect information. Competence after n practice
    # tries is min(1, start + rate * n), in place of any estimate from outcomes.
    known_curves: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # Skill name -> (competence mean, strength, rate mean, rate sd): deploy on the
    # current competence, learning included (`CurrentCompetence`), not the recent average.
    curve_priors: dict[str, tuple[float, float, float, float]] = Field(default_factory=dict)

    def current_competence(self, *, method: EesMethod, ground: GroundSkill) -> float | None:
        prior = self.curve_priors.get(ground.skill.name)
        if prior is None:
            return None
        history = method._competence_models.get(ground)
        outcomes = (
            [bool(o) for cycle in history.cycle_observations for o in cycle]
            if history is not None
            else []
        )
        if history is None or not getattr(history, "skipped_before", None):
            return CurrentCompetence.estimate(prior=prior, outcomes=outcomes)
        return CurrentCompetence.estimate(
            prior=prior,
            outcomes=outcomes,
            try_indices=history.observation_try_indices(),
            num_tries=history.num_tries,
        )

    def known_competence(self, *, method: EesMethod, ground: GroundSkill) -> float | None:
        curve = self.known_curves.get(ground.skill.name)
        if curve is None:
            return None
        history = method._competence_models.get(ground)
        tries = sum(len(c) for c in history.cycle_observations) if history is not None else 0
        return min(1.0, curve[0] + curve[1] * tries)

    def dp_competence(self, *, method: EesMethod, ground: GroundSkill) -> float:
        """DP values its cumulative posterior, rather than EES's recent window."""
        if (known := self.known_competence(method=method, ground=ground)) is not None:
            return known
        if (current := self.current_competence(method=method, ground=ground)) is not None:
            return current
        prior = self.dp_priors.get(ground.skill.name)
        history = method._competence_models.get(ground)
        return LinearSkillCompetenceModel(
            alpha=prior.alpha if prior is not None else method.prior_alpha,
            beta=prior.beta if prior is not None else method.prior_beta,
            cycle_observations=history.cycle_observations if history is not None else [[]],
        ).get_current_competence()

    def recent_competence(self, *, method: EesMethod, ground: GroundSkill) -> float:
        """The recent-window posterior mean of the snapshot's outcome counts, from this
        policy's own prior for the skill where it names one rather than EES's
        optimistic Beta(10, 1), under which a route never tried looks nearly perfect."""
        if (known := self.known_competence(method=method, ground=ground)) is not None:
            return known
        if (current := self.current_competence(method=method, ground=ground)) is not None:
            return current
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

    def best_route(
        self,
        *,
        atoms: frozenset[GroundAtom],
        rewards: dict[str, float],
        depth: int,
        visited: frozenset[frozenset[GroundAtom]],
    ) -> tuple[float, tuple[GroundSkill, ...]] | None:
        """The (value, skills) of the best route from `atoms` to the goal in at most
        `depth` skills; ties go to the route whose skill names sort first."""
        if depth <= 0:
            return None
        found: list[tuple[float, tuple[GroundSkill, ...]]] = []
        for ground in SkillGrounder.applicable_ground_skills(
            skills=self.method.skills(), objects=self.method.objects(), true_atoms=atoms
        ):
            if not ground.preconditions <= atoms:
                continue
            after = DpTaskModel.successor(atoms=atoms, ground=ground)
            if after in visited:
                continue
            competence = self.spec.recent_competence(method=self.method, ground=ground)
            if self.task.goal.atoms <= after:
                found.append((
                    DpTaskModel.goal_reward(atoms=after, goal_rewards=rewards) * competence,
                    (ground,),
                ))
                continue
            rest = self.best_route(
                atoms=after, rewards=rewards, depth=depth - 1, visited=visited | {after}
            )
            if rest is not None:
                found.append((competence * rest[0], (ground, *rest[1])))
        if not found:
            return None
        return min(
            found,
            key=lambda route: (
                -route[0],
                tuple((g.skill.name, tuple(o.name for o in g.objects)) for g in route[1]),
            ),
        )

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
        elif self.spec.kind == "reward_greedy":
            if self.remaining <= 0:
                return self.stop(reason="deployment horizon")
            # The route to the goal with the largest reward times the chance every skill
            # on it succeeds (`RewardedRoutesPracticeDomain`'s deployed policy): one
            # skill for a plain route, the whole chain for one whose skills must
            # succeed in a row. Each step runs the best route's next skill.
            rewards = self.method.skill_provider.goal_rewards()
            best = self.best_route(
                atoms=atoms, rewards=rewards, depth=self.remaining, visited=frozenset({atoms})
            )
            if best is None:
                return self.stop(reason="no route reaches the goal")
            ground = best[1][0]
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
