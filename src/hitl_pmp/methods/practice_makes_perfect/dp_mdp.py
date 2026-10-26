"""DP's task MDP: successful skills apply their symbolic effects; failures absorb.

The task policy is optimized again for every proposed training allocation. Human
help is available during practice, but absent from deployment transitions. This
module models deployment, not a belief-space search over future practice resets.
"""

import heapq
import math
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.planning.grounding import SkillGrounder

SkillKey = TypeVar("SkillKey", bound=Hashable)


class DpTransition(BaseModel, Generic[SkillKey]):
    model_config = ConfigDict(frozen=True)

    state: int
    skill: SkillKey
    successor: int


class DpMdp(BaseModel, Generic[SkillKey]):
    model_config = ConfigDict(frozen=True)

    initial_probabilities: Mapping[int, float]
    terminal_rewards: Mapping[int, float]
    transitions: Sequence[DpTransition[SkillKey]]
    discount: float = 1.0

    @model_validator(mode="after")
    def _validate_distribution(self) -> "DpMdp[SkillKey]":
        if not 0 < self.discount <= 1:
            raise ValueError("DP's discount must be in (0, 1]")
        if any(
            not math.isfinite(value) or value < 0 for value in self.initial_probabilities.values()
        ):
            raise ValueError("DP's initial probabilities must be finite and nonnegative")
        if self.initial_probabilities and not math.isclose(
            sum(self.initial_probabilities.values()), 1
        ):
            raise ValueError("DP's initial probabilities must sum to one")
        if any(not math.isfinite(value) or value < 0 for value in self.terminal_rewards.values()):
            raise ValueError("DP's terminal rewards must be finite and nonnegative")
        return self

    def relevant_skills(self) -> set[SkillKey]:
        forward: dict[int, list[DpTransition[SkillKey]]] = {}
        reverse: dict[int, list[int]] = {}
        for transition in self.transitions:
            if transition.state in self.terminal_rewards:
                continue
            forward.setdefault(transition.state, []).append(transition)
            reverse.setdefault(transition.successor, []).append(transition.state)
        reachable = {
            state for state, probability in self.initial_probabilities.items() if probability > 0
        }
        pending = list(reachable)
        while pending:
            for transition in forward.get(pending.pop(), []):
                if transition.successor not in reachable:
                    reachable.add(transition.successor)
                    pending.append(transition.successor)
        useful = {state for state, reward in self.terminal_rewards.items() if reward > 0}
        pending = list(useful)
        while pending:
            for state in reverse.get(pending.pop(), []):
                if state not in useful:
                    useful.add(state)
                    pending.append(state)
        return {
            transition.skill
            for transition in self.transitions
            if transition.state in reachable
            and transition.state not in self.terminal_rewards
            and transition.successor in useful
        }

    def solve(
        self, *, competences: Mapping[SkillKey, float]
    ) -> tuple[float, dict[int, DpTransition[SkillKey]]]:
        """Solve the Bellman equations exactly by backward maximum-product search.

        Each action either reaches its success successor or absorbs with reward zero.
        Since discount and success probabilities are at most one, cycles cannot
        improve terminal reward. Equal values prefer fewer steps, avoiding unit-
        probability loops in the extracted optimal policy.
        """
        reverse: dict[int, list[DpTransition[SkillKey]]] = {}
        for transition in self.transitions:
            probability = competences[transition.skill]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("DP's skill competences must be probabilities")
            if transition.state not in self.terminal_rewards:
                reverse.setdefault(transition.successor, []).append(transition)
        values = dict(self.terminal_rewards)
        lengths = dict.fromkeys(self.terminal_rewards, 0)
        policy: dict[int, DpTransition[SkillKey]] = {}
        queue = [(-value, 0, state) for state, value in values.items() if value > 0]
        heapq.heapify(queue)
        while queue:
            negative, length, state = heapq.heappop(queue)
            value = -negative
            if value != values[state] or length != lengths[state]:
                continue
            for transition in reverse.get(state, []):
                candidate = self.discount * competences[transition.skill] * value
                previous = values.get(transition.state, 0)
                shorter = length + 1 < lengths.get(transition.state, math.inf)
                if candidate > previous or candidate == previous and candidate > 0 and shorter:
                    values[transition.state] = candidate
                    lengths[transition.state] = length + 1
                    policy[transition.state] = transition
                    heapq.heappush(queue, (-candidate, length + 1, transition.state))
        return sum(
            probability * values.get(state, 0)
            for state, probability in self.initial_probabilities.items()
        ), policy

    def curriculum(self, *, competences: Mapping[SkillKey, float]) -> list[SkillKey]:
        _, policy = self.solve(competences=competences)
        depths: dict[SkillKey, int] = {}
        for initial, probability in self.initial_probabilities.items():
            if probability <= 0:
                continue
            state = initial
            visited = set()
            depth = 0
            while state in policy and state not in visited:
                visited.add(state)
                transition = policy[state]
                depths[transition.skill] = min(depths.get(transition.skill, depth), depth)
                state = transition.successor
                depth += 1
        return sorted(depths, key=lambda skill: depths[skill])


class DpTaskModel:
    @staticmethod
    def build(
        *,
        tasks: Sequence[tuple[frozenset[GroundAtom], frozenset[GroundAtom]]],
        skills: tuple[Skill, ...],
        objects: tuple[Object, ...],
        max_states: int,
    ) -> DpMdp[GroundSkill]:
        states: dict[tuple[int, frozenset[GroundAtom]], int] = {}
        pending: deque[tuple[int, frozenset[GroundAtom]]] = deque()
        initial: dict[int, float] = {}
        rewards: dict[int, float] = {}
        transitions: list[DpTransition[GroundSkill]] = []

        def intern(*, task: int, atoms: frozenset[GroundAtom]) -> int:
            key = (task, atoms)
            if key not in states:
                if len(states) >= max_states:
                    raise ValueError(f"DP task MDP exceeds the cap of {max_states} states")
                states[key] = len(states)
                pending.append(key)
            return states[key]

        for task, (atoms, _goal) in enumerate(tasks):
            initial[intern(task=task, atoms=atoms)] = 1 / len(tasks)
        while pending:
            task, atoms = pending.popleft()
            state = states[(task, atoms)]
            if tasks[task][1] <= atoms:
                rewards[state] = 1
                continue
            applicable = SkillGrounder.applicable_ground_skills(
                skills=skills,
                objects=objects,
                true_atoms=atoms,
            )
            for ground in applicable:
                if not ground.preconditions <= atoms:
                    continue
                kept = frozenset(
                    atom for atom in atoms if atom.predicate not in ground.ignore_effects
                )
                after = (kept - ground.delete_effects) | ground.expected_add_effects(
                    before_atoms=atoms
                )
                transitions.append(
                    DpTransition(
                        state=state,
                        skill=ground,
                        successor=intern(task=task, atoms=after),
                    )
                )
        return DpMdp(
            initial_probabilities=initial, terminal_rewards=rewards, transitions=transitions
        )
