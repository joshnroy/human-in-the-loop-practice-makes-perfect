from typing import Any

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.method.method import InteractionComplete, Method
from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.planning.grounding import SkillGrounder


class RandomSkillsMethod(Method):
    """Random Skills: at each step, uniformly sample among the currently-applicable
    ground skills and execute one -- no planning, no competence model, no sampler
    learning. Matches predicators' own RandomOptionsApproach.

    When *nothing* is applicable -- a dead end -- it degrades exactly the way
    EesMethod does, and along the same phase split: a no-op during evaluation
    (get_labeled_action), InteractionComplete during practice (get_practice_policy).
    It used to assert instead, which crashed every run on the first domain that has a
    genuine dead end.

    Fully domain-agnostic: everything domain-specific (the lifted skills, the
    predicates/objects to ground over, and how a chosen ground skill becomes a raw
    Action) comes from the injected `SkillProvider`, so this one class runs on any
    `--env` -- there is no per-domain `RandomSkillsPolicy` and no
    `isinstance(self.env, ...)` dispatch anymore.

    seed carries this Method's own RNG stream (same private-RNG-derived-from-seed
    pattern as LightSwitchTasks: a public seed field, a PrivateAttr populated in
    model_post_init, never reassigned directly)."""

    skill_provider: SkillProvider
    seed: int = 0

    _rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)

    def applicable_ground_skills(self, *, state: State) -> list[GroundSkill]:
        """Everything whose preconditions hold in `state`. Empty means a dead end.

        Split out because both phases need to *ask* the question before acting on it,
        and they answer it differently -- and because it draws no randomness, so
        checking it never perturbs this Method's RNG stream."""
        provider = self.skill_provider
        objects = provider.objects()
        true_atoms = SkillGrounder.abstract_state(
            state=state, objects=objects, predicates=provider.predicates()
        )
        return SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=objects, true_atoms=true_atoms
        )

    def get_labeled_action(self, *, state: State) -> LabeledAction:
        provider = self.skill_provider
        ground_skills = self.applicable_ground_skills(state=state)
        if not ground_skills:
            # A dead end: no skill's preconditions hold, and this baseline has no
            # planner to route out of one. It used to assert here, which took the
            # paper's own lower-bound arm off the table entirely on the first domain
            # that has a genuine dead end -- Tossing3D, where `Toss` unconditionally
            # deletes `Reachable(cube, barrier)` past a one-way barrier, so every
            # episode ends in one and all 10/10 runs crashed.
            #
            # This is the *evaluation* answer, and it matches EesMethod's: return
            # `LabeledAction(..., "no-op ...")`, because run_task_episode owns
            # termination (goal check + horizon) and a policy must not end its
            # caller's episode from in here. The practice answer is different -- see
            # get_practice_policy.
            return LabeledAction(
                action=self.env.noop_action(), label="no-op (no applicable skills)"
            )
        ground_skill = ground_skills[int(self._rng.integers(len(ground_skills)))]

        params = provider.sample_params(ground_skill=ground_skill, rng=self._rng)
        action = provider.compute_action(ground_skill=ground_skill, params=params, state=state)
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(p), 2) for p in params]}"
        return LabeledAction(action=action, label=label)

    def reset_environment(self, *, start_state: State) -> bool:
        """No irreversible actions matter to this baseline and the base PMP paper has
        no human-in-the-loop layer -- a direct environment set stands in for a real
        "self-navigate without help" recovery (matches SkillOracleMethod)."""
        self.env.set_state(state=start_state)
        return True

    def get_task_policy(self, *, task: Task) -> Policy:
        del task  # never consulted -- this baseline always samples uniformly among
        # applicable ground skills, regardless of which task it's handed
        return lambda state: self.get_labeled_action(state=state)

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Identical to the evaluation policy except on a dead end, where this raises
        InteractionComplete instead of emitting a no-op -- exactly the split EesMethod
        makes.

        Overridden rather than inherited (the default just forwards to
        get_task_policy) because the two phases genuinely differ here, and getting it
        wrong biases a comparison rather than merely wasting steps. practice_loop.py
        charges `num_online_transitions += 1` per step and a practice period has no
        goal check, so a dead-ended period would burn its entire remaining budget on
        no-ops -- ~145 of 150 steps on Tossing3D, where `Toss` dead-ends every episode
        after ~3. EES stops charging at that point. Both arms are plotted against that
        same transition axis, and `--method random-skills --num-cycles 10` over EES's
        budget is exactly how the two are put on one chart, so letting only one arm
        keep spending is the budget-driven-vs-data-driven asymmetry InteractionComplete
        exists to prevent.

        This baseline does not learn from what it collects, so ending the period early
        costs it nothing: unlike EES it has no reason to prefer more transitions."""
        del task  # as in get_task_policy: never consulted
        return lambda state: self.practice_step(state=state)

    def practice_step(self, *, state: State) -> LabeledAction:
        """get_practice_policy's body -- a named method rather than a closure, since
        every parameter in this project is keyword-only (ruff PLR0917) and a `Policy`
        takes its state positionally. Same shape as EesMethod's own
        `lambda state: episode.step(state=state)`."""
        if not self.applicable_ground_skills(state=state):
            raise InteractionComplete
        return self.get_labeled_action(state=state)

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "RandomSkillsMethod.generate_train_task is unreachable: this baseline never "
            "*chooses* what to practice. It can still be run over a practice budget "
            "(--num-cycles > 0, to share EES's transition axis), but practice_loop.py "
            "samples those tasks from Problem.tasks, never from a Method."
        )

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "RandomSkillsMethod.execute_setup_command is unreachable: "
            "no HumanOracle is ever used in this reproduction."
        )

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "RandomSkillsMethod.execute_skill is unreachable: this baseline "
            "computes its own ground skill choice directly, it never practices one."
        )

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "RandomSkillsMethod.improve_skill_parameters is unreachable: "
            "this baseline never learns."
        )
