from typing import Any

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.method.method import InteractionComplete, Method
from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import (
    GroundSkill,
    LabeledAction,
    Policy,
    Rollout,
    SamplerConsultation,
    SetupCommand,
    SkillPracticeTally,
)
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
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

    It learns nothing, but it does *execute* skills, so it records what practicing
    them did (`practice_outcomes`). That is not bookkeeping for its own sake: how often
    a uniform draw succeeds is the reference every learned arm's practice rate is read
    against, and until this existed the reference was absent from every run's
    stats.json on every domain.

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
    # Cumulative per LIFTED skill name, surfaced through Method.practice_outcomes into
    # stats.json. This baseline learns nothing, but it *executes* skills, and how often
    # a uniform draw succeeds is exactly the reference every learned arm is measured
    # against -- see practice_outcomes.
    _practice_tallies: dict[str, SkillPracticeTally] = PrivateAttr()
    # The skill executed on the previous practice step, whose outcome is only knowable
    # from the *next* state. None outside a practice period, and reset at every period
    # boundary -- see get_practice_policy.
    _pending_practice_skill: GroundSkill | None = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._practice_tallies = {}
        self._pending_practice_skill = None

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

    def abstract_state(self, *, state: State) -> frozenset[GroundAtom]:
        """The symbolic reading of `state`, for scoring a skill's add effects against.

        Draws no randomness, so calling it never perturbs this Method's RNG stream --
        which is what lets the practice tally be a pure observer. Same helper EesMethod
        keeps, for the same reason."""
        provider = self.skill_provider
        return SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )

    def get_labeled_action(self, *, state: State) -> LabeledAction:
        """The action alone -- what a `Policy` is required to return. Practice needs the
        chosen ground skill too, so the body lives in `choose_ground_skill` and this
        discards the second half rather than duplicating the draw."""
        return self.choose_ground_skill(state=state)[0]

    def choose_ground_skill(self, *, state: State) -> tuple[LabeledAction, GroundSkill | None]:
        """`(action, the ground skill it executes)`, or `(no-op, None)` on a dead end.

        Split out of `get_labeled_action` so practice can remember what it just ran and
        score it on the next step. Deliberately *one* draw path shared by both phases:
        a second copy for practice would be a second RNG consumer to keep in step, and
        the two phases of this baseline are identical wherever a skill is applicable."""
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
            return (
                LabeledAction(action=self.env.noop_action(), label="no-op (no applicable skills)"),
                None,
            )
        ground_skill = ground_skills[int(self._rng.integers(len(ground_skills)))]

        params = provider.sample_params(ground_skill=ground_skill, rng=self._rng)
        action = provider.compute_action(ground_skill=ground_skill, params=params, state=state)
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(p), 2) for p in params]}"
        return LabeledAction(action=action, label=label), ground_skill

    def reset_environment(self, *, start_state: State) -> bool:
        """Always False: this baseline has no self-navigation to offer, so it declines
        rather than reporting a success it did not achieve (matches SkillOracleMethod).

        This used to `self.env.set_state(state=start_state); return True` -- a
        privileged external state write dressed up as the agent recovering under its
        own power. Nothing calls this method today, so the lie cost nothing; the moment
        a reset-free loop branches on the return value it would silently treat every
        stranded robot as rescued."""
        del start_state  # nothing to navigate towards -- see above
        return False

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
        # Nothing is in flight at a period boundary. Carrying a pending skill across
        # would score it against the next period's initial state -- a different task's
        # state entirely -- so the last skill of a period goes unobserved instead.
        # EesMethod drops it the same way (its fresh per-period _EesEpisode starts with
        # _pending = None), and matching that exactly is what keeps the two arms
        # comparable: every period loses exactly one observation, in both.
        self._pending_practice_skill = None
        return lambda state: self.practice_step(state=state)

    def practice_step(self, *, state: State) -> LabeledAction:
        """get_practice_policy's body -- a named method rather than a closure, since
        every parameter in this project is keyword-only (ruff PLR0917) and a `Policy`
        takes its state positionally. Same shape as EesMethod's own
        `lambda state: episode.step(state=state)`.

        Settles the previous step's skill *before* the dead-end check, so a period that
        ends here still records what it last did -- the order `_EesEpisode.step` uses
        (observe_pending, then InteractionComplete)."""
        self.settle_pending_practice_skill(state=state)
        if not self.applicable_ground_skills(state=state):
            raise InteractionComplete
        labeled, ground_skill = self.choose_ground_skill(state=state)
        self._pending_practice_skill = ground_skill
        return labeled

    def observe_environment_reset(self, *, state: State) -> None:
        """Score the in-flight skill against the state the harness is about to discard,
        rather than against the initial state it is about to be teleported to.

        Only reachable with `practice_reset_interval` set. Without it, an outcome read
        on the next step would check add effects against a freshly reset environment,
        where they essentially never hold -- a spurious failure recorded once per reset,
        mislabelling that scales exactly with reset frequency, which is the confound
        that knob exists to isolate. EesMethod overrides this for the same reason; this
        baseline needs it the moment its outcomes are recorded at all."""
        self.settle_pending_practice_skill(state=state)

    def settle_pending_practice_skill(self, *, state: State) -> None:
        """Tally the previous practice step's skill against what `state` shows, and
        clear it. A no-op when nothing is in flight, so it is safe to call from both
        the normal step and the reset hook."""
        pending = self._pending_practice_skill
        if pending is None:
            return
        self._pending_practice_skill = None
        self.observe_practice_attempt(
            ground_skill=pending, true_atoms=self.abstract_state(state=state)
        )

    def observe_practice_attempt(
        self, *, ground_skill: GroundSkill, true_atoms: frozenset[GroundAtom]
    ) -> None:
        """File one executed skill into its lifted skill's tally.

        Success is `add_effects <= true_atoms`, the same predicate EesMethod scores by,
        so the two arms' practice rates mean the same thing and can be put on one chart.

        **Consultation is never INFORMED or UNINFORMATIVE here**, and that is the whole
        point of this baseline: it consults no sampler at all. A parameterized skill's
        params are a uniform draw over the same space epsilon-greedy exploration would
        use, which is exactly `EPSILON_RANDOM` -- a draw carrying no belief. A skill
        with `param_dim == 0` has no sampler that could ever be constructed for it,
        which is `NO_SAMPLER`. Recording either of the other two would make this arm
        look as though it had learned something, and it is the reference the learned
        arms are measured against."""
        name = ground_skill.skill.name
        consultation = (
            SamplerConsultation.NO_SAMPLER
            if ground_skill.skill.param_dim == 0
            else SamplerConsultation.EPSILON_RANDOM
        )
        self._practice_tallies[name] = self._practice_tallies.get(
            name, SkillPracticeTally()
        ).with_attempt(success=ground_skill.add_effects <= true_atoms, consultation=consultation)

    def practice_outcomes(self) -> dict[str, SkillPracticeTally]:
        """Per lifted skill, cumulative over the run; method_runner.py differences them
        per window. See Method.practice_outcomes.

        Overridden rather than left at the `{}` default, which was wrong here. The
        default is correct for a Method that never scores its own skill executions --
        but this one *does* execute skills, so an empty tally made `--method
        random-skills` uninstrumentable on every domain: the uniform reference arm's
        practice success rate simply did not exist in any run's stats.json.

        A copy, so a caller holding last cycle's reading to difference against cannot
        have it mutated underneath by the next practice step."""
        return dict(self._practice_tallies)

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
