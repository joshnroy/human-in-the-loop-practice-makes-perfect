import time
from pathlib import Path
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
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task
from hitl_pmp.methods.pure_agent.agent_backend import AgentBackend
from hitl_pmp.methods.pure_agent.decision_parser import DecisionParseError, DecisionParser
from hitl_pmp.methods.pure_agent.observation import ObservationBuilder
from hitl_pmp.methods.pure_agent.prompts import DIGEST_REQUEST, PromptArm, PromptBuilder
from hitl_pmp.methods.pure_agent.types import (
    AgentCallKind,
    AgentCallRecord,
    AgentPhase,
)
from hitl_pmp.planning.grounding import SkillGrounder


class PureAgentMethod(Method):
    """The **pure agent** baseline: step 2.5 of Tom Silver's recipe, which slots between
    the random baseline (step 2) and the privileged oracle (step 3).

    **The agent IS the policy.** It is queried once per environment step, handed the
    observation at that decision point, and replies with the skill to execute and the
    parameters to execute it with. There is no generated file, no compiled artifact and no
    code path in which the agent's output is executed more than once. This is the design
    the `prpl-agent-utils` notebook declines on cost grounds and the blog proposes anyway;
    it is affordable here because a decision is one short assistant turn with no tools --
    measured at 2.4-4.2 s and well under a cent per call on Opus.

    Three design decisions carry this class, and each is a real choice with a live
    alternative.

    **1. The conversation persists within an episode and is cleared at its boundary.**
    A practice period is one conversation over its whole 150 steps, so the agent sees what
    its previous action did and can adapt inside the period. An evaluation episode is one
    conversation over its ~12 steps, so it can act coherently across a plan. Neither
    conversation outlives its unit.

    The alternative -- one conversation for the entire run -- was rejected on two grounds,
    the second of which is fatal. It is unaffordable: cost per call grows roughly linearly
    with conversation length (measured on a 40-turn probe, $0.0015 at turn 0 to $0.0079 at
    turn 34), so a 5,000-turn run would spend most of its money re-reading itself. And it
    cannot be firewalled: a single conversation spanning practice and evaluation carries
    every held-out task into the next one.

    **2. Knowledge crosses from practice to evaluation through one written note, and
    nothing else.** At the end of each practice period the agent is asked to write down
    what it has learned (`end_cycle`); that note is prepended to the opening prompt of
    every subsequent episode, practice and evaluation alike. It is natural language the
    agent wrote for itself, it is never executed, and every decision is still made by the
    agent at the moment it acts.

    That note is the *only* thing that crosses, which is what makes the firewall
    structural rather than a convention:

    - practice and evaluation run on **two different `AgentBackend` instances**, with
      separate sandboxes and separate conversations, so there is no shared object for a
      leak to travel through;
    - the note is produced by a query on the **practice** conversation, so it can only
      contain what practice saw;
    - `PromptBuilder.evaluation_opening` and `PromptBuilder.evaluation_step` have **no
      parameter** that could carry an outcome, a tally or a score, so a leak would require
      changing a signature rather than adding a line;
    - the evaluation conversation is reset at every test task, so nothing crosses from one
      held-out task to the next either.

    `test_no_evaluation_step_ever_reaches_the_practice_agent` is the test that bites: it
    asserts the practice backend's call count exactly, so routing evaluation through it
    fails by the evaluation step count rather than by a judgement call.

    **3. A run is reproducible only as a replay of itself, and that is stated rather than
    engineered around.** See `ReplayAgentBackend`: replaying the recorded decisions
    reproduces this run exactly as long as the state sequence is unchanged, and says so
    loudly when it is not. It does not reproduce the *method* -- a second live run makes
    different decisions -- so a replay is a re-scoring of one run, never a second sample.

    Fully domain-agnostic: everything domain-specific (the lifted skills, the predicates
    and objects to ground over, and how a chosen ground skill becomes a raw `Action`) comes
    from the injected `SkillProvider`, so this one class runs on any `--env`."""

    skill_provider: SkillProvider
    # Two backends, deliberately not one. See the class docstring: the split is the
    # firewall, and a single backend would make the guarantee a matter of discipline.
    practice_backend: AgentBackend
    evaluation_backend: AgentBackend
    prompt_arm: PromptArm = PromptArm.MINIMAL
    domain_description: str = ""
    # One JSON line per agent call, appended the moment the call returns. `None` disables
    # journalling entirely, which is what the tests use. See `AgentCallRecord`.
    ledger_path: Path | None = None
    # The note is written into every subsequent opening prompt, so an unbounded one would
    # grow the per-episode cost without bound. Truncation is announced to the agent.
    max_digest_chars: int = 12000
    # How much of each reply to keep in the ledger. Replies are one line of JSON when the
    # agent behaves; the cap only bites on the ones worth reading.
    max_reply_chars: int = 2000

    _digest: str = PrivateAttr(default="")
    _cycle_index: int = PrivateAttr(default=0)
    _step_index: int = PrivateAttr(default=0)
    _episode_index: int = PrivateAttr(default=0)
    _records: list[AgentCallRecord] = PrivateAttr(default_factory=list)
    _practice_tallies: dict[str, SkillPracticeTally] = PrivateAttr(default_factory=dict)
    # The skill executed on the previous practice step, whose outcome is only knowable
    # from the *next* state. `None` outside a practice period and at every period
    # boundary -- see `get_practice_policy`.
    _pending_practice_skill: GroundSkill | None = PrivateAttr(default=None)
    _pending_practice_label: str = PrivateAttr(default="")
    # What to tell the agent about its previous action, built when that action is settled.
    _previous_outcome: str | None = PrivateAttr(default=None)

    # ---- the symbolic layer, shared by both phases -------------------------------

    def applicable_ground_skills(self, *, state: State) -> list[GroundSkill]:
        """Everything whose preconditions hold in `state`. Empty means a dead end.

        Consulted **before** the agent is queried, and that ordering is what bounds the
        cost of a dead end to nothing: there is no decision to make when the applicable set
        is empty, so no call is paid for."""
        provider = self.skill_provider
        objects = provider.objects()
        return SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=objects, true_atoms=self.abstract_state(state=state)
        )

    def abstract_state(self, *, state: State) -> frozenset[GroundAtom]:
        """The symbolic reading of `state`, for the observation and for scoring a skill's
        add effects against. Draws no randomness, so calling it perturbs nothing."""
        provider = self.skill_provider
        return SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )

    def build_observation(
        self, *, state: State, goal: Goal, ground_skills: list[GroundSkill]
    ) -> dict[str, Any]:
        return ObservationBuilder.build(
            state=state,
            goal=goal,
            atoms=self.abstract_state(state=state),
            objects=self.skill_provider.objects(),
            ground_skills=ground_skills,
        )

    # ---- evaluation -------------------------------------------------------------

    def get_task_policy(self, *, task: Task) -> Policy:
        """One held-out test task. Starts a **new** conversation on the evaluation
        backend and opens it with the contract, the symbolic layer and the agent's own
        notes.

        The reset is not tidiness. `PracticeLoop._evaluate` calls this once per test task,
        so without it the agent would carry everything it saw on task k into task k+1 --
        learning across held-out tasks, which is training on the test set by a slower
        route."""
        self.evaluation_backend.reset()
        self._episode_index += 1
        self._step_index = 0
        self.call(
            backend=self.evaluation_backend,
            prompt=PromptBuilder.evaluation_opening(
                skill_provider=self.skill_provider,
                arm=self.prompt_arm,
                domain_description=self.domain_description,
                digest=self._digest,
            ),
            phase=AgentPhase.EVALUATION,
            kind=AgentCallKind.OPENING,
        )
        return lambda state: self.evaluation_step(state=state, task=task)

    def evaluation_step(self, *, state: State, task: Task) -> LabeledAction:
        """One evaluation decision. The agent is told the observation and nothing else.

        A named method rather than a closure, since every parameter in this project is
        keyword-only (ruff PLR0917) and a `Policy` takes its state positionally."""
        ground_skills = self.applicable_ground_skills(state=state)
        if not ground_skills:
            # The evaluation answer to a dead end, matching `EesMethod` and
            # `RandomSkillsMethod`: a no-op, because `run_task_episode` owns termination
            # and a policy must not end its caller's episode from in here.
            return LabeledAction(
                action=self.env.noop_action(), label="no-op (no applicable skills)"
            )
        observation = self.build_observation(
            state=state, goal=task.goal, ground_skills=ground_skills
        )
        labeled, _ = self.decide(
            observation=observation,
            ground_skills=ground_skills,
            state=state,
            prompt=PromptBuilder.evaluation_step(observation=observation),
            backend=self.evaluation_backend,
            phase=AgentPhase.EVALUATION,
        )
        self._step_index += 1
        return labeled

    # ---- practice ---------------------------------------------------------------

    def get_practice_policy(self, *, task: Task) -> Policy:
        """One interaction period. Starts a new conversation on the **practice** backend.

        Overridden rather than inherited because the two phases genuinely differ here, in
        the two ways that matter: this one may be told outcomes, and this one raises
        `InteractionComplete` on a dead end instead of emitting a no-op. That second
        difference is not cosmetic -- `practice_loop.py` charges one online transition per
        step and a practice period has no goal check, so a dead-ended period would burn its
        whole remaining budget on no-ops while every other arm on the same chart stops
        charging. Here it would also burn one agent call per no-op step."""
        self._pending_practice_skill = None
        self._pending_practice_label = ""
        self._previous_outcome = None
        self._step_index = 0
        self.practice_backend.reset()
        self.call(
            backend=self.practice_backend,
            prompt=PromptBuilder.practice_opening(
                skill_provider=self.skill_provider,
                arm=self.prompt_arm,
                domain_description=self.domain_description,
                digest=self._digest,
            ),
            phase=AgentPhase.PRACTICE,
            kind=AgentCallKind.OPENING,
        )
        return lambda state: self.practice_step(state=state, task=task)

    def practice_step(self, *, state: State, task: Task) -> LabeledAction:
        """One practice decision: settle the previous action, tell the agent how it went,
        and ask for the next one.

        Settles **before** the dead-end check, so a period that ends here still records
        what it last did -- the order `EesMethod` and `RandomSkillsMethod` both use."""
        self.settle_pending_practice_skill(state=state)
        ground_skills = self.applicable_ground_skills(state=state)
        if not ground_skills:
            raise InteractionComplete
        observation = self.build_observation(
            state=state, goal=task.goal, ground_skills=ground_skills
        )
        labeled, ground_skill = self.decide(
            observation=observation,
            ground_skills=ground_skills,
            state=state,
            prompt=PromptBuilder.practice_step(
                observation=observation,
                previous_outcome=self._previous_outcome,
                practice_outcomes=self._practice_tallies,
            ),
            backend=self.practice_backend,
            phase=AgentPhase.PRACTICE,
        )
        self._pending_practice_skill = ground_skill
        self._pending_practice_label = labeled.label
        self._previous_outcome = None
        self._step_index += 1
        return labeled

    def observe_environment_reset(self, *, state: State) -> None:
        """Score the in-flight skill against the state the harness is about to discard,
        rather than against the initial state it is about to be teleported to. Only
        reachable with `--practice-reset-interval` set; `EesMethod` and
        `RandomSkillsMethod` override it for the same reason."""
        self.settle_pending_practice_skill(state=state)

    def settle_pending_practice_skill(self, *, state: State) -> None:
        """Tally the previous practice step's skill against what `state` shows, clear it,
        and remember what to tell the agent. A no-op when nothing is in flight, so it is
        safe to call from both the normal step and the reset hook."""
        pending = self._pending_practice_skill
        if pending is None:
            return
        self._pending_practice_skill = None
        achieved = pending.add_effects <= self.abstract_state(state=state)
        self._previous_outcome = PromptBuilder.outcome_line(
            skill_label=self._pending_practice_label, achieved_add_effects=achieved
        )
        self.observe_practice_attempt(ground_skill=pending, achieved_add_effects=achieved)

    def observe_practice_attempt(
        self, *, ground_skill: GroundSkill, achieved_add_effects: bool
    ) -> None:
        """File one executed skill into its lifted skill's tally.

        Success is `add_effects <= true_atoms`, the same predicate `EesMethod` and
        `RandomSkillsMethod` score by, so this arm's practice rate means the same thing as
        theirs and the three can go on one chart.

        **`INFORMED` here does not mean what it means for EES, and the difference must be
        stated wherever this arm's sub-pools are quoted.** This baseline has no
        `LearnedSkillSampler` at all, so none of the four `SamplerConsultation` values
        describes it exactly. `EPSILON_RANDOM` is documented as *"a coin flip, carrying no
        belief"*, which is false -- the agent chose these numbers on purpose from its own
        notes. `NO_SAMPLER` is a property of a `param_dim == 0` skill and using it here
        would stop a reader inferring `param_dim` from it. `INFORMED` -- *"the parameters
        reflect something the classifier learned"* -- is the closest true reading with
        "the agent" for "the classifier". So the headline `num_successes/num_attempts` is
        comparable across arms and the informed/random/uninformative split is **not**;
        quote the first, never the second, when this arm is beside EES."""
        name = ground_skill.skill.name
        consultation = (
            SamplerConsultation.NO_SAMPLER
            if ground_skill.skill.param_dim == 0
            else SamplerConsultation.INFORMED
        )
        self._practice_tallies[name] = self._practice_tallies.get(
            name, SkillPracticeTally()
        ).with_attempt(success=achieved_add_effects, consultation=consultation)

    def end_cycle(self) -> None:
        """The one call that carries anything from practice into evaluation: ask the agent
        to write down what it learned, on the practice conversation.

        Called by `practice_loop.py` after the interaction period and **before** that
        cycle's evaluation sweep, so the sweep measures what the agent just learned rather
        than lagging a cycle behind.

        The in-flight skill is deliberately left unsettled at the period boundary, so this
        arm drops exactly one observation per period -- the same one `EesMethod` and
        `RandomSkillsMethod` drop, which is what keeps the three comparable."""
        _record, reply_text = self.call(
            backend=self.practice_backend,
            prompt=DIGEST_REQUEST,
            phase=AgentPhase.PRACTICE,
            kind=AgentCallKind.DIGEST,
        )
        note = reply_text.strip()
        if len(note) > self.max_digest_chars:
            # Announced rather than silently cut: an agent reading its own truncated note
            # would otherwise treat a severed sentence as something it had decided.
            note = note[: self.max_digest_chars] + "\n[note truncated here by the harness]"
        # An empty reply leaves the previous note in place rather than erasing it. A failed
        # digest query costs the agent this cycle's update; it must not cost every earlier
        # cycle's as well.
        if note:
            self._digest = note
        self._cycle_index += 1

    # ---- the agent call itself --------------------------------------------------

    def decide(
        self,
        *,
        observation: dict[str, Any],
        ground_skills: list[GroundSkill],
        state: State,
        prompt: str,
        backend: AgentBackend,
        phase: AgentPhase,
    ) -> tuple[LabeledAction, GroundSkill | None]:
        """Query the agent once and turn the reply into `(action, the ground skill it
        executes)`, or `(no-op, None)` when the reply could not be parsed.

        One body shared by both phases, deliberately: the phases differ in what goes into
        the prompt and which backend it goes to -- both of which are arguments -- and
        nowhere else. A second copy for evaluation would be a second place for the parsing
        and the ledger to drift.

        The action taken is read back off the ledger record rather than parsed again, so
        the two cannot disagree: whatever the ledger says this call decided is, by
        construction, what the environment was asked to do."""
        record, _reply_text = self.call(
            backend=backend,
            prompt=prompt,
            phase=phase,
            kind=AgentCallKind.DECISION,
            observation=observation,
            ground_skills=ground_skills,
        )
        if record.skill_index is None:
            return (
                LabeledAction(
                    action=self.env.noop_action(), label="no-op (agent reply was not a decision)"
                ),
                None,
            )
        ground_skill = ground_skills[record.skill_index]
        params = np.array(record.params, dtype=float)
        action = self.skill_provider.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(p), 2) for p in params]}"
        return LabeledAction(action=action, label=label), ground_skill

    def call(
        self,
        *,
        backend: AgentBackend,
        prompt: str,
        phase: AgentPhase,
        kind: AgentCallKind,
        observation: dict[str, Any] | None = None,
        ground_skills: list[GroundSkill] | None = None,
    ) -> tuple[AgentCallRecord, str]:
        """One agent query, timed, parsed once and journalled. Every call in the run goes
        through here, which is what makes the ledger complete by construction rather than
        by remembering to record.

        Returns the record and the **untruncated** reply, because the record's
        `reply_text` is capped for the ledger's sake and `end_cycle` needs the whole
        note."""
        started = time.monotonic()
        reply = backend.query(prompt=prompt)
        elapsed = time.monotonic() - started
        skill_index: int | None = None
        params: tuple[float, ...] = ()
        parse_error: str | None = None
        if kind is AgentCallKind.DECISION and ground_skills is not None:
            try:
                choice = DecisionParser.parse(
                    text=reply.text,
                    num_skills=len(ground_skills),
                    param_dims=[skill.skill.param_dim for skill in ground_skills],
                )
                skill_index, params = choice.skill_index, choice.params
            except DecisionParseError as exc:
                parse_error = str(exc)
        metadata = reply.metadata
        # A digest is kept whole (up to its own cap): it is the run's learning artifact
        # and the thing a write-up quotes, where a decision reply is one line of JSON and
        # only worth keeping when it is malformed.
        cap = self.max_digest_chars if kind is AgentCallKind.DIGEST else self.max_reply_chars
        record = AgentCallRecord(
            phase=phase,
            kind=kind,
            cycle_index=self._cycle_index,
            episode_index=self._episode_index if phase is AgentPhase.EVALUATION else None,
            step_index=self._step_index,
            observation_digest=(
                ObservationBuilder.digest(observation=observation)
                if observation is not None
                else ""
            ),
            reply_text=reply.text[:cap],
            skill_index=skill_index,
            params=params,
            parse_error=parse_error,
            seconds=round(elapsed, 3),
            total_cost_usd=metadata.get("total_cost_usd"),
            num_turns=metadata.get("num_turns"),
            num_tool_calls=metadata.get("num_tool_calls"),
            stop_reason=metadata.get("stop_reason"),
            query_error=metadata.get("query_error"),
        )
        self._records.append(record)
        self.journal(record=record)
        return record, reply.text

    def journal(self, *, record: AgentCallRecord) -> None:
        """Append one record to the ledger file, reopening it each time.

        Reopened per call rather than held open because a run of this baseline is hours
        long and a held handle loses everything buffered when the process dies; the open
        costs microseconds against a multi-second query. `None` disables journalling, which
        is what every test uses."""
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def call_records(self) -> list[AgentCallRecord]:
        """Every agent call this method has made, in order -- a copy, so a caller holding
        it cannot have it grow underneath them. The in-memory twin of the ledger file, and
        what the tests assert against."""
        return list(self._records)

    def practice_outcomes(self) -> dict[str, SkillPracticeTally]:
        """Per lifted skill, cumulative over the run; `method_runner.py` differences them
        per window. See `Method.practice_outcomes` and `observe_practice_attempt` for the
        one caveat about the sub-pools."""
        return dict(self._practice_tallies)

    # ---- the parts of Method this baseline does not have --------------------------

    def reset_environment(self, *, start_state: State) -> bool:
        """Always False: this baseline has no self-navigation to offer, so it declines
        rather than reporting a success it did not achieve (matching every other Method
        here)."""
        del start_state
        return False

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "PureAgentMethod.generate_train_task is unreachable: this baseline never "
            "*chooses* what to practice. It runs over a practice budget so it shares the "
            "online-transitions axis with EES, but practice_loop.py samples those tasks "
            "from Problem.tasks, never from a Method."
        )

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "PureAgentMethod.execute_setup_command is unreachable: no HumanOracle is "
            "ever used in this reproduction."
        )

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "PureAgentMethod.execute_skill is unreachable: this baseline computes its "
            "own ground skill choice directly, it never practices one."
        )

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "PureAgentMethod.improve_skill_parameters is unreachable: this baseline "
            "improves by writing itself a note, not by fitting anything. See end_cycle."
        )
