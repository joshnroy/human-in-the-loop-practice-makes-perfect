from typing import Any

import numpy as np
from pydantic import PrivateAttr, model_validator

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
from hitl_pmp.methods.pure_agent.authored_policy import AuthoredPolicy, AuthoredPolicyError
from hitl_pmp.methods.pure_agent.observation import ObservationBuilder
from hitl_pmp.methods.pure_agent.prompts import PromptArm, PromptBuilder
from hitl_pmp.methods.pure_agent.types import AuthoringRound, AuthoringTranscript
from hitl_pmp.planning.grounding import SkillGrounder


class PureAgentMethod(Method):
    """The **pure agent** baseline: step 2.5 of Tom Silver's recipe, which slots between
    the random baseline (step 2) and the privileged oracle (step 3).

    **The agent writes the policy instead of being the policy.** The blog proposes
    querying an agent inside `step()`; `prpl-agent-utils`' own worked example
    (`examples/pendulum_pure_agent.ipynb`) refuses that and says why -- at 20 Hz a
    Pendulum episode is 200 actions and an evaluation is 4000, and no agent closes that
    loop at the necessary rate or price. The same arithmetic is worse here: see
    `num_decisions` for what a be-the-policy variant of *this* domain would cost. So the
    agent authors a `policy.py` in its sandbox, that file is loaded and evaluated, and
    the score is handed back for it to revise. **No API call happens at decision time.**

    **How it maps onto this harness, with no `core/` change.**

    - `end_cycle()` is the seam. `PracticeLoop` calls it after each interaction period
      and *before* that cycle's evaluation sweep, which is exactly the notebook's
      revise-then-score cadence.
    - `practice_outcomes()` is the feedback payload -- per-lifted-skill executions and
      how many achieved that skill's own declared add effects.
    - `get_task_policy()` returns a `Policy` that calls the authored function.

    Round 0 is authored **lazily, on first use**, which is the initial evaluation sweep.
    So a run with `--num-cycles N` authors N+1 policies, and sweep 0 measures the agent's
    zero-feedback policy -- what it can write from the symbolic layer alone, before
    seeing any outcome. That is the scientifically interesting point on the curve, not a
    throwaway: it separates "read the domain and solved it" from "needed the feedback".

    **Domain-agnostic through `SkillProvider`**, the same interface `RandomSkillsMethod`
    acts through. There is no `isinstance` dispatch and no environment import anywhere in
    this subpackage: the prompt, the observation and the action all go through the
    injected provider, so this one class runs on any `--env`.

    **It selects ground skills, not raw actions**, because the comparison of interest is
    against EES, which selects ground skills. An authored policy picks an index into the
    *applicable* ground skills at this state, so it cannot select something whose
    preconditions fail; what it must get right is which one, and what continuous
    parameters to give it.

    **Record-then-replay.** Authoring is nondeterministic and must not happen inside a
    measured run. An authoring run (`backend=...`) queries the agent and writes an
    `AuthoringTranscript`; every measured run (`replay_sources=...`) replays that
    transcript's per-round sources, makes no API call at all, and produces a byte-stable
    `stats.json`. Exactly one of the two must be given. It also resolves the sweep-width
    problem: authoring is serial, replay parallelises across seeds.

    **The security boundary, stated plainly.** The sandbox protects authoring, not
    evaluation: `AuthoredPolicy` executes agent-written code in this process. See its
    docstring. Do not replay a transcript you did not author without reading it.

    **What `practice_outcomes` means here, and what it does not.** This arm has no
    `LearnedSkillSampler`, so `SamplerConsultation`'s pool split does not carry its usual
    meaning. A `param_dim == 0` skill is filed `NO_SAMPLER`, which is exactly true. A
    parameterized skill is filed `INFORMED`, on the reading that its parameters "reflect
    something the method learned" -- the authored policy *is* what this method learned.
    **Only the overall `num_successes/num_attempts` is comparable to EES's**; the pools
    are not, and a chart putting this arm's `informed` bucket beside EES's would be
    comparing a classifier's argmax against a line of hand-written arithmetic."""

    skill_provider: SkillProvider
    prompt_arm: PromptArm = PromptArm.MINIMAL
    # Only read on the DESCRIBED arm; supplied by `--pure-agent-domain-description`,
    # which reads it from a file. A plain string here rather than a path, so this class
    # stays domain-agnostic and testable without a filesystem.
    domain_description: str = ""
    # Authoring mode. Mutually exclusive with `replay_sources` -- see the validator.
    backend: AgentBackend | None = None
    # Replay mode: one `policy.py` source per authoring round, in order, `None` for a
    # round the agent delivered no file for.
    replay_sources: tuple[str | None, ...] | None = None

    # How many rounds have been authored/replayed so far; also the index of the next one.
    _round_index: int = PrivateAttr(default=0)
    # False until the first round has been triggered -- see `_ensure_policy`.
    _started: bool = PrivateAttr(default=False)
    # The policy currently in effect. Stays at the last USABLE one across a failed round,
    # which is what makes a bad revision cost a cycle rather than the whole run.
    _policy: AuthoredPolicy | None = PrivateAttr(default=None)
    _rounds: list[AuthoringRound] = PrivateAttr(default_factory=list)
    _practice_tallies: dict[str, SkillPracticeTally] = PrivateAttr(default_factory=dict)
    _pending_practice_skill: GroundSkill | None = PrivateAttr(default=None)
    _num_decisions: int = PrivateAttr(default=0)
    _num_malformed_decisions: int = PrivateAttr(default=0)
    _last_observation: dict[str, Any] | None = PrivateAttr(default=None)
    # Practice-side goal signal, the only outcome number this Method can honestly observe
    # for itself: it holds the practice task, so it can see that task's goal become
    # satisfied. It never observes an EVALUATION outcome -- `Metrics` owns those and a
    # Method is not shown them, which is what keeps a Method from being able to train on
    # the test set even by accident.
    _num_practice_periods: int = PrivateAttr(default=0)
    _num_practice_goals_reached: int = PrivateAttr(default=0)
    _practice_goal_reached_this_period: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _check_exactly_one_source_of_policies(self) -> "PureAgentMethod":
        if (self.backend is None) == (self.replay_sources is None):
            raise ValueError(
                "PureAgentMethod needs exactly one of `backend` (author, spends money, "
                "nondeterministic) and `replay_sources` (replay a recorded transcript, "
                "free, deterministic). Got "
                f"backend={self.backend is not None}, "
                f"replay_sources={self.replay_sources is not None}."
            )
        if self.prompt_arm is PromptArm.DESCRIBED and not self.domain_description.strip():
            # Refused rather than degraded: a DESCRIBED run with no description is the
            # MINIMAL arm wearing the other arm's label, and its results would be pooled
            # into the wrong column by every reader downstream, including
            # config_snapshot.json.
            raise ValueError(
                "prompt_arm=described requires a non-empty domain description "
                "(--pure-agent-domain-description). Without one this arm is byte-for-byte "
                "the minimal arm, and pooling the two would be the whole comparison lost."
            )
        return self

    # ---------------------------------------------------------------- authoring

    def _ensure_policy(self) -> None:
        """Author (or replay) round 0 the first time anything asks for a policy.

        Lazy rather than in `model_post_init` so that constructing a `PureAgentMethod` --
        which `--help`, a config snapshot and every test do -- never fires an API call."""
        if not self._started:
            self._started = True
            self._author_round()

    def end_cycle(self) -> None:
        """One revision per cycle. See the class docstring: this is the seam."""
        self._ensure_policy()
        self._author_round()

    def _author_round(self) -> None:
        index = self._round_index
        self._round_index += 1
        previous_error = self._previous_error()
        if self.replay_sources is not None:
            source, prompt, reply_text, metadata = self._replayed_round(index=index)
        else:
            prompt = self._prompt(index=index, previous_error=previous_error)
            assert self.backend is not None  # guaranteed by the validator
            reply = self.backend.query(prompt=prompt)
            source = self.backend.policy_source()
            reply_text, metadata = reply.text, reply.metadata
        load_error = self._install(source=source, index=index)
        self._rounds.append(
            AuthoringRound(
                round_index=index,
                prompt=prompt,
                policy_source=source,
                load_error=load_error,
                agent_text=reply_text,
                total_cost_usd=metadata.get("total_cost_usd"),
                num_turns=metadata.get("num_turns"),
                num_tool_calls=metadata.get("num_tool_calls"),
            )
        )
        # A fresh round starts each period's practice-goal bookkeeping clean, so the
        # feedback describes the period just finished rather than the run so far.
        self._practice_goal_reached_this_period = False

    def _replayed_round(self, *, index: int) -> tuple[str | None, str, str, dict[str, Any]]:
        """This round's recorded source, and empty stand-ins for everything a replay does
        not have (there was no query, so there is no prompt, no reply and no cost).

        Running off the end RAISES rather than holding the last round's policy. A
        truncated artifact would otherwise evaluate a fully-revised policy at every
        remaining checkpoint, flattening the curve into something indistinguishable from a
        method that converged early -- the same argument the environment's weight schedule
        makes for never wrapping."""
        assert self.replay_sources is not None
        if index >= len(self.replay_sources):
            raise RuntimeError(
                f"replay round {index} has no recorded source: the transcript holds "
                f"{len(self.replay_sources)} rounds, and a run with --num-cycles N needs "
                "N+1 (round 0 is authored before the first evaluation sweep). Re-author, "
                "or run with fewer cycles."
            )
        return self.replay_sources[index], "(replay: no query was made)", "", {}

    def _prompt(self, *, index: int, previous_error: str | None) -> str:
        if index == 0:
            return PromptBuilder.initial(
                skill_provider=self.skill_provider,
                arm=self.prompt_arm,
                domain_description=self.domain_description,
            )
        return PromptBuilder.revision(
            practice_outcomes=self.practice_outcomes(),
            num_practice_goals_reached=self._num_practice_goals_reached,
            num_practice_periods=self._num_practice_periods,
            previous_error=previous_error,
        )

    def _install(self, *, source: str | None, index: int) -> str | None:
        """Compile this round's source and make it the policy in effect, or report why
        not. A failed round leaves the previous policy in place."""
        if source is None:
            return "policy.py was not created"
        try:
            self._policy = AuthoredPolicy(source=source, round_index=index)
        except AuthoredPolicyError as exc:
            return str(exc)
        return None

    def _previous_error(self) -> str | None:
        """Why the last round's policy is unusable, if it is -- the thing the next prompt
        quotes back. Covers both a file that never loaded and one that loaded and then
        raised on its first real call."""
        if not self._rounds:
            return None
        last = self._rounds[-1]
        if last.load_error is not None:
            return last.load_error
        return self._policy.failure() if self._policy is not None else None

    def _record_call_failure(self) -> None:
        """Back-fill the transcript when the policy in effect fails at call time.

        Filed against the round that AUTHORED that policy, not against the latest round:
        after a failed revision the policy in effect is an older one, and blaming the
        newer round for the older policy's crash would put the error in front of the
        agent attached to code it did not just write."""
        policy = self._policy
        if policy is None:
            return
        failure = policy.failure()
        if failure is None:
            return
        for position, round_ in enumerate(self._rounds):
            if round_.round_index == policy.round_index and round_.load_error is None:
                self._rounds[position] = round_.model_copy(update={"load_error": failure})
                return

    # ---------------------------------------------------------------- acting

    def applicable_ground_skills(self, *, state: State) -> list[GroundSkill]:
        """Everything whose preconditions hold in `state`. Empty means a dead end.

        Identical to `RandomSkillsMethod`'s, and deliberately so: the two baselines must
        be offered the same choice set at every state, or a difference between them is a
        difference in what they were allowed to do rather than in what they chose."""
        provider = self.skill_provider
        objects = provider.objects()
        true_atoms = SkillGrounder.abstract_state(
            state=state, objects=objects, predicates=provider.predicates()
        )
        return SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=objects, true_atoms=true_atoms
        )

    def abstract_state(self, *, state: State) -> frozenset[GroundAtom]:
        provider = self.skill_provider
        return SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )

    def choose_ground_skill(
        self, *, state: State, goal: Goal
    ) -> tuple[LabeledAction, GroundSkill | None]:
        """`(action, the ground skill it executes)`, or `(no-op, None)`.

        Counts one decision per call, BEFORE the dead-end check, because the count exists
        to price a be-the-policy variant and that variant would be queried at a dead end
        too -- it has no way to know it is at one without asking."""
        self._num_decisions += 1
        ground_skills = self.applicable_ground_skills(state=state)
        if not ground_skills:
            return self._no_op(reason="no applicable skills"), None
        observation = ObservationBuilder.build(
            state=state,
            goal=goal,
            atoms=self.abstract_state(state=state),
            objects=self.skill_provider.objects(),
            ground_skills=ground_skills,
        )
        self._last_observation = observation
        policy = self._policy
        if policy is None:
            return self._no_op(reason="no authored policy yet"), None
        choice = policy.choose(observation=observation)
        if choice is None:
            # Deterministic, and deliberately NOT a random draw: an RNG stream in this
            # path would be consumed a different number of times depending on how often
            # the authored policy misbehaved, which is exactly the byte-stability the
            # replay design turns on.
            self._num_malformed_decisions += 1
            self._record_call_failure()
            return self._no_op(reason="authored policy returned no usable decision"), None
        ground_skill = ground_skills[choice.skill_index]
        params = np.array(choice.params, dtype=float)
        action = self.skill_provider.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(p), 2) for p in params]}"
        return LabeledAction(action=action, label=label), ground_skill

    def _no_op(self, *, reason: str) -> LabeledAction:
        return LabeledAction(action=self.env.noop_action(), label=f"no-op ({reason})")

    def get_task_policy(self, *, task: Task) -> Policy:
        """Evaluation: run the authored function, learn nothing.

        The goal IS consulted, unlike `RandomSkillsMethod`'s -- a goal-dependent domain
        (Tossing Room, whose state cannot distinguish throw-recycling from throw-trash)
        is unsolvable without it, and the authored policy is handed it in the
        observation."""
        self._ensure_policy()
        goal = task.goal
        return lambda state: self.choose_ground_skill(state=state, goal=goal)[0]

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Practice: the same authored function, plus the bookkeeping the next revision
        is written from.

        Overridden rather than inherited for the same reason `RandomSkillsMethod`
        overrides it: on a dead end this raises `InteractionComplete` instead of emitting
        a no-op, so a dead-ended period stops being charged transitions it is not using.
        Both arms are plotted against that same axis."""
        self._ensure_policy()
        self._pending_practice_skill = None
        self._num_practice_periods += 1
        self._practice_goal_reached_this_period = False
        goal = task.goal
        return lambda state: self.practice_step(state=state, goal=goal)

    def practice_step(self, *, state: State, goal: Goal) -> LabeledAction:
        """`get_practice_policy`'s body -- a named method rather than a closure, since
        every parameter in this project is keyword-only and a `Policy` takes its state
        positionally.

        Settles the previous step's skill BEFORE the dead-end check, so a period that ends
        here still records what it last did -- the order `RandomSkillsMethod` and
        `_EesEpisode.step` both use."""
        self.settle_pending_practice_skill(state=state)
        self._observe_practice_goal(state=state, goal=goal)
        if not self.applicable_ground_skills(state=state):
            self._num_decisions += 1
            raise InteractionComplete
        labeled, ground_skill = self.choose_ground_skill(state=state, goal=goal)
        self._pending_practice_skill = ground_skill
        return labeled

    def _observe_practice_goal(self, *, state: State, goal: Goal) -> None:
        """Count this period as having reached its train task's goal, at most once.

        The only outcome signal this Method gives itself, and it is a PRACTICE one: the
        train task's goal, seen in the practice environment. Evaluation outcomes belong
        to `Metrics` and are never shown to a Method, which is what makes training on the
        test set structurally impossible rather than merely discouraged."""
        if self._practice_goal_reached_this_period:
            return
        if goal.is_satisfied(state=state):
            self._practice_goal_reached_this_period = True
            self._num_practice_goals_reached += 1

    def observe_environment_reset(self, *, state: State) -> None:
        """Score the in-flight skill against the state the harness is about to discard.
        Same reason as `RandomSkillsMethod`'s: without it, a mid-period reset records a
        spurious failure once per reset, and the mislabelling scales with reset
        frequency."""
        self.settle_pending_practice_skill(state=state)

    def settle_pending_practice_skill(self, *, state: State) -> None:
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
        """File one executed skill into its lifted skill's tally. Success is
        `add_effects <= true_atoms`, the same predicate `EesMethod` and
        `RandomSkillsMethod` score by, so all three arms' practice rates mean the same
        thing and can be put on one chart. See the class docstring for what the
        `SamplerConsultation` pools do and do not mean on this arm."""
        name = ground_skill.skill.name
        consultation = (
            SamplerConsultation.NO_SAMPLER
            if ground_skill.skill.param_dim == 0
            else SamplerConsultation.INFORMED
        )
        self._practice_tallies[name] = self._practice_tallies.get(
            name, SkillPracticeTally()
        ).with_attempt(success=ground_skill.add_effects <= true_atoms, consultation=consultation)

    def practice_outcomes(self) -> dict[str, SkillPracticeTally]:
        """Per lifted skill, cumulative over the run; `method_runner.py` differences them
        per window. A copy, so a caller holding last cycle's reading to difference against
        cannot have it mutated underneath by the next practice step."""
        return dict(self._practice_tallies)

    # ---------------------------------------------------------------- reporting

    def authoring_transcript(self) -> AuthoringTranscript:
        """Everything this run authored, plus the two decision counts. This is the
        artifact a later replay is driven from -- see `AuthoringTranscript`."""
        self._record_call_failure()
        return AuthoringTranscript(
            rounds=list(self._rounds),
            num_decisions=self._num_decisions,
            num_malformed_decisions=self._num_malformed_decisions,
        )

    def num_decisions(self) -> int:
        """Every point at which the harness asked this Method for an action, over both
        phases and every evaluation sweep.

        **This is the price of the be-the-policy arm**, not of this one. This arm queries
        the agent once per authoring round -- a handful of times per run. A variant that
        queried inside the policy, as the blog proposes, would make exactly this many API
        calls. Recorded so that arm can be priced from a run that has already happened."""
        return self._num_decisions

    def num_malformed_decisions(self) -> int:
        """Decisions the authored policy returned that could not be executed, each of
        which became a no-op. See `AuthoringTranscript.num_malformed_decisions`."""
        return self._num_malformed_decisions

    def last_observation(self) -> dict[str, Any] | None:
        """The most recent observation handed to the authored policy. Kept so a failure
        can be reproduced from the transcript, and so the observation contract is
        assertable from a test without reaching inside the authored module."""
        return self._last_observation

    # ---------------------------------------------------------------- unreachable

    def reset_environment(self, *, start_state: State) -> bool:
        """Always False: this baseline has no self-navigation to offer, so it declines
        rather than reporting a success it did not achieve (matches every other Method
        here). Satisfying this by calling `env.set_state` would be a privileged external
        state write dressed up as the agent recovering under its own power."""
        del start_state
        return False

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "PureAgentMethod.generate_train_task is unreachable: this baseline never "
            "chooses what to practice -- practice_loop.py samples those tasks from "
            "Problem.tasks."
        )

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "PureAgentMethod.execute_setup_command is unreachable: no HumanOracle is "
            "used by this baseline (v0 has no human help)."
        )

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "PureAgentMethod.execute_skill is unreachable: this baseline computes its "
            "own ground skill choice directly, it never practices one."
        )

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "PureAgentMethod.improve_skill_parameters is unreachable: this baseline's "
            "only learning happens in end_cycle, where the agent rewrites the whole "
            "policy rather than adjusting one skill's parameters."
        )
