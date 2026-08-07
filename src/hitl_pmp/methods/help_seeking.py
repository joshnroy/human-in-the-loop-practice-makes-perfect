import enum

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from hitl_pmp.core.method.method import HumanHelpRequested
from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.types import State

from .stuck_detector import StuckDetector


class HelpSeekingTrigger(str, enum.Enum):
    """WHEN a `Method` asks a human to reposition it.

    A `(str, Enum)` matching this project's other flag enums (`PracticeResetPolicy`,
    `TossingRoomGoalType`) so argparse can offer the members directly as `choices`, a
    member compares equal to its own wire string, and the chosen value lands in
    `config_snapshot.json` as a readable word rather than an integer.

    Named for the *asking*, not for the human: what the human then does is
    `--human-reset-target`, a separate flag on the global CLI, because it is a property
    of the human rather than of the Method. The two axes are orthogonal and the arms
    this ladder runs are points in their product."""

    # Never ask -- the incumbent behaviour, and the default. A Method configured this
    # way holds no HelpSeekingPolicy at all (the method-CLI builds None for it), so a
    # run takes exactly the code path it took before help-seeking existed and needs no
    # HumanOracle wired. That is what makes `--ask-for-help never` byte-identical.
    NEVER = "never"
    # When practice has stopped reaching anywhere new -- see StuckDetector for what
    # that means and, more importantly, for what it excludes.
    ON_STUCK = "on-stuck"
    # On a schedule of the Method's own, independent of whether it was getting
    # anywhere: Bernoulli(1 / mean_steps_between_requests) per policy call. The control
    # for ON_STUCK -- it pays the same intervention cost at the same rate without the
    # timing carrying any information about the robot's situation.
    AT_RANDOM = "at-random"

    def __str__(self) -> str:
        return self.value


class HelpSeekingPolicy(BaseModel):
    """The reusable machinery a `Method` uses to decide when to ask for a human, and
    the wrapper that turns that decision into a `HumanHelpRequested`.

    **Method-side on purpose.** Josh's ruling, quoted verbatim: "part of 'when the agent
    asks for help' is part of the method -- that's not a baseline", and "this should all
    be programmed to happen on agent signal". (Everywhere outside that quotation this
    file says *robot*, per CLAUDE.md's naming rule: an "agent" here would mean an LLM,
    and nothing in this module is one.) The harness previously owned an equivalent
    object, watched the state stream itself and summoned the human; that is an external
    monitor noticing, which measures the monitor. `PracticeLoop` is now pure mechanism
    -- it answers a request, prices it, banks it and carries on -- and everything about
    *whether* to ask lives here.

    **How it observes, and why that is not a downgrade.** A `Method` sees the state only
    when it is asked for an action, so this wrapper observes the state *before* each
    action while the old harness observed the state *after* it. Same sequence with the
    period's opening state prepended; see `StuckDetector.observe`, which spells out why
    that makes the rule fire on the same condition at the same moment.

    **Order inside a call is load-bearing:** the ask/do-not-ask decision happens BEFORE
    the inner policy is consulted. That is what leaves `InteractionComplete` propagating
    out of the inner policy exactly as it does with no wrapper at all -- EES's own
    "nothing is applicable" behaviour is otherwise untouched -- and it is what makes a
    request cost the step the inner policy would have spent, so a Method that asks every
    call cannot spin.

    **Randomness is consumed by exactly one mode.** `AT_RANDOM` draws exactly once per
    policy call; `ON_STUCK` and `NEVER` draw nothing at all. Keeping that separation is
    what makes an on-stuck arm bit-identical whatever `--seed` it is handed, and what
    stops `--stuck-patience` shifting a random arm's stream.

    **Genuinely per-run state**, hence a real pydantic instance rather than a
    static-method container (CLAUDE.md's dividing line): it carries an RNG stream and a
    `StuckDetector`'s visited set, neither of which survives being recomputed."""

    trigger: HelpSeekingTrigger
    # How many consecutive already-visited states count as stuck. Only read under
    # ON_STUCK. The method-CLI's default is sized from the domain's own cycle length
    # rather than from this class, which has no domain to derive one from.
    stuck_patience: int = Field(default=20, ge=1)
    # Only read under AT_RANDOM: the mean gap in policy calls between requests, i.e.
    # each call asks with probability 1/this. Expressed as a period rather than a
    # probability so it is directly comparable to --max-steps-per-interaction -- at
    # equal values the arm gets about one rescue per practice period, which is the rate
    # a scheduled per-period reset would have given it for free.
    mean_steps_between_requests: int = Field(default=150, ge=1)
    seed: int = 0

    _rng: np.random.Generator = PrivateAttr()
    _detector: StuckDetector = PrivateAttr()

    @model_validator(mode="after")
    def _build_streams(self) -> "HelpSeekingPolicy":
        """Both are built here rather than by `default_factory` because both depend on
        another field (`seed`, `stuck_patience`), which a factory cannot see."""
        self._rng = np.random.default_rng(self.seed)
        self._detector = StuckDetector(patience=self.stuck_patience)
        return self

    def wrap(self, *, inner_policy: Policy) -> Policy:
        """`inner_policy`, but asking for a human first whenever this policy says to.

        A lambda around a bound method rather than a nested closure, matching how every
        other `Policy` in this project is produced (`EesMethod.get_task_policy` returns
        `lambda state: episode.step(state=state)`) -- `Policy` is one of the two
        interfaces CLAUDE.md exempts from the no-bare-function rule."""
        return lambda state: self.step(state=state, inner_policy=inner_policy)

    def step(self, *, state: State, inner_policy: Policy) -> LabeledAction:
        """One policy call: observe, decide, then either ask or delegate.

        Raises `HumanHelpRequested` when it asks -- it never returns a sentinel action,
        because a sentinel would have to be a real action the environment would then
        execute, and a rescue is deliberately not an online transition."""
        self._detector.observe(state=state)
        if self._should_ask():
            raise HumanHelpRequested
        return inner_policy(state)

    def _should_ask(self) -> bool:
        """Whether to ask right now. Consumes exactly one random draw under AT_RANDOM
        and none under ON_STUCK or NEVER -- see the class docstring for why that
        separation is load-bearing.

        Tested against AT_RANDOM/NEVER rather than for ON_STUCK so that anything
        unexpected -- a hand-built Namespace carrying a bare string, say -- degrades to
        the stuck rule rather than silently consuming randomness that another arm's
        stream is aligned against."""
        if self.trigger is HelpSeekingTrigger.AT_RANDOM:
            return bool(self._rng.random() < 1.0 / self.mean_steps_between_requests)
        if self.trigger is HelpSeekingTrigger.NEVER:
            return False
        return self._detector.is_stuck()

    def begin_period(self) -> None:
        """A new interaction period is starting. Starts a fresh stretch for the stuck
        detector: whatever the previous period left behind, the robot has not yet failed
        to make progress in *this* one.

        Deliberately does NOT touch the RNG, so a random arm's schedule is one
        continuous stream over the whole run rather than restarting per period --
        restarting it would make the number of requests a function of `--num-cycles` in
        a way the rate flag does not express."""
        self._detector.restart()

    def note_help_granted(self) -> None:
        """A rescue just happened and the environment has been written.

        Restarts the stuck detector, which is what stops a rescued robot asking again on
        the very next call forever: the human by construction puts it back somewhere it
        has already been, so every state is instantly non-novel until the visited set is
        cleared. See `StuckDetector.restart`."""
        self._detector.restart()


class HelpSeekingMixin(BaseModel):
    """The three `Method` members a help-seeking `Method` needs, in one place so a
    second one gets them for free.

    A mixin rather than a base class because it is orthogonal to what a `Method` *is*:
    it adds one optional field and overrides two concrete hooks, and a Method composes
    it by listing it alongside `Method` (`class EesMethod(HelpSeekingMixin, Method)`).
    Nothing about the hierarchy is contorted to fit it -- a Method that does not want it
    simply does not list it, and keeps `Method`'s own defaults.

    `None` means "this Method never asks", and it is the default. That is deliberately
    the *same* representation as `--ask-for-help never`: the method-CLI builds no policy
    for that flag value, so such a run holds no detector, draws no randomness, and
    returns its practice policy completely unwrapped -- structurally the pre-change code
    path rather than a configured-off version of the new one."""

    help_seeking: HelpSeekingPolicy | None = None

    def may_request_human_help(self) -> bool:
        """True exactly when a policy was configured. Read once, up front, by
        practice_loop.py's validation -- see `Method.may_request_human_help`."""
        return self.help_seeking is not None

    def observe_help_granted(self, *, state: State) -> None:
        """Restart the detector on the state the human actually left behind.

        `state` is unread: `StuckDetector.restart` deliberately forgets everything, so
        there is nothing to seed it with. It stays in the signature because it is
        `Method`'s contract and because a detector that reasoned about *where* it was
        put -- rather than only that it was moved -- would need it."""
        del state
        if self.help_seeking is not None:
            self.help_seeking.note_help_granted()

    def seeking_help(self, *, policy: Policy) -> Policy:
        """Wrap a practice policy so it can ask, and start a fresh stretch for the
        period it is about to drive.

        The per-period restart lives here rather than in a hook of its own because
        `get_practice_policy` is called exactly once per interaction period, so this
        call *is* the period boundary as the Method sees it -- and a Method cannot
        forget to announce a boundary it never has to announce.

        Returns `policy` untouched when nothing was configured, so an unwrapped run has
        no extra frame in its call stack, not merely an inert one."""
        if self.help_seeking is None:
            return policy
        self.help_seeking.begin_period()
        return self.help_seeking.wrap(inner_policy=policy)
