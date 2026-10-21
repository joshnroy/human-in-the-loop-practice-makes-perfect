"""The always-ask reset: a full human reset forced after every attempt at a skill with
continuous parameters, whenever the robot cannot plan to the goal on its own, and after
every practice session that was only a STOP.

A wrapper, not a planner. The method's own planner (EES, expectimax or determinized A*)
still chooses every robot action and learns from every outcome; this rule only inserts
a reset at three moments:

1. **Parameterized skill.** After the robot executes any skill with continuous
   parameters (`param_dim > 0`), whether it succeeded or failed. Read off the skill
   definitions, so no domain lists its trigger skills: on Tossing3D it is the toss, on
   SweepOffShelf3D the sweep.
2. **Stranded.** Whenever the goal is unreachable using the robot's own skills (no plan
   with no human reset offered). There is no cap: a reset that does not cure the
   stranding fires again on the next step, and each re-fire is logged with its run
   length so the need for a cap can be judged.
3. **STOP-only session.** After a practice session whose only act was a STOP. The
   session is over, so the step protocol executes this one at the session's end
   (`EesMethod.stop_only_session_reset`); the session stays STOP-only, so it still
   counts toward `repeated_stop`.

The reset is the human *full* reset to a freshly sampled start state
(`HumanStartResetRequested`), at the domain's existing human-reset price unless
configured otherwise.
"""

import logging

from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import GroundSkill

logger = logging.getLogger(__name__)

PARAMETERIZED_SKILL = "parameterized_skill"
STRANDED = "stranded"
STOP_ONLY_SESSION = "stop_only_session"


class AlwaysAskReset(BaseModel):
    """Owes one reset after each parameterized-skill execution, and counts what it
    forced. A method holding one calls `observe_dispatch` for every robot skill it
    dispatches while practising, `take_owed` at the top of every practice step (after the
    previous skill's outcome was observed), and `record_forced` when it asks."""

    # The price used when no practice-cost file prices the reset by name.
    cost: float = Field(ge=0.0, allow_inf_nan=False)

    _owed: bool = PrivateAttr(default=False)
    _skill_attempts: int = PrivateAttr(default=0)
    _forced: dict[str, int] = PrivateAttr(default_factory=dict)
    _stranded_run: int = PrivateAttr(default=0)
    _longest_stranded_run: int = PrivateAttr(default=0)
    _events: list[dict[str, object]] = PrivateAttr(default_factory=list)

    @staticmethod
    def for_provider(
        *, skill_provider: SkillProvider, cost: float | None = None
    ) -> "AlwaysAskReset":
        """Priced at `cost`, or by default at the domain's own human-reset price."""
        if cost is None:
            resets = skill_provider.human_cube_bin_reset_skills()
            if not resets:
                raise ValueError(
                    "this domain offers no human reset to take the price from; give the "
                    "price explicitly"
                )
            cost = resets[0].evaluate_practice_cost()
        return AlwaysAskReset(cost=cost)

    @staticmethod
    def triggers(*, ground_skill: GroundSkill) -> bool:
        """Whether executing this skill owes a reset: it has continuous parameters."""
        return ground_skill.skill.param_dim > 0

    @property
    def owed(self) -> bool:
        return self._owed

    @property
    def skill_attempts(self) -> int:
        return self._skill_attempts

    @property
    def forced(self) -> dict[str, int]:
        return dict(self._forced)

    @property
    def longest_stranded_run(self) -> int:
        return self._longest_stranded_run

    @property
    def events(self) -> list[dict[str, object]]:
        return list(self._events)

    def observe_dispatch(self, *, ground_skill: GroundSkill) -> None:
        if AlwaysAskReset.triggers(ground_skill=ground_skill):
            self._skill_attempts += 1
            self._owed = True
        # A robot skill ran, so any run of back-to-back stranded resets is over.
        self._stranded_run = 0

    def take_owed(self) -> bool:
        owed, self._owed = self._owed, False
        return owed

    def record_forced(self, *, reason: str) -> int:
        """Count a forced reset; returns how many stranded resets have now fired in a
        row (0 for a parameterized-skill reset)."""
        self._forced[reason] = self._forced.get(reason, 0) + 1
        if reason == STRANDED:
            self._stranded_run += 1
            self._longest_stranded_run = max(self._longest_stranded_run, self._stranded_run)
            if self._stranded_run > 1:
                logger.info(
                    "always-ask reset: stranded again after a reset (%d in a row)",
                    self._stranded_run,
                )
        else:
            self._stranded_run = 0
        consecutive = self._stranded_run if reason == STRANDED else 0
        self._events.append({"reason": reason, "consecutive_stranded": consecutive})
        return consecutive
