"""The always-ask rule: one human reset after every execution of the practice target.

Kept out of `_EesEpisode` on purpose. A method that owns one of these makes exactly two
calls, and those two calls are the whole rule: `observe_target_dispatched` when the skill
it chose to practise is dispatched (not the setup skills that reach its preconditions),
and `raise_if_owed` at the top of every practice step, which is also the point at which
the target's outcome has just been observed. Any practice method with a notion of "the
skill I chose to practise" can hold one -- `EesMethod` does today, and a POMDP method's
episode would make the same two calls at its own dispatch and step boundaries.
"""

from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.core.method.method import HumanStartResetRequested
from hitl_pmp.core.method.skill_provider import SkillProvider


class AlwaysResetAfterPracticeTarget(BaseModel):
    """Owes one full human reset (to a freshly sampled start state) per practice target
    executed, and asks for it at the next practice step, whatever the outcome.

    It never decides whether to ask. The debt survives the end of a session: a target
    that was a session's last action is followed by a reset as the next session's first
    step, so over a run the resets granted equal the target executions, short of one
    still owed at the end."""

    # The price used when no practice-cost file prices the reset by name.
    cost: float = Field(ge=0.0, allow_inf_nan=False)

    _owed: bool = PrivateAttr(default=False)
    _target_executions: int = PrivateAttr(default=0)
    _requests: int = PrivateAttr(default=0)

    @staticmethod
    def for_provider(
        *, skill_provider: SkillProvider, cost: float | None = None
    ) -> "AlwaysResetAfterPracticeTarget":
        """Priced at `cost`, or by default at the price the domain already charges for
        its own human reset."""
        if cost is None:
            resets = skill_provider.human_cube_bin_reset_skills()
            if not resets:
                raise ValueError(
                    "this domain offers no human reset to take the full reset's price "
                    "from; give the price explicitly"
                )
            cost = resets[0].evaluate_practice_cost()
        return AlwaysResetAfterPracticeTarget(cost=cost)

    @property
    def owed(self) -> bool:
        return self._owed

    @property
    def target_executions(self) -> int:
        return self._target_executions

    @property
    def requests(self) -> int:
        return self._requests

    def observe_target_dispatched(self) -> None:
        self._target_executions += 1
        self._owed = True

    def raise_if_owed(self, *, cost: float) -> None:
        """Ask for the reset now, if a target has executed since the last one."""
        if not self._owed:
            return
        self._owed = False
        self._requests += 1
        raise HumanStartResetRequested(cost=cost)
