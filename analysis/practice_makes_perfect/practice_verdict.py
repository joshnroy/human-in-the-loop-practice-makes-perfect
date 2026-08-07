"""The starvation-versus-inability decision rule, amended and made domain-agnostic.

**What this decides.** A learning curve says a method scored 21/100. It cannot say *why*,
and the two candidate answers need opposite fixes: the sampler was never given enough
labels (*starvation* -- buy more transitions), or it has the labels and cannot fit them
(*inability* -- change the representation). This module decides that question from one
lifted skill's practice pools: its informed draws against a control drawn from inside the
same runs, plus the trajectory those informed successes traced across the run.

**Why it was amended, which is the whole point of this module.** PR #127 introduced the
rule and PR #131 refuted one of its verdicts by direct measurement. On `tossingroomsplit`
the #127 rule assigns `ThrowRecycling` to `inability`, because 56/160 of its practice
executions were informed (a large enough *share*) and its informed draws landed 11/56,
within 0.10 of its 11/57 epsilon-random control. Ten times the budget -- same seeds, same
code, same sampler architecture, nothing changed but the number of transitions -- takes
that same sampler to **901/982, +69.73pp, p < 0.0001**. A classifier that cannot learn
does not do that. The skill was starved.

Two faults produced the false verdict, and this module fixes both:

1. **The `inability` cell had no power requirement.** A *share* of draws says nothing
   about whether there are enough of them. At `I = 56` the design's own MDE is 20.87
   points, so "the informed rate is within 10 points of the control" was satisfied by any
   sampler that had not learned *yet*, including one that was about to.
2. **There was no plateau check.** `ThrowRecycling`'s informed successes are 0, 0, 2, 3, 6
   by fifths of the standard run -- still climbing when the budget ends. A rising curve is
   the starvation signature, and the rule read it as inability.

**And a third cell, because two were never enough.** With only `starvation` and
`inability` available, a rule that cannot support either is forced to guess. `ThrowRecycling`
at the standard budget is exactly that case: the honest answer is that the budget was too
small to tell, which is `indeterminate` -- and that is a *finding* (buy more transitions
before touching the representation), not a failure to produce one. `learned` is the
fourth, for the case the #127 rule had no name for at all.

**Direction of the conservatism.** Every gate is set so that a marginal case falls to
`indeterminate` rather than to `inability`. That asymmetry is deliberate: `inability` is a
claim that a representation has to change, which is expensive and, as #131 showed, hard to
retract once it is in a merged experiment log. `indeterminate` costs a longer run.

Domain-agnostic on purpose. The #127 rule was welded to Tossing3D -- hardcoded skill names
and one hardcoded uniform-draw reference -- so it could not be run against the domain that
later falsified it, which is part of why it went unchecked for as long as it did. Every
input here is a count the caller supplies.

Counts, never bare percentages: every number in a returned reasoning string is `x/y`, with
a percentage-point gap only ever alongside the counts it came from.
"""

from collections.abc import Sequence

#: `z(0.975) + z(0.20)` = 1.959964 + 0.841621. The standard-error multiple for 80% power
#: at a two-sided 5% level -- this project's MDE convention, shared with
#: `tossingroom_goal_family_curves.py` and `tossingroomsplit_throw_rates.py`.
MDE_MULTIPLE = 2.801585


class PracticeVerdict:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    #: The margin at which the `inability` cell asserts equivalence: it fires only when the
    #: informed rate sits within this of the control rate. Carried over unchanged from
    #: #127's `INABILITY_TOLERANCE`, and now doing a second job -- see `has_power`.
    EQUIVALENCE_MARGIN = 0.10

    @staticmethod
    def noise_floor(*, successes_a: int, attempts_a: int, successes_b: int, attempts_b: int) -> float:
        """The two-proportion standard error under the pooled rate,
        `sqrt(p_bar (1 - p_bar) (1/n1 + 1/n2))`.

        Pooled rather than per-arm because the question the gate asks is the null one --
        could these two arms have come from the same rate -- which is what the `inability`
        cell is trying to assert."""
        if attempts_a <= 0 or attempts_b <= 0:
            return float("inf")
        pooled = (successes_a + successes_b) / (attempts_a + attempts_b)
        return (pooled * (1.0 - pooled) * (1.0 / attempts_a + 1.0 / attempts_b)) ** 0.5

    @staticmethod
    def minimum_detectable_effect(
        *, successes_a: int, attempts_a: int, successes_b: int, attempts_b: int
    ) -> float:
        """The smallest difference this pair of sample sizes could detect at 80% power,
        two-sided 5%. Derived from *these two* denominators, never inherited from another
        comparison -- an MDE quoted from a differently-sized pair describes a design that
        was not run."""
        return MDE_MULTIPLE * PracticeVerdict.noise_floor(
            successes_a=successes_a,
            attempts_a=attempts_a,
            successes_b=successes_b,
            attempts_b=attempts_b,
        )

    @staticmethod
    def has_power(
        *, successes_a: int, attempts_a: int, successes_b: int, attempts_b: int
    ) -> bool:
        """Whether the design can support the equivalence claim the `inability` cell makes.

        **The threshold is the cell's own assertion, not a round number.** `inability` says
        the informed rate sits within `EQUIVALENCE_MARGIN` (0.10) of the control. An
        equivalence claim at a 10-point margin means nothing unless the design could have
        *detected* 10 points -- otherwise "within 10 points" is a property of the noise
        rather than of the sampler. So the gate is `MDE <= EQUIVALENCE_MARGIN`.

        What that buys, stated as an effect size: at the threshold the cell can detect a
        10-point difference at 80% power, two-sided 5%. In symmetric arms at the worst-case
        pooled rate `p_bar = 0.5` it takes **393 informed draws against 393 control draws**
        to get there (`2.801585 * sqrt(0.5/n) <= 0.10`); at a lopsided rate it is cheaper.
        `ThrowRecycling`'s standard budget supplies 56 against 57, whose MDE is 20.87
        points -- twice the margin the cell would be asserting."""
        return (
            PracticeVerdict.minimum_detectable_effect(
                successes_a=successes_a,
                attempts_a=attempts_a,
                successes_b=successes_b,
                attempts_b=attempts_b,
            )
            <= PracticeVerdict.EQUIVALENCE_MARGIN
        )

    @staticmethod
    def is_still_rising(*, trajectory: Sequence[int]) -> bool:
        """Whether an informed-success curve was still climbing when the budget ran out:
        its final bucket is a **strict maximum** over every bucket before it.

        Stated so a reader can apply it by eye -- the curve ends on a value it has never
        reached before -- and deliberately the conservative test rather than the sensitive
        one. A flat-but-noisy curve that happens to peak in its final bucket is reported as
        still rising and falls to `indeterminate`; the alternative error, calling a still
        improving sampler unable, is the one #131 had to retract.

        Read off the **counts**, not a rate. That is what #131 measured (`ThrowRecycling`'s
        0, 0, 2, 3, 6 by fifths), and a rate over sparse early buckets is undefined exactly
        where the trend starts -- the first fifth of that run has 0 informed draws.

        Fewer than two buckets cannot show a trend in either direction, and returning
        `False` there is right: it withholds nothing, since the power gate is what stops a
        one-bucket run, and it keeps "not rising" from meaning "no data"."""
        if len(trajectory) < 2:
            return False
        return trajectory[-1] > max(trajectory[:-1])

    @staticmethod
    def classify(
        *,
        informed_successes: int,
        informed_attempts: int,
        control_successes: int,
        control_attempts: int,
        informed_success_trajectory: Sequence[int],
    ) -> tuple[str, str]:
        """(cell, the reasoning as x/y counts).

        The control is the caller's to choose, and it should come from inside the same runs
        -- that skill's own epsilon-random pool -- so it absorbs "these tasks were easy" in
        a way an analytic prior does not.

        Four cells, checked in this order:

        - **`learned`** -- the informed draws beat the control by more than this design's
          own MDE. That is self-evidencing: a gap the design detected needs no separate
          power gate.
        - **`inability`** -- consulted enough to resolve the margin it asserts, landing at
          the control rate, and no longer improving. All three, or it does not fire.
        - **`indeterminate`** -- any of those three fails, naming which.
        - **`indeterminate`** again for an empty arm, which is a real state (a skill
          practiced only before its classifier was ever fitted) rather than a rate of zero.
        """
        counts = (
            f"informed {informed_successes}/{informed_attempts}, "
            f"control {control_successes}/{control_attempts}"
        )
        if informed_attempts == 0 or control_attempts == 0:
            return (
                "indeterminate",
                f"one arm is empty, so no inference is supported: {counts}.",
            )
        gap = informed_successes / informed_attempts - control_successes / control_attempts
        mde = PracticeVerdict.minimum_detectable_effect(
            successes_a=informed_successes,
            attempts_a=informed_attempts,
            successes_b=control_successes,
            attempts_b=control_attempts,
        )
        margin = PracticeVerdict.EQUIVALENCE_MARGIN
        scale = f"{counts}, a gap of {gap * 100:+.2f}pp against an MDE of {mde * 100:.2f}pp"
        if gap > mde:
            return (
                "learned",
                f"the informed draws beat their own control by more than this design can "
                f"resolve: {scale}.",
            )
        rising = PracticeVerdict.is_still_rising(trajectory=informed_success_trajectory)
        blocked: list[str] = []
        if mde > margin:
            blocked.append(
                f"underpowered -- the MDE is {mde * 100:.2f}pp, wider than the "
                f"{margin * 100:.0f}pp margin the inability cell would assert, so landing "
                f"at the control rate is a statement about the sample size"
            )
        if abs(gap) > margin:
            blocked.append(
                f"the gap of {gap * 100:+.2f}pp is outside the {margin * 100:.0f}pp margin, "
                f"but not resolved by this design either"
            )
        if rising:
            blocked.append(
                f"informed successes are still rising -- the final bucket of "
                f"{list(informed_success_trajectory)} is higher than any before it, which "
                f"is the starvation signature"
            )
        if not blocked:
            return (
                "inability",
                f"consulted enough to resolve the margin, landing at the control rate, and "
                f"no longer improving: {scale}.",
            )
        return ("indeterminate", f"{'; '.join(blocked)}. {scale}.")
