"""Which task family a Tossing Room `Goal.describe()` string belongs to.

**Why this is its own module, and why it is not named for an experiment.** This rule was
written out twice, character-identical, in two per-experiment Tossing Room reset-policy
reports (both since deleted by #141), and imported out of the later one by
`reset_free_ledge_curves.py`. Naming a shared helper after the one experiment that
happened to need it first is precisely what let a second copy appear unnoticed -- the same
thing that happened to `PairedTests` (see `paired_tests.py`), except that those two had
actually diverged. These had not, which is luck rather than design.

So it is named for the concern. Every Tossing Room analysis that splits results by throw
family needs exactly this rule, and there is now one place to change it.
"""

# Goal-family classification, as an ORDERED rule list, and the order is load-bearing.
# `Goal.describe()` renders the EMPTY family as
# "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)" -- it names BOTH bins, so
# a naive "does it mention recycling?" test swallows it and silently reports 16
# RECYCLING / 0 EMPTY. Matching the `BinEmpty` predicate first is what keeps the two
# throw families' denominators honest. A caller's `composition_violations` is the
# backstop: a misclassification shows up there as a wrong denominator rather than as a
# plausible wrong answer.
_FAMILY_RULES = (("BinEmpty", "EMPTY"), ("Trash", "TRASH"), ("Recycling", "RECYCLING"))


class GoalFamilies:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def classify(*, goal: str) -> str:
        """The task family one `Goal.describe()` string belongs to.

        Walks `_FAMILY_RULES` in order, so EMPTY is tested for first -- see that constant
        for why the order is the whole point. An unmatched goal raises rather than
        bucketing as "other": it means a domain change or the wrong sweep directory, and
        a silent "other" bucket would quietly shrink a denominator.
        """
        for token, family in _FAMILY_RULES:
            if token in goal:
                return family
        raise ValueError(f"unrecognised goal description: {goal!r}")
