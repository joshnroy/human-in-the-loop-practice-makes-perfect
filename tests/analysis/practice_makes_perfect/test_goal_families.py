"""Tests for the Tossing Room goal-family classification rule.

These moved here with the rule itself, from the tests of the Tossing Room Split
two-way-ledge report (deleted by #141). The rule was duplicated character-identically in
two analysis modules and tested twice, once against each copy; deduplicating it leaves one
rule and one set of tests.

The ordering case is the one that matters: the EMPTY goal string names both bins, so a
rule list that tested either throw family first would silently fold EMPTY's tasks into a
throw family's denominator.
"""

import pytest

from analysis.practice_makes_perfect.goal_families import GoalFamilies

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"


def test_empty_is_classified_before_recycling_and_the_naive_rule_would_be_wrong() -> None:
    """The EMPTY goal string contains both "Recycling" and "Trash", so a rule list that
    tested either throw family first would bucket EMPTY's 2 tasks per seed into a throw
    family's denominator and report 16 RECYCLING / 0 EMPTY."""
    assert "Recycling" in _EMPTY
    assert "Trash" in _EMPTY
    assert GoalFamilies.classify(goal=_EMPTY) == "EMPTY"
    assert GoalFamilies.classify(goal=_TRASH) == "TRASH"
    assert GoalFamilies.classify(goal=_RECYCLING) == "RECYCLING"


def test_an_unrecognised_goal_raises_rather_than_bucketing_as_other() -> None:
    with pytest.raises(ValueError, match="unrecognised goal"):
        GoalFamilies.classify(goal="SomethingElse(a, b)")
