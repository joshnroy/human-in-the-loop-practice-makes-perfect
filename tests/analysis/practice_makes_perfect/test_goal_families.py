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

# `--unsplit-skills` collapses the two throw predicates into one shared `ItemInBin` and
# the two BinEmpty predicates into one shared `BinEmpty`, so the *predicate* no longer
# names the family -- only the bound object does. Captured from a real run rather than
# written from the source: `--env tossingroom --unsplit-skills --num-cycles 0`.
_UNSPLIT_TRASH = "ItemInBin(trash, trash_bin)"
_UNSPLIT_RECYCLING = "ItemInBin(recycling, recycling_bin)"
_UNSPLIT_EMPTY = "BinEmpty(recycling_bin) & BinEmpty(trash_bin)"


def test_empty_is_classified_before_recycling_and_the_naive_rule_would_be_wrong() -> None:
    """The EMPTY goal string contains both "Recycling" and "Trash", so a rule list that
    tested either throw family first would bucket EMPTY's 2 tasks per seed into a throw
    family's denominator and report 16 RECYCLING / 0 EMPTY."""
    assert "Recycling" in _EMPTY
    assert "Trash" in _EMPTY
    assert GoalFamilies.classify(goal=_EMPTY) == "EMPTY"
    assert GoalFamilies.classify(goal=_TRASH) == "TRASH"
    assert GoalFamilies.classify(goal=_RECYCLING) == "RECYCLING"


def test_the_unsplit_skills_rendering_classifies_into_the_same_three_families() -> None:
    """`--unsplit-skills` is a flag on the same domain with the same 14/14/2 test set, so
    a per-family analysis has to work on both renderings. Before this rule existed
    `classify` raised on every throw goal of an unsplit run, which made "plot all tasks
    separately" impossible on that configuration."""
    assert GoalFamilies.classify(goal=_UNSPLIT_EMPTY) == "EMPTY"
    assert GoalFamilies.classify(goal=_UNSPLIT_TRASH) == "TRASH"
    assert GoalFamilies.classify(goal=_UNSPLIT_RECYCLING) == "RECYCLING"


def test_unsplit_empty_is_still_classified_before_either_throw_family() -> None:
    """The ordering trap survives the rename: the unsplit EMPTY string still names both
    bins, and now names them *only* in the lowercase object slots that the unsplit throw
    rules key on -- so an ItemInBin-first rule list would bucket EMPTY into a throw
    family exactly as a Recycling-first one would."""
    assert "recycling" in _UNSPLIT_EMPTY
    assert "trash" in _UNSPLIT_EMPTY
    assert GoalFamilies.classify(goal=_UNSPLIT_EMPTY) == "EMPTY"


def test_an_unrecognised_goal_raises_rather_than_bucketing_as_other() -> None:
    with pytest.raises(ValueError, match="unrecognised goal"):
        GoalFamilies.classify(goal="SomethingElse(a, b)")
