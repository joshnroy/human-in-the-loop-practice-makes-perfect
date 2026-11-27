"""A successful kitchen pour takes its dumplings out of the jug, in the forecast only.

Six finished Curtis runs of the 2026-11-11 gripper sweep logged 38 successful pours.
Each removed exactly n dumplings from the jug into the skillet, yet the forecast left
all three in the jug, so a second pour was believed possible at once. Which dumplings
leave is not determined by anything symbolic (Pour1: 3/15, 5/15, 7/15 for dumpling 0,
1, 2), so the forecast branches over every subset of n.
"""

import itertools
import json
from pathlib import Path

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.pouring3d.kitchen.predicates import SPLIT_COUNTS
from hitl_pmp.environments.pouring3d.kitchen.predicates import KitchenPredicates as P
from hitl_pmp.environments.pouring3d.kitchen.skills import KitchenSkills
from hitl_pmp.environments.pouring3d.kitchen.types import KitchenObjects as O
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    FrozenSkillEffects,
    success_effect_outcomes,
)
from hitl_pmp.planning.pddl import PddlWriter
from tests.environments.pouring3d.kitchen_symbolic import KitchenSymbols as K

LOGGED = json.loads(
    (Path(__file__).parent / "fixtures" / "kitchen_logged_successful_pours.json").read_text()
)
WHERE = ("InPitcher", "InTray", "OnFloor", "OnSurface", "HoldingShrimp")
POURS = pytest.mark.parametrize("count", SPLIT_COUNTS)


def _forecast(
    *, ground: GroundSkill, atoms: frozenset[GroundAtom]
) -> tuple[tuple[float, frozenset[GroundAtom]], ...]:
    effects = FrozenSkillEffects(
        table={g: (g.add_effects, g.delete_effects, g.ignore_effects) for g in K.ground_skills()}
    )
    ground = next(g for g in effects if g == ground)
    return success_effect_outcomes(true_atoms=atoms, ground_skill=ground, effects=effects)


def _named(*, atoms: frozenset[GroundAtom], predicates: tuple[str, ...]) -> set[str]:
    return {
        f"{a.predicate.name}({','.join(o.name for o in a.objects)})"
        for a in atoms
        if a.predicate.name in predicates
    }


def _left(*, atoms: frozenset[GroundAtom]) -> frozenset[str]:
    inside = {a.objects[0].name for a in atoms if a.predicate.name == "InPitcher"}
    return frozenset(d.name for d in K.DUMPLINGS) - inside


@POURS
def test_a_forecast_success_takes_exactly_n_dumplings_from_the_jug_to_the_skillet(
    *, count: int
) -> None:
    outcomes = _forecast(ground=K.pour(count=count), atoms=K.holding_a_full_jug())
    subsets = {
        frozenset(names) for names in itertools.combinations([d.name for d in K.DUMPLINGS], count)
    }
    assert {_left(atoms=after) for _, after in outcomes} == subsets
    assert len(outcomes) == len(subsets)
    assert [probability for probability, _ in outcomes] == pytest.approx(
        [1.0 / len(subsets)] * len(subsets)
    )
    for _, after in outcomes:
        left = _left(atoms=after)
        assert _named(atoms=after, predicates=("InTray",)) == {f"InTray({d},tray)" for d in left}
        assert _named(atoms=after, predicates=("OnFloor", "OnSurface", "HoldingShrimp")) == set()


@POURS
def test_no_pour_is_applicable_after_a_forecast_success(*, count: int) -> None:
    pours = [K.pour(count=n) for n in SPLIT_COUNTS]
    assert all(pour.preconditions <= K.holding_a_full_jug() for pour in pours)
    for _, after in _forecast(ground=K.pour(count=count), atoms=K.holding_a_full_jug()):
        assert not any(pour.preconditions <= after for pour in pours)


@POURS
def test_at_most_one_split_holds_after_a_forecast_success(*, count: int) -> None:
    stale = K.atoms(texts=[f"Split{n}()" for n in SPLIT_COUNTS])
    for _, after in _forecast(ground=K.pour(count=count), atoms=K.holding_a_full_jug() | stale):
        assert _named(atoms=after, predicates=tuple(f"Split{n}" for n in SPLIT_COUNTS)) == {
            f"Split{count}()"
        }


def test_one_split_holds_along_a_pour_a_human_pour_back_and_another_pour() -> None:
    reset = next(g for g in K.ground_skills() if g.skill.name == "human_bulk_pour_back")
    for _, poured in _forecast(ground=K.pour(count=1), atoms=K.holding_a_full_jug()):
        assert reset.preconditions <= poured
        ((_, refilled),) = _forecast(ground=reset, atoms=poured)
        assert _left(atoms=refilled) == frozenset()
        assert _named(atoms=refilled, predicates=("Split1", "Split2", "Split3")) == set()
        pick = next(g for g in K.ground_skills() if g.skill.name == "PickPitcher")
        assert pick.preconditions <= refilled
        ((_, held),) = _forecast(ground=pick, atoms=refilled)
        for _, again in _forecast(ground=K.pour(count=2), atoms=held):
            assert _named(atoms=again, predicates=("Split1", "Split2", "Split3")) == {"Split2()"}
            assert len(_left(atoms=again)) == 2


@POURS
def test_a_pour_is_still_judged_by_its_split_alone(*, count: int) -> None:
    pour = K.pour(count=count)
    assert pour.expected_add_effects(before_atoms=K.holding_a_full_jug()) == {
        K.atom(text=f"Split{count}()")
    }
    assert pour.delete_effects == frozenset()
    assert [v.name for v in pour.skill.parameters] == ["r", "p", "s0", "s1", "s2"]


def test_the_planners_pddl_does_not_read_the_alternatives() -> None:
    skills = KitchenSkills.build()
    assert any(skill.success_alternatives for skill in skills)

    def domain(*, stripped: bool) -> str:
        return PddlWriter.domain_str(
            skills=tuple(
                s.model_copy(update={"success_alternatives": ()}) if stripped else s for s in skills
            ),
            predicates=P.all(),
            types=O.types(),
        )

    assert domain(stripped=False) == domain(stripped=True)


def test_only_the_pours_declare_alternatives() -> None:
    assert {s.name for s in KitchenSkills.build() if s.success_alternatives} == {
        f"Pour{n}" for n in SPLIT_COUNTS
    }


def test_every_logged_successful_pour_is_judged_a_success_and_is_one_the_forecast_expects() -> None:
    before = K.atoms(texts=LOGGED["before"])
    assert before == K.holding_a_full_jug()
    assert len(LOGGED["successes"]) == 38
    for row in LOGGED["successes"]:
        pour = K.pour(count=int(row["skill"].removeprefix("Pour")))
        after = K.atoms(texts=row["after"])
        assert pour.expected_add_effects(before_atoms=before) <= after
        # `Settled` is compared apart: the forecast leaves it as it was (false while
        # carried), and every logged success left all three dumplings settled.
        compared = (*WHERE, "HoldingPitcher", "PickPlannable", "Split1", "Split2", "Split3")
        assert _named(atoms=after, predicates=compared) in [
            _named(atoms=forecast, predicates=compared)
            for _, forecast in _forecast(ground=pour, atoms=before)
        ]
