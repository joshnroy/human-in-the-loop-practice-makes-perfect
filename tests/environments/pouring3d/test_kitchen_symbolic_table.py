"""The kitchen's predicates, skills and goals are the table Josh approved on 2026-11-06.

The legacy shrimp benchmark keeps its own vocabulary, pinned by digest below. The three
dumplings stay skill parameters (s0, s1, s2) under the constant `PayloadOrder` chain, and
the skillet stays the parameter `t` wherever `InTray` appears: the framework binds no
fixed object in a precondition (Josh chose this fallback, 2026-11-06).
"""

import hashlib
import itertools

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.predicates import KitchenPredicates as P
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.kitchen.tasks import KitchenTasks
from hitl_pmp.environments.pouring3d.kitchen.types import KitchenObjects
from hitl_pmp.environments.pouring3d.practice.predicates import PourPredicates
from hitl_pmp.environments.pouring3d.practice.skills import PourSkills
from hitl_pmp.environments.pouring3d.practice.types import PourObjects

Atoms = frozenset[tuple[str, tuple[str, ...]]]
EACH = ("s0", "s1", "s2")
SPLITS = {"Split1", "Split2", "Split3"}
CHAIN = {("PayloadOrder", ("s0", "s1")), ("PayloadOrder", ("s1", "s2"))}


def _lifted(*, atoms) -> Atoms:  # noqa: ANN001 -- LiftedAtoms
    return frozenset((a.predicate.name, tuple(v.name for v in a.variables)) for a in atoms)


def _skill(*, name: str):  # noqa: ANN202 -- Skill
    return next(s for s in KitchenSkillProvider.domain(count=3) if s.name == name)


def _each(*, name: str, rest: tuple[str, ...] = ()) -> set[tuple[str, tuple[str, ...]]]:
    return {(name, (s, *rest)) for s in EACH}


def test_the_predicates_are_the_table():
    assert {p.name: tuple(t.name for t in p.types) for p in P.all()} == {
        "HandEmpty": ("pour_robot",),
        "HoldingPitcher": ("pour_robot", "pitcher"),
        "HoldingShrimp": ("pour_robot", "shrimp"),
        "InPitcher": ("shrimp", "pitcher"),
        "InTray": ("shrimp", "pour_tray"),
        "OnFloor": ("shrimp",),
        "OnSurface": ("shrimp",),
        "Settled": ("shrimp",),
        "PitcherHome": ("pitcher",),
        "Split1": (),
        "Split2": (),
        "Split3": (),
        "PickPlannable": ("pour_robot", "shrimp"),
        # The fallback's constant chain over the dumpling parameters.
        "PayloadOrder": ("shrimp", "shrimp"),
    }


def test_the_kitchen_has_no_request_objects():
    objects = KitchenEnvironment().objects()
    assert objects.requests() == ()
    assert not any(o.type.name == "pour_request" for o in objects.all())
    assert "pour_request" not in {t.name for t in KitchenObjects.types()}


TABLE = {
    "PickPitcher": (
        ("r", "p", *EACH),
        {("HandEmpty", ("r",)), ("PitcherHome", ("p",)), *_each(name="InPitcher", rest=("p",))}
        | _each(name="Settled")
        | CHAIN,
        {("HoldingPitcher", ("r", "p"))},
        {("HandEmpty", ("r",)), ("PitcherHome", ("p",))},
        set(),
    ),
    **{
        f"Pour{n}": (
            ("r", "p", *EACH),
            {("HoldingPitcher", ("r", "p")), *_each(name="InPitcher", rest=("p",))} | CHAIN,
            {(f"Split{n}", ())},
            set(),
            {"InPitcher", "InTray", "OnFloor", "OnSurface", "Settled", *SPLITS, "PickPlannable"},
        )
        for n in (1, 2, 3)
    },
    "PlacePitcherAtHome": (
        ("r", "p"),
        {("HoldingPitcher", ("r", "p"))},
        {("HandEmpty", ("r",)), ("PitcherHome", ("p",))},
        {("HoldingPitcher", ("r", "p"))},
        set(),
    ),
    "PickShrimpFromTray": (
        ("r", "s", "t"),
        {("HandEmpty", ("r",)), ("InTray", ("s", "t")), ("PickPlannable", ("r", "s"))},
        {("HoldingShrimp", ("r", "s"))},
        {("HandEmpty", ("r",)), ("InTray", ("s", "t"))},
        SPLITS,
    ),
    **{
        f"PickShrimpFrom{where}": (
            ("r", "s"),
            {("HandEmpty", ("r",)), (f"On{where}", ("s",)), ("PickPlannable", ("r", "s"))},
            {("HoldingShrimp", ("r", "s"))},
            {("HandEmpty", ("r",)), (f"On{where}", ("s",))},
            SPLITS,
        )
        for where in ("Floor", "Surface")
    },
    "DropShrimpIntoPitcher": (
        ("r", "s", "p"),
        {("HoldingShrimp", ("r", "s")), ("PitcherHome", ("p",))},
        {("HandEmpty", ("r",)), ("InPitcher", ("s", "p")), ("Settled", ("s",))},
        {("HoldingShrimp", ("r", "s"))},
        SPLITS,
    ),
    HUMAN_RESET: (
        ("r", "p", "t", *EACH),
        CHAIN,
        {("HandEmpty", ("r",)), ("PitcherHome", ("p",)), *_each(name="InPitcher", rest=("p",))}
        | _each(name="Settled"),
        {("HoldingPitcher", ("r", "p")), *(("HoldingShrimp", ("r", s)) for s in EACH)}
        | _each(name="InTray", rest=("t",))
        | _each(name="OnFloor")
        | _each(name="OnSurface")
        | {(name, ()) for name in SPLITS},
        {"PickPlannable"},
    ),
}


def test_the_skill_set_is_the_table_and_nothing_else():
    assert [s.name for s in KitchenSkillProvider.domain(count=3)] == list(TABLE)


@pytest.mark.parametrize("name", list(TABLE))
def test_each_skill_is_its_row(*, name):
    skill = _skill(name=name)
    parameters, pre, add, delete, ignore = TABLE[name]
    assert tuple(v.name for v in skill.parameters) == parameters
    assert _lifted(atoms=skill.preconditions) == pre
    assert _lifted(atoms=skill.add_effects) == add
    assert _lifted(atoms=skill.delete_effects) == delete
    assert {p.name for p in skill.ignore_effects} == ignore
    assert not skill.conditional_add_effects
    assert not skill.conditional_ignore_effects
    assert skill.param_dim == (1 if name.startswith("Pour") else 0)


def _names(*, atoms) -> Atoms:  # noqa: ANN001 -- GroundAtoms
    return frozenset((a.predicate.name, tuple(o.name for o in a.objects)) for a in atoms)


def _apply(*, ground, atoms: Atoms) -> Atoms:  # noqa: ANN001 -- GroundSkill
    assert _names(atoms=ground.preconditions) <= atoms, ground.skill.name
    ignored = {p.name for p in ground.ignore_effects}
    kept = {a for a in atoms - _names(atoms=ground.delete_effects) if a[0] not in ignored}
    return frozenset(kept | _names(atoms=ground.add_effects))


START: Atoms = frozenset({
    ("HandEmpty", ("robot",)),
    ("PitcherHome", ("pitcher",)),
    *(("InPitcher", (f"potsticker_{i}", "pitcher")) for i in range(3)),
    *(("Settled", (f"potsticker_{i}",)) for i in range(3)),
    *(("PickPlannable", ("robot", f"potsticker_{i}")) for i in range(3)),
    ("PayloadOrder", ("potsticker_0", "potsticker_1")),
    ("PayloadOrder", ("potsticker_1", "potsticker_2")),
})


@pytest.mark.parametrize("n", [1, 2, 3])
def test_pick_pour_place_reaches_the_task_goal_symbolically(*, n):
    env = KitchenEnvironment(requested_count=n)
    provider = KitchenSkillProvider(env=env)
    goal = _names(atoms=KitchenTasks(env=env).goal().atoms)
    atoms = START
    for name in ("PickPitcher", f"Pour{n}", "PlacePitcherAtHome"):
        atoms = _apply(ground=provider.ground(name=name), atoms=atoms)
    assert goal <= atoms
    assert {a for a in atoms if a[0] in SPLITS} == {(f"Split{n}", ())}


def test_after_a_pour_the_pick_back_and_drop_skills_chain_to_the_reset_state():
    provider = KitchenSkillProvider(env=KitchenEnvironment(requested_count=1))
    atoms = _apply(ground=provider.ground(name="PickPitcher"), atoms=START)
    atoms = _apply(ground=provider.ground(name="Pour1"), atoms=atoms)
    atoms = _apply(ground=provider.ground(name="PlacePitcherAtHome"), atoms=atoms)
    # What the next observation would measure: one dumpling in the skillet.
    atoms = (atoms - {("Split1", ())}) | {
        ("InTray", ("potsticker_0", "tray")),
        ("PickPlannable", ("robot", "potsticker_0")),
    }
    atoms = _apply(ground=provider.ground(name="PickShrimpFromTray", shrimp=0), atoms=atoms)
    assert ("HoldingShrimp", ("robot", "potsticker_0")) in atoms
    atoms = _apply(ground=provider.ground(name="DropShrimpIntoPitcher", shrimp=0), atoms=atoms)
    assert {("HandEmpty", ("robot",)), ("InPitcher", ("potsticker_0", "pitcher"))} <= atoms
    assert ("Settled", ("potsticker_0",)) in atoms


def test_the_human_reset_from_any_state_restores_the_pour_start():
    provider = KitchenSkillProvider(env=KitchenEnvironment())
    human = provider.human_cube_bin_reset_skill()
    assert human is not None
    assert human.skill.name == HUMAN_RESET
    messy = frozenset({
        ("HoldingPitcher", ("robot", "pitcher")),
        ("OnFloor", ("potsticker_0",)),
        ("InTray", ("potsticker_1", "tray")),
        ("OnSurface", ("potsticker_2",)),
        ("Split2", ()),
        ("PayloadOrder", ("potsticker_0", "potsticker_1")),
        ("PayloadOrder", ("potsticker_1", "potsticker_2")),
    })
    after = _apply(ground=human, atoms=messy)
    assert after == START - {a for a in START if a[0] == "PickPlannable"}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_task_goal_is_its_split_with_the_hand_empty_and_the_jug_home(*, n):
    env = KitchenEnvironment(requested_count=n)
    tasks = KitchenTasks(env=env, requested_counts=(1, 2, 3))
    o = env.objects()
    expected = frozenset({
        GroundAtom(predicate=P.get(name=f"Split{n}"), objects=()),
        GroundAtom(predicate=P.get(name="HandEmpty"), objects=(o.robot,)),
        GroundAtom(predicate=P.get(name="PitcherHome"), objects=(o.pitcher,)),
    })
    assert tasks.goal().atoms == expected
    assert tasks.goal(requested=n).atoms == expected
    assert [_names(atoms=g.atoms) for g in tasks.test_task_goals()] == [
        {(f"Split{k}", ()), ("HandEmpty", ("robot",)), ("PitcherHome", ("pitcher",))}
        for k in (1, 2, 3)
    ]


# The measured split, hand and vessels: every combination the legacy backend can emit.
def _legacy_states() -> list[Atoms]:
    states = []
    for split, hand, jug, tray in itertools.product(
        (None, 1, 2, 3), ("empty", "jug", "dumpling"), (True, False), (True, False)
    ):
        atoms: set[tuple[str, tuple[str, ...]]] = {
            ("ActiveRequest", ("request_2",)),
            *((f"Count{k}", (f"request_{k}",)) for k in (1, 2, 3)),
        }
        atoms |= {
            "empty": {("HandEmpty", ("robot",))},
            "jug": {("HoldingPitcher", ("robot", "pitcher"))},
            "dumpling": {("HoldingShrimp", ("robot", "potsticker_0"))},
        }[hand]
        if jug:
            atoms.add(("PitcherHome", ("pitcher",)))
        if tray:
            atoms.add(("TrayHome", ("tray",)))
        if split is not None and hand != "dumpling":
            atoms.add(("SplitCorrect", (f"request_{split}",)))
            if hand == "empty" and jug and tray:
                atoms.add(("Goal", (f"request_{split}",)))
        states.append(frozenset(atoms))
    return states


@pytest.mark.parametrize("n", [1, 2, 3])
def test_solved_means_what_the_legacy_goal_atom_meant_on_the_same_states(*, n):
    goal = _names(atoms=KitchenTasks(env=KitchenEnvironment(requested_count=n)).goal().atoms)
    checked = 0
    for legacy in _legacy_states():
        if ("TrayHome", ("tray",)) not in legacy:
            continue  # The welded skillet is always home (`test_kitchen_skillet_weld.py`).
        old = ("Goal", (f"request_{n}",)) in legacy
        new = goal <= P.translate(atoms=legacy)
        assert old == new, sorted(legacy)
        checked += 1
    assert checked == 4 * 3 * 2


def test_the_translation_drops_every_removed_atom_and_renames_the_split():
    legacy = _legacy_states()
    removed = {
        "TrayHome",
        "ActiveRequest",
        "Count1",
        "Count2",
        "Count3",
        "SplitCorrect",
        "Goal",
        "NeedsReset",
        "NoSpills",
        "ReadyToPour",
    }
    for atoms in legacy:
        translated = P.translate(
            atoms=atoms | {("NeedsReset", ()), ("NoSpills", ()), ("ReadyToPour", ())}
        )
        assert not {a[0] for a in translated} & removed
        assert {a for a in translated if a[0] in SPLITS} == {
            (f"Split{a[1][0].removeprefix('request_')}", ())
            for a in atoms
            if a[0] == "SplitCorrect"
        }
    names = {p.name for p in P.all()}
    assert all(a[0] in names for atoms in legacy for a in P.translate(atoms=atoms))


def _digest(*, count: int) -> str:
    def atom(*, a) -> str:  # noqa: ANN001 -- LiftedAtom
        return f"{a.predicate.name}({','.join(v.name for v in a.variables)})"

    def describe(*, s) -> str:  # noqa: ANN001 -- Skill
        return "|".join([
            s.name,
            ",".join(f"{v.name}:{v.type.name}" for v in s.parameters),
            *(
                ",".join(sorted(atom(a=x) for x in g))
                for g in (s.preconditions, s.add_effects, s.delete_effects)
            ),
            ",".join(sorted(p.name for p in s.ignore_effects)),
            ",".join(
                sorted(
                    repr(sorted(atom(a=x) for x in c.conditions))
                    + repr(sorted(atom(a=x) for x in c.add_effects))
                    for c in s.conditional_add_effects
                )
            ),
            str(s.param_dim),
        ])

    text = "\n".join(describe(s=s) for s in PourSkills.build(count=count))
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_legacy_benchmark_vocabulary_is_unchanged():
    # Digests taken on 28b7d7125 + #742, before the kitchen table landed.
    assert _digest(count=3) == "5388111a26c90094dafb441238a03ba9463091c5055b52fc9a6352d5af17adab"
    assert _digest(count=5) == "3da4fc607c9e7252ef94539dd2702a3e47940c53a4da9a4640a14f29931af2b7"
    predicates = "\n".join(
        f"{p.name}:{','.join(t.name for t in p.types)}" for p in PourPredicates.all()
    )
    assert (
        hashlib.sha256(predicates.encode()).hexdigest()
        == "77c861c40c3d057d11d4a2052d7a0d2e34eb1988d0eb3dcf9d7e80b74572ad26"
    )
    assert [o.name for o in PourObjects(count=3).all()] == [
        "robot",
        "pitcher",
        "tray",
        "shrimp_0",
        "shrimp_1",
        "shrimp_2",
        "request_1",
        "request_2",
        "request_3",
    ]
