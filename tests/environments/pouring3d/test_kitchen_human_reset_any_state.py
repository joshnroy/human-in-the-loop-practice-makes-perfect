"""The kitchen's human reset is available from every practice state, not only an empty hand.

Della 16949734 expectimax/pomdp/2: a closed-loop Pour1 flung potsticker_0 onto the floor
while the robot still held the jug. PlacePitcherAtHome was then refused 100 times at
0 ticks, and the human pour-back required an empty hand and the jug at home, so nothing
was applicable and practice ended with `repeated_zero_step` at tick 72,152. The fixture
is that run's recorded state after the Pour (recording tick seq 159).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.practice.skills import RESET_NAME, PourSkills

needs_kinder = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER")
FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_holding_jug_dumpling_on_floor.json"
Atoms = frozenset[tuple[str, tuple[str, ...]]]


def _names(*, atoms) -> Atoms:  # noqa: ANN001 -- GroundAtoms
    return frozenset((a.predicate.name, tuple(o.name for o in a.objects)) for a in atoms)


def _reset(*, env: KitchenEnvironment):  # noqa: ANN202 -- GroundSkill
    human = KitchenSkillProvider(env=env).human_cube_bin_reset_skill()
    assert human is not None
    return human


def _applicable(*, ground, atoms: Atoms) -> bool:  # noqa: ANN001
    return _names(atoms=ground.preconditions) <= atoms


def _successor(*, ground, atoms: Atoms) -> Atoms:  # noqa: ANN001
    ignored = {p.name for p in ground.ignore_effects}
    kept = {a for a in atoms - _names(atoms=ground.delete_effects) if a[0] not in ignored}
    return frozenset(kept | _names(atoms=ground.add_effects))


def _rebuild_holding_jug(*, env: KitchenEnvironment) -> None:
    import mujoco

    state = json.loads(FIXTURE.read_text())
    c = env.controller()
    c.data.qpos[:] = state["qpos"]
    c.data.qvel[:] = state["qvel"]
    c.data.time = state["time"]
    mujoco.mj_forward(c.model, c.data)
    # The recording carries MuJoCo state only; the run held the jug, closed, here.
    c.held, c.grip = "pitcher", 1.0
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()


HOLDING_JUG_DUMPLING_ON_FLOOR: Atoms = frozenset({
    ("HoldingPitcher", ("robot", "pitcher")),
    ("TrayHome", ("tray",)),
    ("OnFloor", ("potsticker_0",)),
    ("InPitcher", ("potsticker_1", "pitcher")),
    ("InPitcher", ("potsticker_2", "pitcher")),
    ("PayloadOrder", ("potsticker_0", "potsticker_1")),
    ("PayloadOrder", ("potsticker_1", "potsticker_2")),
    ("NeedsReset", ()),
})
LEGACY_STATE: Atoms = frozenset({
    ("HandEmpty", ("robot",)),
    ("PitcherHome", ("pitcher",)),
    ("TrayHome", ("tray",)),
    ("OnFloor", ("potsticker_0",)),
    ("InPitcher", ("potsticker_1", "pitcher")),
    ("InPitcher", ("potsticker_2", "pitcher")),
    ("PayloadOrder", ("potsticker_0", "potsticker_1")),
    ("PayloadOrder", ("potsticker_1", "potsticker_2")),
    ("NeedsReset", ()),
    ("PickPlannable", ("robot", "potsticker_0")),
})


def test_the_human_reset_is_applicable_while_holding_the_jug_and_restores_everything():
    human = _reset(env=KitchenEnvironment())
    assert _applicable(ground=human, atoms=HOLDING_JUG_DUMPLING_ON_FLOOR)
    after = _successor(ground=human, atoms=HOLDING_JUG_DUMPLING_ON_FLOOR)
    assert {
        ("HandEmpty", ("robot",)),
        ("PitcherHome", ("pitcher",)),
        ("ReadyToPour", ()),
        ("NoSpills", ()),
        *((("InPitcher", (f"potsticker_{i}", "pitcher"))) for i in range(3)),
        *((("Settled", (f"potsticker_{i}",))) for i in range(3)),
    } <= after
    assert not {a for a in after if a[0] in {"HoldingPitcher", "HoldingShrimp", "OnFloor"}}
    assert ("NeedsReset", ()) not in after


def test_the_human_reset_is_applicable_while_holding_a_dumpling_with_the_jug_away():
    human = _reset(env=KitchenEnvironment())
    atoms = frozenset({
        ("HoldingShrimp", ("robot", "potsticker_1")),
        ("TrayHome", ("tray",)),
        ("InTray", ("potsticker_0", "tray")),
        ("OnSurface", ("potsticker_2",)),
        ("PayloadOrder", ("potsticker_0", "potsticker_1")),
        ("PayloadOrder", ("potsticker_1", "potsticker_2")),
        ("NeedsReset", ()),
    })
    assert _applicable(ground=human, atoms=atoms)
    after = _successor(ground=human, atoms=atoms)
    assert {("HandEmpty", ("robot",)), ("PitcherHome", ("pitcher",))} <= after
    assert not {a for a in after if a[0] in {"HoldingShrimp", "InTray", "OnSurface"}}


def test_from_a_state_the_reset_already_covered_its_successor_is_unchanged():
    env = KitchenEnvironment()
    human = _reset(env=env)
    legacy_skill = next(s for s in PourSkills.build(count=3) if s.name == RESET_NAME)
    legacy = human.model_copy(update={"skill": legacy_skill})
    assert _applicable(ground=legacy, atoms=LEGACY_STATE)
    assert _applicable(ground=human, atoms=LEGACY_STATE)
    expected = _successor(
        ground=legacy, atoms=LEGACY_STATE - {a for a in LEGACY_STATE if a[0] == "PickPlannable"}
    )
    assert _successor(ground=human, atoms=LEGACY_STATE) == expected


def test_the_human_reset_keeps_its_name_and_is_not_applicable_with_nothing_to_reset():
    human = _reset(env=KitchenEnvironment())
    assert human.skill.name == HUMAN_RESET
    reset_state = frozenset(a for a in LEGACY_STATE if a != ("NeedsReset", ()))
    assert not _applicable(ground=human, atoms=reset_state)


@needs_kinder
@pytest.mark.xdist_group("kitchen_human_reset_any_state")
def test_from_the_recorded_dead_end_the_reset_is_offered_and_puts_everything_back():
    env = KitchenEnvironment(requested_count=1)
    try:
        _rebuild_holding_jug(env=env)
        c = env.controller()
        assert c.scene.cube_locations() == ["floor", "pitcher", "pitcher"]
        base, arm = c.data.qpos[c.base_q].copy(), c.data.qpos[c.arm_q].copy()
        atoms = env.observe().abstract_atoms
        assert ("HoldingPitcher", ("robot", "pitcher")) in atoms
        assert ("NeedsReset", ()) in atoms
        human = _reset(env=env)
        assert _applicable(ground=human, atoms=atoms)

        assert env.reset_movables(destination=HUMAN_RESET)
        assert env.last_info["success"]
        assert env.human_interventions == 1
        assert env.interruption_cost_total == 5
        assert c.held is None
        assert c.scene.attached_payload() is None
        assert c.scene.cube_locations() == ["pitcher"] * 3
        assert c.vessel_home(name="pitcher")
        assert c.vessel_home(name="tray")
        assert c.ready()
        after = env.observe().abstract_atoms
        assert _names(atoms=human.add_effects) <= after
        assert not _names(atoms=human.delete_effects) & after
        assert ("NeedsReset", ()) not in after
        # The human moves food and the jug only; the robot stays where it was.
        np.testing.assert_array_equal(base, c.data.qpos[c.base_q])
        np.testing.assert_array_equal(arm, c.data.qpos[c.arm_q])
        # The released jug at home is not inside the open hand.
        assert not c.finger_contacts(name="pitcher")
    finally:
        env.close()


@needs_kinder
@pytest.mark.xdist_group("kitchen_human_reset_any_state")
def test_from_an_empty_hand_and_jug_at_home_the_reset_is_exactly_the_legacy_food_respawn():
    import mujoco

    from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment

    def spilled() -> KitchenEnvironment:
        env = KitchenEnvironment(requested_count=3)
        c = env.controller()
        c.scene.get_object("potsticker_0").set_pose([0.35, 0.2, 0.002], [1, 0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        assert c.scene.cube_locations()[0] == "floor"
        return env

    new, old = spilled(), spilled()
    try:
        assert new.reset_movables(destination=HUMAN_RESET)
        assert FrozenPourEnvironment.reset_movables(old, destination=RESET_NAME)
        np.testing.assert_array_equal(new.controller().data.qpos, old.controller().data.qpos)
        np.testing.assert_array_equal(new.controller().data.qvel, old.controller().data.qvel)
        assert new.controller().grip == old.controller().grip
    finally:
        new.close()
        old.close()
