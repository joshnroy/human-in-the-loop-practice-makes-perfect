"""Recovery actions must be offered only for their actual physical contracts."""

from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols


def test_pick_requires_actual_planned_feasibility():
    skills = {s.name: s for s in SweepSymbols.skills()}
    for i in range(5):
        assert f"Pickable{i}" in {a.predicate.name for a in skills[f"PickCube{i}"].preconditions}


def test_parking_requires_physical_wiper_not_coarse_native_gripper_fact():
    park = next(s for s in SweepSymbols.skills() if s.name == "ParkWiper")
    assert {a.predicate.name for a in park.preconditions} == {"PhysicallyHoldingWiper"}


def test_recovery_opening_uses_its_own_target_admission():
    skill = next(s for s in SweepSymbols.skills() if s.name == "OpenResetDrawer")
    assert {a.predicate.name for a in skill.preconditions} == {
        "RecoveryHandEmpty",
        "ResetDrawerNotOpen",
    }


def test_release_has_no_fictional_cube_destination():
    skill = next(s for s in SweepSymbols.skills() if s.name == "OpenGripper")
    assert {a.predicate.name for a in skill.preconditions} == {"RecoveryGripperClosed"}
    assert not any(a.predicate.name.startswith(("InPile", "OnTable")) for a in skill.add_effects)
    assert {f"HoldingCube{i}" for i in range(5)} <= {a.predicate.name for a in skill.delete_effects}


def test_missing_recovery_capabilities_are_individually_selectable():
    names = {s.name for s in SweepSymbols.skills()}
    assert {
        "RecoverWiper",
        "WiggleDrawerCube0",
        "PickGroup01",
        "PlaceGroup01",
        "PickGroup012",
        "PlaceGroup012",
    } <= names


def test_successful_release_actions_delete_closed_gripper_fact():
    for skill in SweepSymbols.skills():
        adds = {a.predicate.name for a in skill.add_effects}
        if "RecoveryHandEmpty" in adds:
            assert "RecoveryGripperClosed" in {a.predicate.name for a in skill.delete_effects}


def test_group_placement_removes_loose_and_holding_facts_for_every_member():
    skill = next(s for s in SweepSymbols.skills() if s.name == "PlaceGroup123")
    deletes = {a.predicate.name for a in skill.delete_effects}
    assert {
        f"{prefix}{i}" for prefix in ("Loose", "HoldingCube", "Blocked") for i in (1, 2, 3)
    } <= deletes


def test_recover_wiper_success_requires_finished_physical_stow():
    skill = next(s for s in SweepSymbols.skills() if s.name == "RecoverWiper")
    assert "RecoveryArmHome" in {a.predicate.name for a in skill.add_effects}
