"""Damage shares the objective, never the experience clock."""

from hitl_pmp.core.practice_costs import ChargeFunction, PracticeAccounting, PracticeCosts
from hitl_pmp.step_protocol import PracticeClock


def test_damage_is_additive_and_does_not_buy_experience():
    accounting = PracticeAccounting(costs=PracticeCosts(damage_contact=ChargeFunction(value=10)))
    accounting.record_damage(events=[dict(cube="cube_0", event_id=1)])
    assert accounting.total_cost == 10
    assert accounting.practice_steps == 0
    assert accounting.summary()["damage_contacts"] == 1
    accounting.record(charge=accounting.robot_charge())
    assert accounting.total_cost == 11
    assert accounting.practice_steps == 1


def test_measurement_includes_damage_from_boundary_step():
    accounting = PracticeAccounting(costs=PracticeCosts(damage_contact=ChargeFunction(value=10)))
    measured = []
    clock = PracticeClock(
        budget=10,
        interval=1,
        accounting=accounting,
        robot_active=True,
        damage_events=lambda: [dict(event_id=1)],
        on_measure=lambda: measured.append(accounting.total_cost),
    )
    clock.before_robot_step()
    clock.after_robot_step()
    assert measured == [11]
