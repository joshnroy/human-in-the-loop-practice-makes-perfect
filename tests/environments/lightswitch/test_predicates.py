from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_ON,
    ROBOT_IN_CELL,
    AdjacentClassifier,
    LightOnClassifier,
)

# build_initial_state/get_cells read instance-level grid_size now -- light_on_tolerance
# and the type/Object constants these predicates otherwise reference stay ClassVar
# (see environment.py's own docstring), so a bare default-constructed instance is
# all any test below needs.
_ENV = LightSwitchEnvironment()


def test_classifier_holds_when_level_matches_target() -> None:
    state = _ENV.build_initial_state(light_level=0.5, light_target=0.5)
    assert LightOnClassifier.holds(state=state, light=LightSwitchEnvironment.light) is True


def test_light_on_predicate_adapts_classifier_to_the_holds_contract() -> None:
    state = _ENV.build_initial_state(light_level=0.5, light_target=0.9)
    light = LightSwitchEnvironment.light
    assert LIGHT_ON.holds(state, (light,)) == LightOnClassifier.holds(state=state, light=light)


def test_light_on_holds_when_level_matches_target() -> None:
    state = _ENV.build_initial_state(light_level=0.5, light_target=0.5)
    light = LightSwitchEnvironment.light
    atom = LIGHT_ON(state=state, objects=(light,))
    assert atom.predicate.holds(state, atom.objects) is True


def test_light_on_holds_within_tolerance() -> None:
    tolerance = LightSwitchEnvironment.light_on_tolerance
    state = _ENV.build_initial_state(light_level=0.5, light_target=0.5 + tolerance / 2)
    light = LightSwitchEnvironment.light
    atom = LIGHT_ON(state=state, objects=(light,))
    assert atom.predicate.holds(state, atom.objects) is True


def test_light_on_does_not_hold_outside_tolerance() -> None:
    tolerance = LightSwitchEnvironment.light_on_tolerance
    state = _ENV.build_initial_state(light_level=0.5, light_target=0.5 + tolerance * 2)
    light = LightSwitchEnvironment.light
    atom = LIGHT_ON(state=state, objects=(light,))
    assert atom.predicate.holds(state, atom.objects) is False


def test_light_on_predicate_declares_light_type() -> None:
    assert LIGHT_ON.types == (LightSwitchEnvironment.light_type,)
    assert LIGHT_ON.name == "LightOn"


def test_robot_in_cell_holds_for_the_robots_own_cell() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.5)
    robot = LightSwitchEnvironment.robot
    cells = _ENV.get_cells()
    atom = ROBOT_IN_CELL(state=state, objects=(robot, cells[0]))
    assert atom.predicate.holds(state, atom.objects) is True


def test_robot_in_cell_does_not_hold_for_a_different_cell() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.5)
    robot = LightSwitchEnvironment.robot
    cells = _ENV.get_cells()
    atom = ROBOT_IN_CELL(state=state, objects=(robot, cells[1]))
    assert atom.predicate.holds(state, atom.objects) is False


def test_light_in_cell_holds_for_the_lights_own_cell() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.5)
    light = LightSwitchEnvironment.light
    cells = _ENV.get_cells()
    atom = LIGHT_IN_CELL(state=state, objects=(light, cells[-1]))
    assert atom.predicate.holds(state, atom.objects) is True


def test_adjacent_classifier_holds_for_consecutive_cells() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.5)
    cells = _ENV.get_cells()
    assert AdjacentClassifier.holds(state=state, cell1=cells[0], cell2=cells[1]) is True


def test_adjacent_classifier_does_not_hold_for_distant_cells() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.5)
    cells = _ENV.get_cells()
    assert AdjacentClassifier.holds(state=state, cell1=cells[0], cell2=cells[-1]) is False


def test_adjacent_predicate_declares_cell_types() -> None:
    assert ADJACENT.types == (LightSwitchEnvironment.cell_type, LightSwitchEnvironment.cell_type)
    assert ADJACENT.name == "Adjacent"
