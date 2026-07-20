from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import LIGHT_ON, LightOnClassifier


def test_classifier_holds_when_level_matches_target() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.5)
    assert LightOnClassifier.holds(state=state, light=LightSwitchEnvironment.light) is True


def test_light_on_predicate_adapts_classifier_to_the_holds_contract() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.9)
    light = LightSwitchEnvironment.light
    assert LIGHT_ON.holds(state, (light,)) == LightOnClassifier.holds(state=state, light=light)


def test_light_on_holds_when_level_matches_target() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.5)
    light = LightSwitchEnvironment.light
    atom = LIGHT_ON(state=state, objects=(light,))
    assert atom.predicate.holds(state, atom.objects) is True


def test_light_on_holds_within_tolerance() -> None:
    tolerance = LightSwitchEnvironment.light_on_tolerance
    state = LightSwitchEnvironment.build_initial_state(
        light_level=0.5, light_target=0.5 + tolerance / 2
    )
    light = LightSwitchEnvironment.light
    atom = LIGHT_ON(state=state, objects=(light,))
    assert atom.predicate.holds(state, atom.objects) is True


def test_light_on_does_not_hold_outside_tolerance() -> None:
    tolerance = LightSwitchEnvironment.light_on_tolerance
    state = LightSwitchEnvironment.build_initial_state(
        light_level=0.5, light_target=0.5 + tolerance * 2
    )
    light = LightSwitchEnvironment.light
    atom = LIGHT_ON(state=state, objects=(light,))
    assert atom.predicate.holds(state, atom.objects) is False


def test_light_on_predicate_declares_light_type() -> None:
    assert LIGHT_ON.types == (LightSwitchEnvironment.light_type,)
    assert LIGHT_ON.name == "LightOn"
