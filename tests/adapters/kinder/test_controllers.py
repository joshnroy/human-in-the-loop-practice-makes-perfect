"""KINDER's lifted parameterized controllers behind `core`'s `Skill` vocabulary.

Fakes rather than the real controllers, for the same reason `test_abstraction.py` uses
one: the properties here are about *delegation* -- that the sampler consulted is the
controller's own, that a raising controller becomes a reported failure rather than an
exception, that a `?`-prefixed KINDER variable name never reaches PDDL -- and a real
controller would bury each of them under seconds of motion planning. The real ones are
exercised through these same entry points in `tests/environments/tossing3d/`.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.adapters.kinder.controllers import KinderControllers
from hitl_pmp.adapters.kinder.state_translation import KinderStateTranslator

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("relational_structs") is None,
    reason="relational_structs is part of the optional tossing3d extra",
)


def _kinder_state():
    from relational_structs import Object as KinderObject
    from relational_structs import ObjectCentricState
    from relational_structs import Type as KinderType

    robot_type = KinderType("robot")
    block_type = KinderType("block")
    type_features = {robot_type: ["grip"], block_type: ["x"]}
    data = {
        KinderObject("robot", robot_type): np.array([0.0], dtype=np.float64),
        KinderObject("block_0", block_type): np.array([1.5], dtype=np.float64),
    }
    return ObjectCentricState(data, type_features)


class _FakeGroundController:
    def __init__(self, *, objects, script) -> None:
        self.objects = objects
        self.script = script
        self.steps = 0
        self.reset_kwargs: dict = {}

    def sample_parameters(self, x, rng):  # noqa: PLR0917  (KINDER's own signature)
        # Deliberately reads `x`: PickCubeController's real sampler does, and the point of
        # delegating is that whatever upstream reads, upstream gets.
        span = float(x.get(x.get_object_from_name("block_0"), "x"))
        return np.array([rng.uniform(0.0, span), rng.uniform(-1.0, 1.0)])

    def reset(self, x, params, **kwargs) -> None:  # noqa: PLR0917  (KINDER's own signature)
        del x, params
        self.reset_kwargs = kwargs
        if self.script.get("raise_on_reset"):
            raise AssertionError("Motion planning failed")

    def step(self):
        self.steps += 1
        if self.script.get("raise_on_step") == self.steps:
            raise AssertionError("controller blew up mid-execution")
        return np.zeros(3)

    def observe(self, x) -> None:  # noqa: PLR0917  (KINDER's own signature)
        del x

    def terminated(self) -> bool:
        return self.steps >= self.script.get("terminate_after", 2)


class _FakeLiftedController:
    def __init__(self, *, variables, script=None) -> None:
        self.variables = variables
        self.script = script or {}

    def ground(self, objects):  # noqa: PLR0917  (KINDER's own signature)
        return _FakeGroundController(objects=objects, script=self.script)


def _variables():
    from relational_structs import Type as KinderType
    from relational_structs import Variable

    # KINDER's own convention: the name carries the leading "?".
    return [Variable("?robot", KinderType("robot")), Variable("?block", KinderType("block"))]


def _build(*, script=None):
    kinder_state = _kinder_state()
    translator = KinderStateTranslator.from_kinder_state(kinder_state=kinder_state)
    controllers = KinderControllers(
        translator=translator,
        lifted_controllers={"do_it": _FakeLiftedController(variables=_variables(), script=script)},
    )
    return controllers, translator, translator.to_core_state(kinder_state=kinder_state)


def test_a_kinder_variable_loses_its_question_mark_on_the_way_into_core() -> None:
    """**Load-bearing, and it has bitten this repo before.** `core.Variable.name` carries
    no sigil -- `PddlWriter._variable_str` adds one at write time -- so a name arriving as
    `?robot` renders `??robot`, which Fast Downward's translator splits into two tokens
    and every plan call in the domain fails silently. KINDER's own variables *do* carry
    the `?`, so the strip has to happen here."""
    controllers, _, _ = _build()

    assert [v.name for v in controllers.variables(key="do_it")] == ["robot", "block"]


def test_a_variables_type_is_the_translated_kinder_type() -> None:
    controllers, translator, _ = _build()
    variables = controllers.variables(key="do_it")

    assert variables[0].type is translator.core_types["robot"]
    assert variables[1].type is translator.core_types["block"]


def test_sampling_calls_the_controllers_own_sampler_rather_than_a_local_range() -> None:
    """The reason this bridge exists at all. hitl previously hardcoded its own bounds
    beside upstream's; here the draw is upstream's, so a band upstream narrows narrows
    here too, with nothing to keep in step."""
    controllers, _, state = _build()

    drawn = controllers.sample_params(
        key="do_it", object_names=("robot", "block_0"), state=state, rng=np.random.default_rng(0)
    )

    expected = _FakeGroundController(objects=(), script={}).sample_parameters(
        _kinder_state(), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(drawn, expected)


def test_the_sampler_is_handed_the_live_state_not_a_blank_one() -> None:
    """`PickCubeController.sample_parameters` reads the target's pose and rejection-tests
    against other cubes, so the state it gets has to be the real one."""
    controllers, translator, state = _build()
    moved = state.model_copy(deep=True)
    moved.set(obj=translator.core_objects["block_0"], feature_name="x", feature_val=99.0)

    near = controllers.sample_params(
        key="do_it", object_names=("robot", "block_0"), state=state, rng=np.random.default_rng(1)
    )
    far = controllers.sample_params(
        key="do_it", object_names=("robot", "block_0"), state=moved, rng=np.random.default_rng(1)
    )

    assert far[0] > near[0]


def test_running_a_controller_reports_the_step_count_it_took_to_terminate() -> None:
    controllers, _, state = _build(script={"terminate_after": 3})
    seen: list = []

    run = controllers.run(
        key="do_it",
        object_names=("robot", "block_0"),
        params=np.zeros(2),
        state=state,
        step=lambda action: (seen.append(action), _kinder_state())[1],
        limit=10,
    )

    assert run.terminated is True
    assert run.steps == 3
    assert run.error is None
    assert len(seen) == 3


def test_a_controller_that_fails_to_plan_is_a_reported_failure_not_an_exception() -> None:
    """KINDER's motion planners `assert plan is not None`, so an unreachable grasp raises
    out of `reset()`. `core.Environment.take_action` must be total over its action space,
    so this has to come back as data."""
    controllers, _, state = _build(script={"raise_on_reset": True})

    run = controllers.run(
        key="do_it",
        object_names=("robot", "block_0"),
        params=np.zeros(2),
        state=state,
        step=lambda action: _kinder_state(),
        limit=10,
    )

    assert run.terminated is False
    assert run.steps == 0
    assert "Motion planning failed" in (run.error or "")


def test_a_controller_that_raises_mid_execution_reports_the_steps_it_managed() -> None:
    controllers, _, state = _build(script={"raise_on_step": 2, "terminate_after": 99})

    run = controllers.run(
        key="do_it",
        object_names=("robot", "block_0"),
        params=np.zeros(2),
        state=state,
        step=lambda action: _kinder_state(),
        limit=10,
    )

    assert run.terminated is False
    assert run.steps == 1
    assert "blew up" in (run.error or "")


def test_a_controller_that_never_terminates_stops_at_the_limit() -> None:
    controllers, _, state = _build(script={"terminate_after": 99})

    run = controllers.run(
        key="do_it",
        object_names=("robot", "block_0"),
        params=np.zeros(2),
        state=state,
        step=lambda action: _kinder_state(),
        limit=4,
    )

    assert run.terminated is False
    assert run.steps == 4
    assert run.error is None


def test_reset_keywords_reach_the_controller_only_when_supplied() -> None:
    """`disable_collision_objects`, `release_speed` and friends are declared on some
    controllers' `reset` and not others, so passing one to a controller that does not take
    it is a `TypeError`. Forwarding only what was asked for is what keeps this generic."""
    controllers, _, state = _build()
    captured: list = []

    class _Recorder(_FakeLiftedController):
        def ground(self, objects):  # noqa: PLR0917  (KINDER's own signature)
            ground = super().ground(objects)
            captured.append(ground)
            return ground

    controllers.lifted_controllers["do_it"] = _Recorder(variables=_variables())
    controllers.run(
        key="do_it",
        object_names=("robot", "block_0"),
        params=np.zeros(2),
        state=state,
        step=lambda action: _kinder_state(),
        limit=4,
        reset_kwargs={"disable_collision_objects": ["block_0"]},
    )

    assert captured[0].reset_kwargs == {"disable_collision_objects": ["block_0"]}


def test_an_unknown_controller_key_names_the_ones_that_exist() -> None:
    controllers, _, state = _build()

    with pytest.raises(KeyError, match="do_it"):
        controllers.variables(key="nope")
