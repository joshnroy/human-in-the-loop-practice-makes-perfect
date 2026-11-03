"""Measured practice and deployment scoring parity without native physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.predicates import RackServing3DPredicates
from hitl_pmp.environments.racktoserving3d.semantics import RackServingLedger, SupportRegion
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend
from hitl_pmp.step_protocol import EvaluationClock


@pytest.fixture
def scoring_scene(*, monkeypatch):
    names = [food.name for food in Objects.food]
    addresses = 7 + 6 * np.arange(len(names))
    model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=names.index(name)),
        body_jntadr=np.arange(len(names)),
        jnt_dofadr=addresses,
    )
    data = SimpleNamespace(time=0.0, qvel=np.zeros(7 + 6 * len(names)))
    supports = {name: {"box_0"} for name in names}
    held = set()
    monkeypatch.setattr(RackServingBackend, "model", lambda self: model)
    monkeypatch.setattr(RackServingBackend, "data", lambda self: data)
    monkeypatch.setattr(RackServingBackend, "held", lambda self, *, name: name in held)
    monkeypatch.setattr(
        RackServingBackend, "supporting_bodies", lambda self, *, name: supports[name]
    )
    monkeypatch.setattr(
        RackServingBackend,
        "corners",
        lambda self, *, name: np.array([[-0.01, -0.01, 1.0], [0.01, 0.01, 1.025]]),
    )
    monkeypatch.setattr(
        RackServingBackend,
        "support",
        lambda self, *, name: SupportRegion(x_min=-1, x_max=1, y_min=-1, y_max=1, top_z=1),
    )
    # A reset also lays the food out (`test_layout.py`); these tests score a fixed scene.
    monkeypatch.setattr(RackServingBackend, "food_placements", lambda self, *, names: {})
    monkeypatch.setattr(RackServingBackend, "_commit", lambda self, *, placements: None)
    backend = RackServingBackend(ledger=RackServingLedger(elapsed_seconds=300))
    backend._tool_home = lambda: True
    return SimpleNamespace(
        backend=backend, data=data, supports=supports, held=held, addresses=addresses
    )


def _ground_atoms(*, atoms):
    predicates = {predicate.name: predicate for predicate in RackServing3DPredicates.all()}
    objects = {obj.name: obj for obj in Objects.all_objects()}
    return (
        frozenset(
            GroundAtom(predicate=predicates[name], objects=tuple(objects[key] for key in keys))
            for name, keys in atoms
        )
        | RackServing3DPredicates.static_atoms()
    )


@pytest.mark.parametrize(
    "velocity, expected",
    [
        ([0, 0, 0, 0, 0, 0], True),
        ([0.02, 0, 0, 0.5, 0, 0], True),
        ([0.020001, 0, 0, 0, 0, 0], False),
        ([0, 0, 0, 0.500001, 0, 0], False),
        ([0.015, 0.015, 0, 0, 0, 0], False),
        ([0, 0, 0, 0.4, 0.4, 0], False),
        ([0, 0, -0.03, 0, 0, 0], False),
    ],
)
@pytest.mark.parametrize("food_index", [0, 4])
def test_practice_goal_and_deployment_share_settled_thresholds(
    *, scoring_scene, velocity, expected, food_index
):
    scene = scoring_scene
    name = Objects.food[food_index].name
    address = scene.addresses[food_index]
    scene.data.qvel[address : address + 6] = velocity
    assert scene.backend.location(name=name) == "OnTray"
    atoms = scene.backend.abstract_atoms()
    assert (("OnTray", (name, Objects.tray.name)) in atoms) is expected
    assert scene.backend.task_succeeded() is expected
    assert (RackServing3DPredicates.goal_atoms() <= _ground_atoms(atoms=atoms)) is expected
    if not expected:
        assert not any(atom[0].startswith("Spilled") and name in atom[1] for atom in atoms)


@pytest.mark.parametrize("component", range(6))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_velocity_never_scores(*, scoring_scene, component, value):
    scene = scoring_scene
    scene.data.qvel[scene.addresses[0] + component] = value
    assert not scene.backend.task_succeeded()
    assert (
        "OnTray",
        (Objects.food[0].name, Objects.tray.name),
    ) not in scene.backend.abstract_atoms()


@pytest.mark.parametrize("location", [None, "HoldingFood", "OnRack", "SpilledCounter"])
def test_zero_velocity_without_unheld_tray_support_never_scores(*, scoring_scene, location):
    scene = scoring_scene
    name = Objects.food[0].name
    scene.supports[name] = {
        None: set(),
        "HoldingFood": {"box_0"},
        "OnRack": {"rack_0"},
        "SpilledCounter": {"cupboard_1"},
    }[location]
    if location == "HoldingFood":
        scene.held.add(name)
    assert scene.backend.location(name=name) == location
    atoms = scene.backend.abstract_atoms()
    assert ("OnTray", (name, Objects.tray.name)) not in atoms
    assert not scene.backend.task_succeeded()
    assert not RackServing3DPredicates.goal_atoms() <= _ground_atoms(atoms=atoms)
    if location is not None:
        assert any(atom[0] == location and name in atom[1] for atom in atoms)


def test_readiness_is_separate_from_settled_delivery(*, scoring_scene):
    backend = scoring_scene.backend
    backend.ledger = RackServingLedger(elapsed_seconds=299)
    atoms = backend.abstract_atoms()
    assert all(("OnTray", (food.name, Objects.tray.name)) in atoms for food in Objects.food)
    assert ("Waiting", ()) in atoms
    assert not backend.task_succeeded()
    assert not RackServing3DPredicates.goal_atoms() <= _ground_atoms(atoms=atoms)


@pytest.mark.parametrize("skill", ["sweep", "place"])
def test_unsettled_delivery_does_not_satisfy_practice_effects(*, scoring_scene, skill):
    scene = scoring_scene
    name = Objects.food[0].name
    if skill == "sweep":
        scene.held.add(Objects.scraper.name)
        scene.supports[name] = {"rack_0"}
        ground = GroundSkill(
            skill=Skills.GATHER_AND_SWEEP,
            objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
        )
    else:
        scene.held.add(name)
        ground = GroundSkill(
            skill=Skills.PLACE_FOOD_ON_TRAY, objects=(Objects.robot, Objects.food[0], Objects.tray)
        )
    before = _ground_atoms(atoms=scene.backend.abstract_atoms())
    assert ground.preconditions <= before
    promised = ground.expected_add_effects(before_atoms=before)
    scene.held.discard(name)
    scene.supports[name] = {"box_0"}
    scene.data.qvel[scene.addresses[0] + 3] = 0.6
    assert not promised <= _ground_atoms(atoms=scene.backend.abstract_atoms())
    assert not scene.backend.task_succeeded()
    scene.data.qvel[:] = 0
    assert promised <= _ground_atoms(atoms=scene.backend.abstract_atoms())
    assert scene.backend.task_succeeded()


def test_new_observation_does_not_latch_transient_delivery(*, scoring_scene):
    scene = scoring_scene
    name = Objects.food[0].name
    delivered = ("OnTray", (name, Objects.tray.name))
    assert delivered in scene.backend.abstract_atoms()
    scene.data.qvel[scene.addresses[0]] = 0.1
    assert delivered not in scene.backend.abstract_atoms()
    assert not scene.backend.task_succeeded()
    scene.data.qvel[:] = 0
    scene.supports[name] = set()
    assert scene.backend.location(name=name) is None
    assert delivered not in scene.backend.abstract_atoms()
    assert not scene.backend.task_succeeded()


def test_legacy_cube_in_box_callback_requires_settled_delivery(*, scoring_scene, monkeypatch):
    scene = scoring_scene
    core = SimpleNamespace()
    monkeypatch.setattr(SweepOffShelf3DBackend, "reset", lambda self, *, seed: None)
    monkeypatch.setattr(RackServingBackend, "core", lambda self: core)
    scene.backend.reset(seed=0)
    name = Objects.food[0].name
    assert core.cube_in_box(name)
    scene.data.qvel[scene.addresses[0] + 3] = 0.6
    assert not core.cube_in_box(name)
    scene.data.qvel[:] = 0
    scene.held.add(name)
    assert not core.cube_in_box(name)


def test_variant_enables_soft_finger_contacts_only_on_the_gripper_pads(
    *, scoring_scene, monkeypatch
):
    backend = scoring_scene.backend
    model = backend.model()
    model.geom_condim = np.full(8, 3)
    backend._pads = (frozenset({1, 2}), frozenset({3, 4}))
    monkeypatch.setattr(SweepOffShelf3DBackend, "reset", lambda self, *, seed: None)
    monkeypatch.setattr(RackServingBackend, "core", lambda self: SimpleNamespace())
    backend.reset(seed=0)
    np.testing.assert_array_equal(model.geom_condim, [3, 4, 4, 4, 4, 3, 3, 3])


def test_overhanging_tray_food_remains_available_for_autonomous_pick(*, scoring_scene, monkeypatch):
    backend = scoring_scene.backend
    name = Objects.food[0].name
    original_corners = backend.corners
    monkeypatch.setattr(
        RackServingBackend,
        "corners",
        lambda self, *, name: (
            np.array([[0.99, -0.01, 1.0], [1.03, 0.01, 1.025]])
            if name == Objects.food[0].name
            else original_corners(name=name)
        ),
    )
    assert backend.location(name=name) is None
    assert not backend.task_succeeded()
    atoms = backend.abstract_atoms()
    assert ("OnTray", (name, Objects.tray.name)) not in atoms
    assert ("FoodAvailable", (name,)) in atoms
    ground = GroundSkill(
        skill=Skills.PICK_FOOD, objects=(Objects.robot, Objects.food[0], Objects.rack, Objects.tray)
    )
    assert ground.preconditions <= _ground_atoms(atoms=atoms)


def test_supported_but_unsettled_food_is_pickable_without_delivery_credit(*, scoring_scene):
    scene = scoring_scene
    name = Objects.food[0].name
    scene.data.qvel[scene.addresses[0]] = 0.1
    atoms = scene.backend.abstract_atoms()
    assert ("OnTray", (name, Objects.tray.name)) not in atoms
    assert ("FoodAvailable", (name,)) in atoms


def test_step_evaluation_does_not_latch_rolling_or_unsupported_contact(*, scoring_scene):
    scene = scoring_scene
    env = RackServingEnvironment()
    env._backend = scene.backend
    clock = EvaluationClock(
        budget=10,
        check_goal=RackServingStepProtocol.goal_check(problem=SimpleNamespace(env=env), task=None),
    )
    name = Objects.food[0].name
    scene.data.qvel[scene.addresses[0] + 3] = 0.6
    clock.after()
    assert not clock.solved
    scene.data.qvel[:] = 0
    scene.supports[name] = set()
    clock.after()
    assert not clock.solved
    scene.supports[name] = {"box_0"}
    scene.held.add(name)
    clock.after()
    assert not clock.solved
    scene.held.clear()
    clock.after()
    assert clock.solved
