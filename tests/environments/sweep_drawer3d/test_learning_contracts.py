"""Continuous-practice and declared-region contracts without simulator execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom
from hitl_pmp.environments.sweep_drawer3d import environment as environment_module
from hitl_pmp.environments.sweep_drawer3d.environment import (
    InvalidSweepStart,
    SweepDrawerEnvironment,
)
from hitl_pmp.environments.sweep_drawer3d.start_regions import StartValidation, SweepRegions
from hitl_pmp.environments.sweep_drawer3d.symbolic import SWEEP_PREDICATES, SweepSymbols
from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects


def _atoms(*, names):
    return frozenset(
        GroundAtom(predicate=SWEEP_PREDICATES[name], objects=(SweepSymbols.SCENE,))
        for name in names
    )


def _state(*, names=(), seed=0):
    values = [
        float(seed) if name == "seed" else float(name in names)
        for name in SweepSymbols.SCENE_TYPE.feature_names
    ]
    return State(data={SweepSymbols.SCENE: np.asarray(values)})


@pytest.mark.parametrize(
    ("yaw", "ranges", "tolerance", "expected"),
    [
        (0, [[0, 0]], 0, True),
        (np.deg2rad(359), [[0, 0]], 5, True),
        (np.deg2rad(6), [[0, 0]], 5, False),
        (-np.pi, [[0, 360]], 0, True),
        (np.pi, [[-180, -180]], 0, True),
        (np.deg2rad(20), [[10, 30]], 0, True),
    ],
)
def test_declared_yaw_ranges_wrap_without_changing_pose(*, yaw, ranges, tolerance, expected):
    assert SweepRegions.yaw_matches(yaw=yaw, ranges=ranges, tolerance_deg=tolerance) is expected


def test_goal_uses_declared_region_not_heuristic_drawer_location(*, monkeypatch):
    core = SimpleNamespace(task_config={"goal_state": [["in", S.CUBES[0], "exact_goal"]]})
    monkeypatch.setattr(SweepRegions, "core", staticmethod(lambda **kwargs: core))
    contains = Mock(return_value=False)
    monkeypatch.setattr(SweepRegions, "contains", staticmethod(contains))
    session = object()
    assert not SweepRegions.in_goal(session=session, cube=S.CUBES[0])
    contains.assert_called_once_with(session=session, name=S.CUBES[0], region="exact_goal")


def test_validation_records_each_failed_constraint_without_mutation(*, monkeypatch):
    names = (S.ROBOT, S.WIPER, *S.CUBES)
    core = SimpleNamespace(
        task_config={
            "initial_state": [["in", name, name + "_region"] for name in names],
            "regions": {
                name + "_region": {"yaw_ranges": [[0, 0]] if name == S.WIPER else [[0, 360]]}
                for name in names
            },
        }
    )
    monkeypatch.setattr(SweepRegions, "core", staticmethod(lambda **kwargs: core))
    monkeypatch.setattr(
        SweepRegions, "contains", staticmethod(lambda **kwargs: kwargs["name"] != S.CUBES[2])
    )
    session = SimpleNamespace(
        seed=28,
        base=lambda: (0, 0, 0),
        position=lambda **kwargs: np.zeros(3),
        quaternion=lambda **kwargs: np.asarray([0, 0, 0, 1]),
        drawer_pos=lambda: 0.02,
    )
    result = SweepRegions.validate(session=session)
    assert not result.valid
    assert set(result.reasons) == {S.CUBES[2] + ":region", "drawer:closed"}
    assert len(result.poses) == 7
    assert len(result.checks) == 15
    assert result.seed == 28


def test_invalid_seed_is_attempted_once_and_not_replaced(*, monkeypatch, tmp_path):
    session = Mock()
    constructor = Mock(return_value=session)
    monkeypatch.setattr(environment_module, "SweepDrawerSession", constructor)
    result = StartValidation(
        seed=28, valid=False, checks={"cube:region": False}, poses={}, reasons=("cube:region",)
    )
    monkeypatch.setattr(SweepRegions, "validate", staticmethod(lambda **kwargs: result))
    env = SweepDrawerEnvironment(canonical_seed=28, output_dir=tmp_path)
    try:
        with pytest.raises(InvalidSweepStart, match="seed 28"):
            env.hard_reset()
        assert constructor.call_count == 1
        assert constructor.call_args.kwargs["seed"] == 28
        assert env.current_state is None
        assert env._hard_reset_count == 0
        assert '"seed": 28' in (tmp_path / "sweep_events.jsonl").read_text()
    finally:
        env.close()
    session.close.assert_called_once()


def test_practice_cannot_reset_after_even_one_noop(*, monkeypatch):
    env = SweepDrawerEnvironment(current_state=_state())
    constructor = Mock(side_effect=AssertionError("must reject before constructing a simulator"))
    monkeypatch.setattr(environment_module, "SweepDrawerSession", constructor)
    before = env.get_current_state()
    assert env.take_action(action=env.noop_action()) is before
    for operation in (
        env.hard_reset,
        lambda: env.reset_to_seed(seed=1),
        lambda: env.set_state(state=_state(seed=1)),
    ):
        with pytest.raises(RuntimeError):
            operation()
    assert constructor.call_count == 0


def test_human_symbolic_reset_reaches_entire_declared_target():
    reset = SweepSymbols.human_reset(cost=3.0)
    before = _atoms(names=SweepSymbols.FACT_NAMES)
    effects = {reset: (reset.add_effects, reset.delete_effects, reset.ignore_effects)}
    after = apply_success_effects(true_atoms=before, ground_skill=reset, effects=effects)
    expected = {
        "HandEmpty",
        "WiperHome",
        "DrawerClosed",
        "RobotHome",
        "AnyCubeInPile",
        *(f"InPile{i}" for i in range(5)),
    }
    assert after == _atoms(names=expected)
    assert reset.evaluate_practice_cost() == 3.0
    assert not reset.preconditions
    assert reset.skill.name not in SweepDrawerEnvironment.ACTION_NAMES


def test_goal_requires_all_five_cubes_and_no_partial_credit():
    names = tuple(f"InDrawer{i}" for i in range(5))
    goal = Goal(atoms=_atoms(names=names))
    for missing in range(5):
        assert not goal.is_satisfied(
            state=_state(names=tuple(name for i, name in enumerate(names) if i != missing))
        )
    assert goal.is_satisfied(state=_state(names=names))


def test_all_declared_recovery_actions_have_dispatch_paths(*, monkeypatch):
    env = SweepDrawerEnvironment()
    primitives = Mock()
    primitives.pick.return_value = ("grasp", None)
    reset = Mock(primitives=primitives, open_to=0.2)
    reset._target.return_value = ((0, 0), 0)
    monkeypatch.setattr(SweepDrawerEnvironment, "recovery", lambda self: reset)
    skills = SweepSymbols.skills()
    assert tuple(skill.name for skill in skills) == env.ACTION_NAMES
    for skill in skills:
        if skill.name in SweepSymbols.TRAINABLE:
            assert skill.param_dim == 2
            continue
        assert skill.param_dim == 0
        if skill.name.startswith("PlaceCube"):
            env._held[S.CUBES[int(skill.name.removeprefix("PlaceCube"))]] = "grasp"
        env._execute_recovery(name=skill.name)
    with pytest.raises(ValueError, match="dispatcher missing"):
        env._execute_recovery(name="UnimplementedRecovery")


@pytest.mark.xfail(
    strict=True,
    reason="ParkWiper does not yet forecast newly feasible picks",
)
def test_park_wiper_can_expose_reachable_cube_without_unnecessary_nudge():
    skills = {
        s.name: GroundSkill(skill=s, objects=(SweepSymbols.SCENE,)) for s in SweepSymbols.skills()
    }
    park, pick = skills["ParkWiper"], skills["PickCube0"]
    before = _atoms(names=("HoldingWiper", "DrawerOpen", "Loose0", "Blocked0"))
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=park,
        effects={park: (park.add_effects, park.delete_effects, park.ignore_effects)},
    )
    # A parked-wiper scene may have a feasible direct pick in the real controller.
    # This fails until the model can represent that possibility without inventing
    # an unconditional Pickable effect for every loose cube.
    assert pick.preconditions <= after
