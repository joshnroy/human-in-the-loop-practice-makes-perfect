"""A Serving practice simulation saved between skills continues identically elsewhere.

The parts without a simulator run everywhere; the live one needs KINDER and belongs to
the physics suite, as Tossing's engine checkpoint test does.
"""

import importlib.util
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.simulation_checkpoint import (
    ControllerState,
    ModelArrays,
)
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.tossing3d.simulation_checkpoint import (
    capture_engine,
    model_fingerprint,
    restore_engine,
)
from hitl_pmp.step_protocol import RngStates

WELDED = """<mujoco><option timestep="0.002"/><worldbody>
<geom type="plane" size="1 1 .1"/>
<body name="hand" pos="0 0 .3"><joint type="slide" axis="0 0 1"/>
<geom type="sphere" size=".05" mass="1"/></body>
<body name="tool" pos="0 0 .5"><freejoint/><geom type="box" size=".05 .02 .01" mass=".1"/></body>
</worldbody><equality><weld name="grasp" body1="hand" body2="tool" active="false"/></equality>
</mujoco>"""

# The budget sweep's Serving scene (`scripts/racktoserving3d_two_route_budget_sweep.sh`).
SWEEP_SCENE = {
    "two_route_rewards": True,
    "start_layout": "middle_three",
    "side_offset": 0.13,
    "plate_tuck": 0.035,
    "sampler_features": "parameters",
    "bulk_transfer_only": True,
    "surface_kind": "wire_solid",
    "scraper_attachment": "grasp_weld",
    "human_interruption_cost": 5,
}


def test_a_weld_written_into_the_compiled_model_travels_with_the_checkpoint():
    """A grasp weld's relative pose lives in `model.eq_data`, which no MuJoCo state
    vector carries: a scene rebuilt from its seed has the weld's compiled pose."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(WELDED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    model.eq_data[0, 3:6] = [0.01, 0.02, 0.2]
    data.eq_active[0] = 1
    for _ in range(25):
        mujoco.mj_step(model, data)
    arrays, engine = ModelArrays.capture(model=model), capture_engine(model=model, data=data)
    expected = []
    for _ in range(20):
        mujoco.mj_step(model, data)
        expected.append(capture_engine(model=model, data=data)["integration"])

    rebuilt = mujoco.MjModel.from_xml_string(WELDED)
    rebuilt_data = mujoco.MjData(rebuilt)
    with pytest.raises(ValueError, match="fingerprint"):
        restore_engine(model=rebuilt, data=rebuilt_data, checkpoint=engine)
    ModelArrays.restore(model=rebuilt, arrays=pickle.loads(pickle.dumps(arrays)))
    restore_engine(model=rebuilt, data=rebuilt_data, checkpoint=engine)
    for state in expected:
        mujoco.mj_step(rebuilt, rebuilt_data)
        np.testing.assert_array_equal(
            capture_engine(model=rebuilt, data=rebuilt_data)["integration"], state
        )


def test_model_arrays_refuse_a_scene_of_another_shape():
    import mujoco

    arrays = ModelArrays.capture(model=mujoco.MjModel.from_xml_string(WELDED))
    other = mujoco.MjModel.from_xml_string(
        WELDED.replace('<body name="tool"', '<body name="x"/>\n<body name="tool"')
    )
    with pytest.raises(ValueError, match="body_mass"):
        ModelArrays.restore(model=other, arrays=arrays)


def test_controller_state_carries_plain_data_and_never_a_live_handle():
    live = object()
    owner = SimpleNamespace(
        model=live,
        data=live,
        arm_reference=np.arange(7.0),
        grip=1.0,
        held_cube=None,
        notes=["a"],
        metrics={"strokes": [{"d": 0.1, "path": np.ones(2)}]},
    )
    state = pickle.loads(pickle.dumps(ControllerState.capture(owner=owner)))
    assert set(state) == {"arm_reference", "grip", "held_cube", "notes", "metrics"}
    owner.arm_reference[0] = 99.0
    rebuilt = SimpleNamespace(
        model=live,
        data=live,
        arm_reference=np.zeros(7),
        grip=0.0,
        held_cube="x",
        notes=[],
        metrics={},
    )
    ControllerState.restore(owner=rebuilt, state=state)
    np.testing.assert_array_equal(rebuilt.arm_reference, np.arange(7.0))
    assert (rebuilt.grip, rebuilt.held_cube, rebuilt.notes) == (1.0, None, ["a"])
    assert rebuilt.model is live


def test_controller_state_refuses_what_it_cannot_carry():
    """A controller that grows a cache this module does not know stops the checkpoint
    instead of resuming without it."""
    with pytest.raises(TypeError, match="planner_cache"):
        ControllerState.capture(owner=SimpleNamespace(grip=0.0, planner_cache=object()))
    with pytest.raises(ValueError, match="held_cube"):
        ControllerState.restore(
            owner=SimpleNamespace(grip=0.0, held_cube=None), state={"grip": 1.0}
        )


def test_controller_state_restores_what_a_skill_added_after_construction():
    """`finger_force_default` exists only once a food skill has run, so a rebuilt
    controller does not have it yet (the DP arm's first resume, 2026-11-13)."""
    rebuilt = SimpleNamespace(grip=0.0)
    ControllerState.restore(owner=rebuilt, state={"grip": 1.0, "finger_force_default": 255.0})
    assert (rebuilt.grip, rebuilt.finger_force_default) == (1.0, 255.0)


def test_the_step_protocol_offers_serving_simulation_checkpoints():
    assert RackServingStepProtocol.simulation_checkpoints is True
    env = SimpleNamespace(
        simulation_checkpoint=lambda: {"saved": 1},
        restore_simulation_checkpoint=lambda *, checkpoint: restored.append(checkpoint),
    )
    restored: list[dict[str, int]] = []
    problem = SimpleNamespace(env=env)
    original = RackServingStepProtocol.env
    RackServingStepProtocol.env = staticmethod(lambda *, problem: problem.env)
    try:
        saved = RackServingStepProtocol.checkpoint_practice_environment(problem=problem)
        RackServingStepProtocol.restore_checkpoint_environment(problem=problem, checkpoint=saved)
    finally:
        RackServingStepProtocol.env = original
    assert restored == [{"saved": 1}]


def _environment():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    return RackServingEnvironment(
        config=RackServingConfig.model_validate(SWEEP_SCENE), skill_step_limit=12000
    )


def _skill(*, env, name: str, d: float = 0.0) -> None:
    env.take_action(action=np.array([SKILL_IDS[name], 0.0, 0.0, 0.0, d]))


def _observed(*, env) -> dict[str, object]:
    backend, state = env.backend(), env.get_current_state()
    shelf = env.rack_controllers().shelf
    return {
        "engine": capture_engine(model=backend.model(), data=backend.data())["integration"],
        "model": model_fingerprint(model=backend.model()),
        "state": {obj.name: np.asarray(values) for obj, values in state.data.items()},
        "atoms": state.abstract_atoms,
        "ledger": backend.ledger,
        "control_steps": backend.control_steps(),
        "error": env.last_skill_error(),
        "skill_steps": env.last_controller_steps(),
        "arm_reference": shelf.arm_reference.copy(),
        "base_target": shelf.base_target.copy(),
        "in_hand": None if shelf.in_hand is None else shelf.in_hand.copy(),
        "grip": (shelf.grip, backend.grip),
        "human_events": env.human_events(),
        "bulk_verdict": env.bulk_verdict(),
        "layout_rng": str(backend._rng.bit_generator.state),
        "human_rng": str(backend._human_rng.bit_generator.state),
    }


def _assert_identical(*, left: dict[str, object], right: dict[str, object], at: str) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if key == "state":
            assert left[key].keys() == right[key].keys(), at
            for name in left[key]:
                np.testing.assert_array_equal(
                    left[key][name], right[key][name], err_msg=f"{at} {name}"
                )
        elif isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key], err_msg=f"{at} {key}")
        else:
            assert left[key] == right[key], f"{at} {key}"


def _assert_same(*, left: object, right: object, at: str) -> None:
    """Deep equality with exact arrays; a pickle's bytes also encode object sharing."""
    assert type(left) is type(right), at
    if isinstance(left, dict):
        assert left.keys() == right.keys(), at
        for key in left:
            _assert_same(left=left[key], right=right[key], at=f"{at}.{key}")
    elif isinstance(left, list | tuple):
        assert len(left) == len(right), at
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            _assert_same(left=one, right=other, at=f"{at}[{index}]")
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=at)
    elif hasattr(left, "data") and hasattr(left, "abstract_atoms"):
        _assert_same(
            left={obj.name: np.asarray(v) for obj, v in left.data.items()},
            right={obj.name: np.asarray(v) for obj, v in right.data.items()},
            at=f"{at}.data",
        )
        assert left.abstract_atoms == right.abstract_atoms, at
    elif hasattr(left, "allclose"):
        assert left.allclose(right, atol=0.0), at
    else:
        assert left == right, at


def _both(*, envs, step, at: str) -> None:
    """Run `step` on each simulator from the same process-global RNG state (practice
    restores those streams with the checkpoint), then require identical outcomes."""
    streams = RngStates.capture_global()
    for env in envs:
        RngStates.restore_global(states=streams)
        step(env)
    _assert_identical(left=_observed(env=envs[0]), right=_observed(env=envs[1]), at=at)


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator")
def test_a_restored_serving_simulation_steps_exactly_as_the_one_it_was_saved_from():
    """Saved after a food pick and return (planned in PyBullet) and with the scraper
    welded in the hand after a Sweep, restored into a second simulator, then both
    driven through a person's batch return, a Sweep, a ReturnScraper and a planned
    PickFood: every float of the engine state and every observed fact is equal, bit
    for bit, after each."""
    original, restored = _environment(), _environment()
    try:
        original.reset_to_seed(seed=0)
        for name, d in (
            ("PickFood", 0.0),
            ("ReturnFoodToRack", 0.0),
            ("PickScraper", 0.0),
            ("Sweep", 0.06),
        ):
            _skill(env=original, name=name, d=d)
            assert original.last_skill_error() is None, name
        assert original.backend().held(name="wiper_0")
        before = _observed(env=original)
        saved = pickle.dumps(original.simulation_checkpoint())
        # Saving only reads: a run that checkpoints steps as one that does not.
        _assert_identical(left=before, right=_observed(env=original), at="after saving")
        restored.restore_simulation_checkpoint(checkpoint=pickle.loads(saved))
        _assert_identical(left=_observed(env=original), right=_observed(env=restored), at="restore")
        _assert_same(
            left=pickle.loads(saved), right=restored.simulation_checkpoint(), at="checkpoint"
        )
        envs = (original, restored)
        _both(
            envs=envs,
            step=lambda env: env.reset_movables(destination="human_batch_return"),
            at="batch return",
        )
        for name, d in (("Sweep", 0.10), ("ReturnScraper", 0.0), ("PickFood", 0.0)):
            _both(
                envs=envs, step=lambda env, name=name, d=d: _skill(env=env, name=name, d=d), at=name
            )
        assert original.backend().control_steps() > 1000
    finally:
        original.close()
        restored.close()
