"""Fixed home dispatch must not inherit previously sampled pickup parameters."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.sweep_simple3d import environment as module
from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment


def test_fixed_home_after_different_pickups_keeps_one_action_per_dispatch(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = State(data={})
    env = SweepSimpleEnvironment.model_construct(current_state=state)
    calls = []
    events = []
    endings = []
    session = SimpleNamespace(
        ticks=0, begin=lambda **kwargs: None, end=lambda **kwargs: endings.append(kwargs)
    )
    primitive = SimpleNamespace(distance=0.7, heading_offset=0.0)

    def record(*, name: str) -> None:
        calls.append((name, primitive.distance, primitive.heading_offset))
        session.ticks += 3

    primitive.recover_wiper = lambda: record(name="pickup")
    primitive.place_wiper_at_start = lambda: record(name="home")
    env._session = session
    env._primitive = primitive
    # Isolate dispatch from physical predicates while retaining real skill IDs,
    # parameter interpretation, primitive reuse, and action accounting.
    monkeypatch.setattr(
        module, "GroundSkill", lambda **kwargs: SimpleNamespace(preconditions=(), add_effects=())
    )
    monkeypatch.setattr(SweepSimpleEnvironment, "observe", lambda self: state)
    monkeypatch.setattr(
        SweepSimpleEnvironment, "_write_event", lambda self, *, event: events.append(event)
    )
    pickup_id = env.ACTION_NAMES.index("PickFloorWiper")
    home_id = env.ACTION_NAMES.index("PlaceWiperAtStart")

    for distance, heading in ((0.55, -0.2), (0.85, 0.2)):
        env.take_action(action=np.array([pickup_id, -1, distance, heading]))
        # Fixed-skill action slots are deliberately nondefault as well.
        env.take_action(action=np.array([home_id, -1, 0.63, -0.1]))

    assert calls == [
        ("pickup", 0.55, -0.2),
        ("home", 0.7, 0.0),
        ("pickup", 0.85, 0.2),
        ("home", 0.7, 0.0),
    ]
    assert env._action_count == 4
    assert [event["index"] for event in events] == [1, 2, 3, 4]
    assert [event["name"] for event in events] == [
        "PickFloorWiper",
        "PlaceWiperAtStart",
        "PickFloorWiper",
        "PlaceWiperAtStart",
    ]
    assert all(event["ticks"] == 3 and event["symbolic_success"] for event in events)
    assert len(endings) == 4
    assert env._hard_reset_count == env._human_reset_count == 0
