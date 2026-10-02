"""Shorter transport routes never bypass carried collision checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


@pytest.mark.parametrize("short_blocked", [False, True])
def test_selects_shortest_fully_checked_route(*, monkeypatch, short_blocked: bool) -> None:
    native = [(0.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1.0, 0.0, 0.0)]
    short = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0)]
    base = [None]
    checked, driven, logged = [], [], []
    bodies = {4, 5, 6}

    def sync(*, base=None) -> None:
        current[0] = base

    current = base

    def collision(**kwargs) -> bool:  # noqa: ANN003 -- fake native planning callback
        assert kwargs["bodies"] == bodies
        assert kwargs["held"] == 7
        checked.append(current[0])
        return short_blocked and current[0] == short[1]

    def drive(**kwargs) -> bool:  # noqa: ANN003 -- fake physical motion callback
        driven.append(kwargs["path"])
        kwargs["tick_guard"]()
        return True

    scene = SimpleNamespace(
        sync=sync, ee_now=lambda: Pose((0.0, 0.0, 0.0)), bodies=lambda: bodies,
        planning_fingers=lambda **kwargs: np.zeros(13), in_collision=collision,
        wiper_body=7, capture_path_rejections=False, _last_collision_rejection={},
    )
    session = SimpleNamespace(
        arm=lambda: np.zeros(7), position=lambda **kwargs: np.zeros(3),
        quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0), ticks=3,
        mj_data=SimpleNamespace(qpos=np.zeros(2)), _write=lambda **kwargs: logged.append(kwargs),
    )
    primitive = FloorPrimitives.model_construct(
        scene=scene, session=session, motion=SimpleNamespace(drive=drive)
    )
    monkeypatch.setattr(FloorPrimitives, "transport_base_candidates",
                        lambda self, **kwargs: [("native", native), ("upper_aisle", short)])
    monkeypatch.setattr(FloorPrimitives, "require_handle", lambda self, **kwargs: None)
    primitive.transport_wiper(target=(1.0, 0.0, 0.0))
    assert driven == [native if short_blocked else short]
    assert checked[:3] == native
    assert short[1] in checked
    selected = [x["record"] for x in logged if x["record"]["kind"] == "transport_route_selected"]
    assert selected[0]["path"] == driven[0]


def test_native_blocked_aisle_leg_is_never_inserted(*, monkeypatch) -> None:
    import kinder_models.dynamic3d.utils as native

    calls = []
    fallback = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

    def plan(*, target, start=None):
        calls.append((start, target))
        return fallback if start is None else None

    monkeypatch.setattr(native, "get_bounding_box", lambda *args: (0.4, 0.4, 0.5))
    core = SimpleNamespace(task_config={"regions": {"sweep_region": {
        "ranges": [(0.0, 1.0, 1.0, 1.2)],
    }}})
    session = SimpleNamespace(
        base=lambda: (0.0, 0.0, 0.0),
        state=SimpleNamespace(get_objects=lambda *args: [object()]),
        env=SimpleNamespace(unwrapped=SimpleNamespace(_object_centric_env=core)),
    )
    primitive = FloorPrimitives.model_construct(
        session=session, scene=SimpleNamespace(plan_base=plan)
    )
    assert primitive.transport_base_candidates(target=(2.0, 0.0, 0.0)) == [("native", fallback)]
    assert len(calls) == 2
    assert calls[1][0] == session.base()
    assert calls[1][1][1] > 1.2 + 0.2  # Native region plus chassis clearance.


def test_route_burden_wraps_every_heading_segment() -> None:
    path = [(0.0, 0.0, 3.13), (0.0, 0.0, -3.13)] * 5
    assert FloorPrimitives.transport_route_burden(path=path) < 0.22
