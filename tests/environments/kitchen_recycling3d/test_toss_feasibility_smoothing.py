"""The toss feasibility check plans without shortcut smoothing; the toss controller
still smooths.

`KinderBackend.toss_base_plan_failure` only asks whether a base path exists. BiRRT
smooths a path after it has found one, so smoothing cannot change that answer -- it only
costs time. The stands here are recorded kitchen dispatch states where the check falls
through to BiRRT and succeeds, plus one it refuses.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.toss_direction import TossDirectionSelector

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("kitchen_recycling3d_toss_feasibility_smoothing"),
]

FIXTURE = Path(__file__).parent / "fixtures" / "curtis0_toss_feasibility_stands.json"
RECORDED = json.loads(FIXTURE.read_text())
STANDS = [
    (index, stand) for index, entry in enumerate(RECORDED["states"]) for stand in entry["stands"]
]


@pytest.fixture(scope="module")
def env():
    environment = KitchenRecycling3DEnvironment(
        kitchen_config=KitchenConfig.model_validate(RECORDED["kitchen_config"]), scene_bg=False
    )
    environment.reset_to_seed(seed=RECORDED["run_seed"])
    yield environment
    environment.close()


def dispatch_state(*, env: KitchenRecycling3DEnvironment, index: int) -> Any:
    backend = env.kinder_backend()
    backend.restore(
        snapshot=backend.plain_to_snapshot(
            plain=RECORDED["states"][index]["dispatch_state"], dtype=np.float64
        )
    )
    return env._observed_state(seed=0, steps_taken=0)  # noqa: SLF001


class SmoothingSpy:
    """Records the `smooth_amt` of every BiRRT that found a path and smoothed it."""

    def __init__(self, *, monkeypatch: pytest.MonkeyPatch) -> None:
        from prpl_utils.motion_planning import RRT

        self.smooth_amts: list[int] = []
        smooth = RRT._smooth_path  # noqa: SLF001

        def recorded(rrt: Any, path: list[Any]) -> list[Any]:  # noqa: PLR0917 (a method)
            self.smooth_amts.append(rrt._smooth_amt)  # noqa: SLF001
            return smooth(rrt, path)

        monkeypatch.setattr(RRT, "_smooth_path", recorded)


def check(*, state: Any, stand: dict[str, Any]) -> str | None:
    return KinderBackend.toss_base_plan_failure(
        snapshot=state.object_centric,
        distance=stand["distance"],
        rotation=TossDirectionSelector.rotation(direction_deg=stand["direction_deg"]),
        drive_zone=state.drive_zone,
    )


def test_the_feasibility_check_finds_its_paths_without_smoothing_them(*, env, monkeypatch):
    spy = SmoothingSpy(monkeypatch=monkeypatch)
    for index, stand in STANDS:
        if stand["plannable"]:
            check(state=dispatch_state(env=env, index=index), stand=stand)
    # Every recorded success fell through to BiRRT, so a path was found and handed on.
    assert spy.smooth_amts
    assert set(spy.smooth_amts) == {0}


@pytest.mark.parametrize(("index", "stand"), STANDS)
def test_skipping_smoothing_leaves_the_verdict_unchanged(*, env, monkeypatch, index, stand):
    import kinder_models.dynamic3d.utils as kinder_utils

    state = dispatch_state(env=env, index=index)
    verdict = check(state=state, stand=stand)

    planner = kinder_utils.run_base_motion_planning

    def smoothing(**kwargs: Any) -> Any:
        return planner(**{**kwargs, "smooth_amt": 50})

    monkeypatch.setattr(kinder_utils, "run_base_motion_planning", smoothing)
    smoothed_verdict = check(state=state, stand=stand)

    assert verdict == smoothed_verdict
    assert (verdict is None) == stand["plannable"]


def test_the_toss_controller_still_smooths_its_base_path(*, env, monkeypatch):
    spy = SmoothingSpy(monkeypatch=monkeypatch)
    index, stand = next((i, s) for i, s in STANDS if s["plannable"])
    state = dispatch_state(env=env, index=index)
    backend = env.kinder_backend()
    controller = backend._ground_controller(  # noqa: SLF001
        module="tossing",
        key="move_to_toss_location_and_toss",
        object_names=(backend.robot_name, backend.cube_name, backend.barrier_name),
        state=state.object_centric,
    )
    rotation = TossDirectionSelector.rotation(direction_deg=stand["direction_deg"])
    controller.reset(state.object_centric, np.array([stand["distance"], rotation, 10.0, 300.0]))
    assert spy.smooth_amts
    assert set(spy.smooth_amts) == {50}
