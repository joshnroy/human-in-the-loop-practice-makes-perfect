"""A human reset whose robot blocks every pickable cube spot steps the robot back to its
scene-start pose and tries again, as it already does when the robot blocks the bin."""

from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.kitchen_recycling3d.backend import KitchenKinderBackend


class FakeState:
    def __init__(self, *, robot_xy: tuple[float, float]) -> None:
        self.robot_xy = robot_xy

    def copy(self) -> "FakeState":
        return self

    def get_object_from_name(self, name: str) -> str:  # noqa: PLR0917
        return name

    def get(self, obj: str, feature: str) -> float:  # noqa: PLR0917
        return {"pos_base_x": self.robot_xy[0], "pos_base_y": self.robot_xy[1]}[feature]


class ParkedRobot:
    """The robot is parked where no cube spot is pickable until it steps back."""

    def __init__(self) -> None:
        self.state = FakeState(robot_xy=(3.0, 0.0))
        self.moved_back = 0
        self.tried_from: list[tuple[float, float]] = []

    def install(self, *, monkeypatch: pytest.MonkeyPatch) -> KitchenKinderBackend:
        backend = KitchenKinderBackend.model_construct()
        backend._placements = 0  # noqa: SLF001
        backend._timing = None  # noqa: SLF001
        backend._robot_name = "robot"  # noqa: SLF001
        backend.reset_rejections = []
        cls = KitchenKinderBackend

        def require_state(self: Any) -> FakeState:  # noqa: PLR0917
            return fake.state

        def bin_destination(self: Any, *, side: str, state: Any) -> Any:  # noqa: PLR0917
            return (2.0, 0.0, 0.0)

        def recovery_positions(self: Any, *, robot_xy: np.ndarray, bin_xyz: Any) -> Any:  # noqa: PLR0917
            for i in range(1000):
                fake.tried_from.append((float(robot_xy[0]), float(robot_xy[1])))
                yield (1.0 + 0.001 * i, 0.5)

        def move_back(self: Any) -> None:  # noqa: PLR0917
            fake.moved_back += 1
            fake.state = FakeState(robot_xy=(0.0, 0.0))

        fake = self
        monkeypatch.setattr(cls, "_require_state", require_state)
        monkeypatch.setattr(cls, "_bin_destination", bin_destination)
        monkeypatch.setattr(cls, "_recovery_positions", recovery_positions)
        monkeypatch.setattr(cls, "_write_placement", lambda self, **_: None)
        monkeypatch.setattr(
            cls, "_pick_plannable_here", lambda self: fake.state.robot_xy == (0.0, 0.0)
        )
        monkeypatch.setattr(cls, "_move_robot_to_start_pose", move_back)
        monkeypatch.setattr(cls, "observe", lambda self: "observed")
        return backend


def test_a_robot_blocking_every_cube_spot_steps_back_and_the_reset_succeeds(*, monkeypatch):
    fake = ParkedRobot()
    backend = fake.install(monkeypatch=monkeypatch)

    assert backend._place_cube_and_bin(side="robot_side") == "observed"  # noqa: SLF001
    assert fake.moved_back == 1
    assert fake.tried_from[-1] == (0.0, 0.0)
    assert backend._placements == 1  # noqa: SLF001


def test_a_reset_still_raises_when_even_the_start_pose_blocks_every_spot(*, monkeypatch):
    fake = ParkedRobot()
    backend = fake.install(monkeypatch=monkeypatch)
    monkeypatch.setattr(KitchenKinderBackend, "_pick_plannable_here", lambda self: False)

    with pytest.raises(ValueError, match="seeded candidates"):
        backend._place_cube_and_bin(side="robot_side")  # noqa: SLF001
    assert fake.moved_back == 1
