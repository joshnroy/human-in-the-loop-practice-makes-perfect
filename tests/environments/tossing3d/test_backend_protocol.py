"""`Tossing3DEnvironment` runs on any `Tossing3DBackend`, not only on KINDER's."""

from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.types import (
    AbstractAtom,
    ControllerRun,
    Tossing3DBackend,
    Tossing3DObservation,
)

from .observations import INITIAL_ATOMS, observation


class ScriptedBackend:
    """The smallest backend that satisfies the protocol: it records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.seed: int | None = None

    def reset(self, *, seed: int) -> Tossing3DObservation:
        self.calls.append(f"reset:{seed}")
        self.seed = seed
        return self.observe()

    def close(self) -> None:
        self.calls.append("close")

    def observe(self) -> Tossing3DObservation:
        return observation()

    def snapshot(self) -> Any:
        return {"seed": self.seed}

    def restore(self, *, snapshot: Any) -> Tossing3DObservation:
        self.seed = snapshot["seed"]
        return self.observe()

    def abstract_atoms(self, *, state: Any = None) -> frozenset[AbstractAtom]:
        del state
        return INITIAL_ATOMS

    def abstraction_diagnostics(self, *, state: Any = None) -> dict[str, Any]:
        del state
        return {}

    def check_goals(self) -> bool:
        return False

    def reset_cube_and_bin(
        self, *, bin_region: dict[str, object] | None = None
    ) -> Tossing3DObservation:
        self.calls.append(f"reset_cube_and_bin:{bin_region}")
        return self.observe()

    def run_pick_cube(self) -> ControllerRun:
        self.calls.append("pick")
        return ControllerRun(steps=1, terminated=True)

    def run_open_gripper(self) -> ControllerRun:
        self.calls.append("open_gripper")
        return ControllerRun(steps=1, terminated=True)

    def run_move_to_toss_location_and_toss(
        self,
        *,
        distance: float,
        rotation: float,
        release_speed_deg_s: float,
        gripper_release_ms: float,
    ) -> ControllerRun:
        self.calls.append(f"toss:{distance}:{rotation}:{release_speed_deg_s}:{gripper_release_ms}")
        return ControllerRun(steps=1, terminated=True)

    def set_substep_recording(self, *, enabled: bool) -> None:
        del enabled

    def drain_substep_frames(self) -> list[np.ndarray]:
        return []

    def drain_substep_states(self) -> list[Any]:
        return []

    def set_control_step_observers(self, *, before: Any, after: Any) -> None:
        del before, after

    def snapshot_to_plain(self, *, snapshot: Any) -> dict[str, list[float]]:
        return {"seed": [float(snapshot["seed"])]}

    def plain_to_snapshot(
        self, *, plain: dict[str, list[float]], dtype: type[np.floating] = np.float32
    ) -> Any:
        del dtype
        return {"seed": int(plain["seed"][0])}

    def live_state(self) -> Any:
        return self.snapshot()

    def render(self, *, follow_robot: bool = False) -> np.ndarray:
        del follow_robot
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def render_fps(self) -> int:
        return 1


class ScriptedEnvironment(Tossing3DEnvironment):
    def build_backend(self) -> Tossing3DBackend:
        return ScriptedBackend()


def test_kinder_backend_satisfies_the_protocol(*, no_kinder_import) -> None:
    del no_kinder_import
    assert isinstance(KinderBackend(), Tossing3DBackend)


def test_a_scripted_backend_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedBackend(), Tossing3DBackend)


def test_the_environment_drives_whatever_backend_build_backend_returns() -> None:
    env = ScriptedEnvironment()
    state = env.reset_to_seed(seed=7)
    assert state.get(obj=env.scene, feature_name="seed") == 7
    env.take_action(action=np.array([float(env.pick_cube_id), 0, 0, 0, 0]))
    env.take_action(
        action=np.array([float(env.move_to_toss_location_and_toss_id), 1.0, 0.5, 120, 600])
    )
    backend = env.backend()
    assert isinstance(backend, ScriptedBackend)
    assert backend.calls == ["reset:7", "pick", "toss:1.0:0.5:120.0:600.0"]


def test_the_default_backend_is_kinder(*, no_kinder_import) -> None:
    del no_kinder_import
    assert isinstance(Tossing3DEnvironment().backend(), KinderBackend)


def test_kinder_backend_accessor_refuses_another_backend() -> None:
    with pytest.raises(TypeError, match="KinderBackend"):
        ScriptedEnvironment().kinder_backend()
