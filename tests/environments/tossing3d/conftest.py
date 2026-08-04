"""Shared fixtures for the Tossing3D tests.

Everything under `environments/tossing3d/` except `kinder_backend.py` is pure
arithmetic over feature vectors, so these tests build `State`s by hand and never open
MuJoCo. `kinder_available` marks the handful of tests that genuinely need the
simulator (see `test_kinder_fidelity.py`), which are skipped where `kindergarden` is
not installed -- CI included.
"""

import importlib.util

import pytest

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

# KINDER's blocks_goal_region for both Tossing3D variants, as it appears in the task
# JSON: (x_min, y_min, z_min, x_max, y_max, z_max).
GOAL_REGION = (1.90, -0.10, 0.0, 2.10, 0.10, 0.10)
# The barrier's x from the same JSON's barrier_init_region.
BARRIER_X = 1.3
# KINDER's bin_init_region x for the o1 variant.
BIN_X = 2.2305


def build_state(
    *,
    env: Tossing3DEnvironment | None = None,
    cube: tuple[float, float, float] = (0.65, -0.1, 0.025),
    base: tuple[float, float, float] = (0.0, 0.0, 0.0),
    holding: float = 0.0,
    seed: int = 0,
) -> State:
    """A Tossing3D `State` with everything but the interesting bits pinned to the real
    scene's values."""
    environment = env if env is not None else Tossing3DEnvironment()
    return environment.build_state(
        features={
            "robot": (base[0], base[1], base[2], holding),
            "cube": cube,
            "bin": (BIN_X, 0.0, 0.0),
            "barrier": (BARRIER_X, 0.0, 0.05),
        },
        seed=seed,
        region=GOAL_REGION,
    )


def throw_pose_base() -> tuple[float, float, float]:
    """A base pose exactly `throw_standoff` metres from the bin, facing it."""
    return (BIN_X - Tossing3DEnvironment.throw_standoff, 0.0, 0.0)


kinder_available = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None,
    reason="kindergarden is an optional dependency -- see environments/tossing3d/README.md",
)
