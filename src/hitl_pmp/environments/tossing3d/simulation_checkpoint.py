"""Process-portable MuJoCo continuation state, separate from observation snapshots.

Only used between skills. Controllers have completed, and physics model topology and
parameters must match the seeded rebuilt scene. No observation round-trip is involved.
"""

import copy
import hashlib
from typing import Any

import numpy as np


def model_fingerprint(*, model: Any) -> str:
    """Hash compiled numeric model data, names and options; reject a different scene."""
    digest = hashlib.sha256()
    for prefix, owner in (("model", model), ("option", model.opt)):
        for name in sorted(dir(owner)):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if isinstance(value, np.ndarray):
                digest.update(f"{prefix}.{name}:{value.dtype}:{value.shape}".encode())
                digest.update(value.tobytes())
            elif isinstance(value, int | float | str | bytes):
                digest.update(f"{prefix}.{name}:{value!r}".encode())
    return digest.hexdigest()


def capture_engine(*, model: Any, data: Any) -> dict[str, Any]:
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return {
        "version": 1,
        "mujoco_version": mujoco.__version__,
        "model_sha256": model_fingerprint(model=model),
        "integration": state,
    }


def restore_engine(*, model: Any, data: Any, checkpoint: dict[str, Any]) -> None:
    import mujoco

    if checkpoint["version"] != 1 or checkpoint["mujoco_version"] != mujoco.__version__:
        raise ValueError("simulation checkpoint schema or MuJoCo version differs")
    if checkpoint["model_sha256"] != model_fingerprint(model=model):
        raise ValueError("simulation checkpoint compiled model fingerprint differs")
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = checkpoint["integration"]
    if state.shape != (mujoco.mj_stateSize(model, spec),) or state.dtype != np.float64:
        raise ValueError("simulation checkpoint integration state has incompatible shape/dtype")
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)
    # Forward recomputes derived quantities; restore every integration input again,
    # including solver warmstart, so it cannot perturb the next integration step.
    mujoco.mj_setState(model, data, state, spec)


RUNTIME_COUNTERS = ("timestep", "_elapsed_steps", "_has_reset", "_reset_retries", "_np_random_seed")


def capture_runtime(*, owner: Any) -> dict[str, Any]:
    """KINDER/Gym counters and every directly held NumPy RNG, without live handles."""
    return {
        "counters": {
            name: copy.deepcopy(vars(owner)[name])
            for name in RUNTIME_COUNTERS
            if name in vars(owner)
        },
        "rngs": {
            name: copy.deepcopy(value.bit_generator.state)
            for name, value in vars(owner).items()
            if isinstance(value, np.random.Generator)
        },
    }


def restore_runtime(*, owner: Any, checkpoint: dict[str, Any]) -> None:
    for name, value in checkpoint["counters"].items():
        setattr(owner, name, value)
    for name, state in checkpoint["rngs"].items():
        generator = vars(owner).get(name)
        if not isinstance(generator, np.random.Generator):
            raise ValueError(f"simulation checkpoint RNG {name!r} is absent after scene rebuild")
        generator.bit_generator.state = state
