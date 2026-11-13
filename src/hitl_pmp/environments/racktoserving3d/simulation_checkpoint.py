"""What Serving carries across processes beyond Tossing's MuJoCo engine state.

Tossing's `simulation_checkpoint` module saves the engine's integration state and
refuses a rebuilt scene whose compiled model differs. Serving has two things that
module does not: skills that write into the compiled model (a grasp weld's relative
pose is `model.eq_data`, which no MuJoCo state vector holds), and scripted controllers
that keep targets, grasp transforms and counters from one skill to the next.

Only used between skills, as Tossing's is.
"""

import copy
from typing import Any, ClassVar

import numpy as np


class ModelArrays:
    """The compiled model's parameter tables, small enough to travel with a checkpoint.

    Meshes and textures are megabytes and nothing writes them; the engine checkpoint's
    model fingerprint covers them, so a table left out here that did change still stops
    the restore.
    """

    LIMIT_BYTES: ClassVar[int] = 1 << 16

    @staticmethod
    def capture(*, model: Any) -> dict[str, np.ndarray]:
        arrays = {}
        for name in sorted(dir(model)):
            if name.startswith("_"):
                continue
            value = getattr(model, name)
            if (
                isinstance(value, np.ndarray)
                and value.flags.writeable
                and value.nbytes <= ModelArrays.LIMIT_BYTES
            ):
                arrays[name] = value.copy()
        return arrays

    @staticmethod
    def restore(*, model: Any, arrays: dict[str, np.ndarray]) -> None:
        """Write back every table that differs; all are checked before any is written."""
        live: dict[str, Any] = {name: getattr(model, name, None) for name in arrays}
        unfit = sorted(
            name
            for name, saved in arrays.items()
            if not isinstance(live[name], np.ndarray)
            or live[name].shape != saved.shape
            or live[name].dtype != saved.dtype
        )
        if unfit:
            raise ValueError(
                f"simulation checkpoint model arrays do not fit the rebuilt scene: {unfit}"
            )
        for name, saved in arrays.items():
            if not np.array_equal(live[name], saved):
                live[name][...] = saved


class ControllerState:
    """A scripted controller's own data, by attribute name, without its live handles.

    A controller is rebuilt against the rebuilt scene, so its handles are already the
    right ones; everything else it holds is copied over. An attribute that is neither a
    known handle nor plain data stops the checkpoint: a cache this class cannot carry
    must not be silently left behind.
    """

    HANDLES: ClassVar[frozenset[str]] = frozenset({
        "backend",
        "core",
        "data",
        "env",
        "ik_data",
        "model",
        "shelf",
        "sim",
        "tool",
    })

    @staticmethod
    def plain(*, value: Any) -> bool:
        if value is None or isinstance(value, bool | int | float | str | np.generic | np.ndarray):
            return True
        if isinstance(value, list | tuple | set | frozenset):
            return all(ControllerState.plain(value=item) for item in value)
        if isinstance(value, dict):
            return all(
                ControllerState.plain(value=key) and ControllerState.plain(value=item)
                for key, item in value.items()
            )
        return False

    @staticmethod
    def capture(*, owner: Any) -> dict[str, Any]:
        state = {}
        for name, value in vars(owner).items():
            if name in ControllerState.HANDLES:
                continue
            if not ControllerState.plain(value=value):
                raise TypeError(
                    f"simulation checkpoint cannot carry {type(owner).__name__}.{name} "
                    f"({type(value).__name__})"
                )
            state[name] = copy.deepcopy(value)
        return state

    @staticmethod
    def restore(*, owner: Any, state: dict[str, Any]) -> None:
        absent = sorted(set(state) - set(vars(owner)))
        if absent:
            raise ValueError(
                f"simulation checkpoint has {type(owner).__name__} state the rebuilt "
                f"controller lacks: {absent}"
            )
        for name, value in state.items():
            setattr(owner, name, copy.deepcopy(value))
