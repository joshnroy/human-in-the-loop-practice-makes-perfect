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


class DerivedData:
    """Everything in `MjData` beyond the integration state, as the last step left it.

    MuJoCo computes positions, bias forces, contacts and constraint forces on the way
    into a step, so after it they are one step behind the state. Serving's controllers
    and KINDER's arm torques read them before their next step. A restore that only
    recomputes them (`mj_forward`) hands those readers numbers about 1e-14 away from the
    original's, which is enough for the two runs to part at the first tick whenever
    the scene is still moving (it is right after a scene reset). So they are carried
    and written back over the recomputed ones.
    """

    # Sparse-matrix index tables: the layout of the constraint Jacobian, which follows
    # from the contact list and is recomputed with it. With a dense Jacobian (this
    # scene's) most are never written and read back as whatever memory held.
    SPARSE_INDEX_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "_rowadr",
        "_rownnz",
        "_rowsuper",
        "_colind",
    )
    CONTACT_FIELDS: ClassVar[tuple[str, ...]] = (
        "dist",
        "pos",
        "frame",
        "includemargin",
        "friction",
        "solref",
        "solreffriction",
        "solimp",
        "mu",
        "H",
        "dim",
        "geom1",
        "geom2",
        "geom",
        "flex",
        "elem",
        "vert",
        "exclude",
        "efc_address",
    )

    @staticmethod
    def arrays(*, data: Any) -> dict[str, np.ndarray]:
        found = {}
        for name in sorted(dir(data)):
            if name.startswith("_"):
                continue
            value = getattr(data, name)
            if (
                isinstance(value, np.ndarray)
                and value.flags.writeable
                and not name.endswith(DerivedData.SPARSE_INDEX_SUFFIXES)
            ):
                found[name] = value
        return found

    @staticmethod
    def capture(*, data: Any, before_last_step: np.ndarray | None = None) -> dict[str, Any]:
        """`before_last_step` is the integration state the most recent step started
        from, when the caller tracks it (`RackServingBackend` does)."""
        contact = {}
        for name in DerivedData.CONTACT_FIELDS:
            value = getattr(data.contact, name, None)
            if isinstance(value, np.ndarray):
                contact[name] = value.copy()
        return {
            "ncon": int(data.ncon),
            "nefc": int(data.nefc),
            "arrays": {name: value.copy() for name, value in DerivedData.arrays(data=data).items()},
            "contact": contact,
            "before_last_step": None if before_last_step is None else before_last_step.copy(),
        }

    @staticmethod
    def restore(*, model: Any, data: Any, saved: dict[str, Any], state: np.ndarray) -> None:
        """Onto data already holding the checkpoint's integration `state`, recomputed.

        The arrays are written back over the recomputed ones. The contact and
        constraint lists can only be written back at their saved sizes, and a list one
        step behind can differ in size from a recomputed one (a contact made by the
        last step is in the state and not yet in the list). Then the last step is run
        again from the state it started from: it leaves exactly what it left the first
        time. That replay is used only if it lands on the checkpoint's state bit for
        bit; if something moved the scene after its last step, it does not, and the
        recomputed data are what the original had. A restore that can do neither
        raises instead of carrying on from a different scene."""
        import mujoco

        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        sizes = (saved["ncon"], saved["nefc"])
        before = saved["before_last_step"]
        if (int(data.ncon), int(data.nefc)) != sizes and before is not None:
            mujoco.mj_setState(model, data, before, spec)
            mujoco.mj_step(model, data)
            replayed = np.empty_like(state)
            mujoco.mj_getState(model, data, replayed, spec)
            if not np.array_equal(replayed, state):
                mujoco.mj_setState(model, data, state, spec)
                mujoco.mj_forward(model, data)
                mujoco.mj_setState(model, data, state, spec)
        if (int(data.ncon), int(data.nefc)) != sizes:
            raise ValueError(
                "simulation checkpoint contacts differ from the rebuilt scene's: "
                f"{sizes[0]} contacts and {sizes[1]} constraint rows saved, "
                f"{int(data.ncon)} and {int(data.nefc)} in the rebuilt scene"
            )
        live = DerivedData.arrays(data=data)
        unfit = sorted(
            name
            for name, value in saved["arrays"].items()
            if name not in live or live[name].shape != value.shape
        )
        if unfit:
            raise ValueError(f"simulation checkpoint data arrays do not fit: {unfit}")
        for name, value in saved["arrays"].items():
            live[name][...] = value
        for name, value in saved["contact"].items():
            getattr(data.contact, name)[...] = value


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
        """Onto a freshly rebuilt controller. A skill may have added attributes since
        construction (the food controller's `finger_force_default`), so the checkpoint
        may hold more than the rebuilt controller; never less."""
        unknown = sorted(set(vars(owner)) - set(state) - ControllerState.HANDLES)
        if unknown:
            raise ValueError(
                f"simulation checkpoint lacks state the rebuilt {type(owner).__name__} "
                f"holds: {unknown}"
            )
        for name, value in state.items():
            setattr(owner, name, copy.deepcopy(value))
