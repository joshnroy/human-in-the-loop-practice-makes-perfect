"""Process-portable continuation of the kitchen scene between skills.

Tossing's engine checkpoint (MuJoCo integration state under a compiled-model
fingerprint) plus what the kitchen keeps outside it: weld targets written into the
model, the IK scratch data, and the bookkeeping the native controller, the KinDER scene
and its robot hold in Python.
"""

import copy
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.tossing3d.simulation_checkpoint import capture_engine, restore_engine


class PlainState:
    """An owner's attributes that are data, as opposed to handles on the live scene.

    Taken wholesale rather than from a list of names: a controller field added later is
    carried without anyone remembering to add it here.
    """

    SCALARS: ClassVar[tuple[type, ...]] = (bool, int, float, complex, str, bytes, np.generic)

    @staticmethod
    def attributes(*, owner: Any) -> dict[str, Any]:
        return {**vars(owner), **(getattr(owner, "__pydantic_private__", None) or {})}

    @staticmethod
    def is_data(*, value: Any, engine: list[np.ndarray]) -> bool:
        if value is None or isinstance(value, (*PlainState.SCALARS, Rotation)):
            return True
        if isinstance(value, np.ndarray):
            # A view of MuJoCo's buffers (the robot's `qpos`/`ctrl` references) must stay
            # a view: the engine restore fills it, and a copy would cut it loose.
            return value.dtype != object and not any(
                np.may_share_memory(value, buffer) for buffer in engine
            )
        if isinstance(value, tuple | list | set | frozenset):
            return all(PlainState.is_data(value=item, engine=engine) for item in value)
        if isinstance(value, dict):
            return all(
                PlainState.is_data(value=item, engine=engine)
                for pair in value.items()
                for item in pair
            )
        if isinstance(value, BaseModel):
            return PlainState.is_data(value=PlainState.attributes(owner=value), engine=engine)
        return type(value).__module__.startswith("relational_structs")

    @staticmethod
    def capture(
        *, owner: Any, engine: list[np.ndarray], skip: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        attributes = {
            name: value
            for name, value in PlainState.attributes(owner=owner).items()
            if name not in skip
        }
        return {
            "data": {
                name: copy.deepcopy(value)
                for name, value in attributes.items()
                if PlainState.is_data(value=value, engine=engine)
            },
            "rngs": {
                name: copy.deepcopy(value.bit_generator.state)
                for name, value in attributes.items()
                if isinstance(value, np.random.Generator)
            },
        }

    @staticmethod
    def restore(*, owner: Any, checkpoint: dict[str, Any]) -> None:
        for name, value in checkpoint["data"].items():
            setattr(owner, name, copy.deepcopy(value))
        attributes = PlainState.attributes(owner=owner)
        for name, state in checkpoint["rngs"].items():
            generator = attributes.get(name)
            if not isinstance(generator, np.random.Generator):
                raise ValueError(
                    f"simulation checkpoint RNG {name!r} is absent after scene rebuild"
                )
            generator.bit_generator.state = state


class KitchenSimulationCheckpoint:
    """Capture and restore of one `KitchenController` and everything under it."""

    VERSION: ClassVar[int] = 1
    # Model arrays the kitchen writes while it runs: each weld's target pose (the hand's
    # grasp of the jug or a dumpling, the skillet's and the parked jug's stands) and the
    # hob's colours. Everything else in the model stays under the fingerprint.
    MODEL_EDITS: ClassVar[tuple[str, ...]] = ("eq_data", "geom_rgba")
    # Rebound by the environment after every scene build; never part of the scene.
    CONTROLLER_HOOKS: ClassVar[tuple[str, ...]] = ("callback", "before_step", "after_step")

    @staticmethod
    def owners(*, controller: Any) -> list[tuple[Any, tuple[str, ...]]]:
        """The Gym wrapper chain, KinDER's scene, robot and reward counter, then the
        native controller: every object that holds scene bookkeeping in Python."""
        chain = []
        current = controller.env
        while current is not None:
            chain.append(current)
            current = vars(current).get("env")
        scene = controller.scene
        found: list[tuple[Any, tuple[str, ...]]] = []
        for owner in (*chain, scene, scene._robot_env, scene._reward_calculator):  # noqa: SLF001
            if all(owner is not existing for existing, _ in found):
                found.append((owner, ()))
        found.append((controller, KitchenSimulationCheckpoint.CONTROLLER_HOOKS))
        return found

    @staticmethod
    def identities(*, owners: list[tuple[Any, tuple[str, ...]]]) -> list[tuple[str, str]]:
        return [(type(owner).__module__, type(owner).__qualname__) for owner, _ in owners]

    @staticmethod
    def engine_buffers(*, controller: Any) -> list[np.ndarray]:
        return [
            value
            for holder in (controller.model, controller.data, controller.scratch)
            for name in dir(holder)
            if not name.startswith("_")
            and isinstance(value := getattr(holder, name), np.ndarray)
            and value.size
        ]

    @staticmethod
    def capture(*, controller: Any) -> dict[str, Any]:
        import mujoco

        c = controller
        scratch = np.empty(
            mujoco.mj_stateSize(c.model, mujoco.mjtState.mjSTATE_INTEGRATION), dtype=np.float64
        )
        mujoco.mj_getState(c.model, c.scratch, scratch, mujoco.mjtState.mjSTATE_INTEGRATION)
        engine = {
            "model_edits": {
                name: getattr(c.model, name).copy()
                for name in KitchenSimulationCheckpoint.MODEL_EDITS
            },
            "data": capture_engine(model=c.model, data=c.data),
            "scratch": scratch,
        }
        owners = KitchenSimulationCheckpoint.owners(controller=c)
        buffers = KitchenSimulationCheckpoint.engine_buffers(controller=c)
        checkpoint = {
            "version": KitchenSimulationCheckpoint.VERSION,
            "engine": engine,
            "owners": KitchenSimulationCheckpoint.identities(owners=owners),
            "python": [
                PlainState.capture(owner=owner, engine=buffers, skip=skip) for owner, skip in owners
            ],
        }
        # After `mj_step`, MuJoCo's derived quantities (body poses, bias forces, contacts)
        # still describe the instant before the step, and the controllers read them. A
        # restore can only recompute them at the captured instant, so the scene carrying
        # on here is put through the same restore: both then continue from one state.
        KitchenSimulationCheckpoint.restore_engine_state(controller=c, engine=engine)
        return checkpoint

    @staticmethod
    def restore_engine_state(*, controller: Any, engine: dict[str, Any]) -> None:
        import mujoco

        c = controller
        for name, values in engine["model_edits"].items():
            target = getattr(c.model, name)
            if target.shape != values.shape:
                raise ValueError(f"simulation checkpoint model array {name} has another shape")
            target[...] = values
        restore_engine(model=c.model, data=c.data, checkpoint=engine["data"])
        mujoco.mj_setState(
            c.model, c.scratch, engine["scratch"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(c.model, c.scratch)
        mujoco.mj_setState(
            c.model, c.scratch, engine["scratch"], mujoco.mjtState.mjSTATE_INTEGRATION
        )

    @staticmethod
    def restore(*, controller: Any, checkpoint: dict[str, Any]) -> None:
        """Into a controller just built from the same seed and configuration."""
        if checkpoint["version"] != KitchenSimulationCheckpoint.VERSION:
            raise ValueError("kitchen simulation checkpoint schema differs")
        owners = KitchenSimulationCheckpoint.owners(controller=controller)
        if KitchenSimulationCheckpoint.identities(owners=owners) != [
            tuple(identity) for identity in checkpoint["owners"]
        ]:
            raise ValueError("kitchen simulation checkpoint Gym/KinDER runtime topology differs")
        KitchenSimulationCheckpoint.restore_engine_state(
            controller=controller, engine=checkpoint["engine"]
        )
        for (owner, _), state in zip(owners, checkpoint["python"], strict=True):
            PlainState.restore(owner=owner, checkpoint=state)
