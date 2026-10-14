"""SweepOffShelf3D's skill controllers, driving the TidyBot through KINDER's 11-D action.

PORT PENDING: the controllers come from the scripted solvability check
(`scripts/validate_sweep_off_shelf3d.py`); until they land every skill fails cleanly.
"""

from pydantic import BaseModel, ConfigDict

from .backend import SweepOffShelf3DBackend


class ControllerFailure(Exception):
    """A skill gave up: unreachable target, failed convergence or step budget."""


class SweepOffShelf3DControllers(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: SweepOffShelf3DBackend
    step_limit: int = 8000
    steps: int = 0

    def reset_counter(self) -> None:
        self.steps = 0

    def on_scene_reset(self) -> None:
        """The scene was rebuilt: forget any grasp bookkeeping."""

    def on_objects_moved(self) -> None:
        """A human moved objects: anything held may have been taken."""

    def pick_wiper(self) -> None:
        raise ControllerFailure("PickWiper not implemented")

    def sweep_shelf(self, *, height: float, speed: float) -> None:
        del height, speed
        raise ControllerFailure("SweepShelf not implemented")

    def return_wiper(self) -> None:
        raise ControllerFailure("ReturnWiper not implemented")

    def pick_cube_from_shelf(self, *, cube: str) -> None:
        del cube
        raise ControllerFailure("PickCubeFromShelf not implemented")

    def pick_cube_from_floor(self, *, cube: str) -> None:
        del cube
        raise ControllerFailure("PickCubeFromFloor not implemented")

    def place_cube_in_box(self, *, cube: str) -> None:
        del cube
        raise ControllerFailure("PlaceCubeInBox not implemented")
