"""SweepOffShelf3D's skill controllers, run on the backend's env.

`KinderSkills` (kinder_skills.py) drives the TidyBot through KINDER's 11-D action, with
KINDER's controllers where KINDER has one and `ShelfSkills` (scripted_skills.py) where
it does not. This wrapper gives each planner skill its own step budget, turns
`SkillFailed` into `ControllerFailure`, rebuilds the skills whenever the scene is
rebuilt, and keeps their grasp bookkeeping honest after a human moves things.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .backend import SweepOffShelf3DBackend
from .kinder_skills import KinderSkills
from .scripted_skills import ShelfSkills, SkillFailed
from .types import SkillNote


class ControllerFailure(Exception):
    """A skill gave up: unreachable target, failed convergence or step budget."""


class SweepOffShelf3DControllers(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: SweepOffShelf3DBackend
    step_limit: int = 12000
    steps: int = 0

    _skills: ShelfSkills | None = PrivateAttr(default=None)
    _kinder: KinderSkills | None = PrivateAttr(default=None)

    def reset_counter(self) -> None:
        self.steps = 0

    def clear_notes(self) -> None:
        if self._kinder is not None:
            self._kinder.notes = []
        if self._skills is not None:
            self._skills.notes = []

    def notes(self) -> list[SkillNote]:
        """What the last skill tried: KINDER candidates first, then the scripted
        routines' retries (strings there, kept as `scripted` notes)."""
        kinder = [] if self._kinder is None else list(self._kinder.notes)
        scripted = [] if self._skills is None else self._skills.notes
        return kinder + [SkillNote(stage="scripted", kind="scripted", reason=n) for n in scripted]

    def on_scene_reset(self) -> None:
        """The scene was rebuilt: the skills re-read it from scratch."""
        self._skills = ShelfSkills(env=self.backend.env(), tick_limit=self.step_limit)
        self._kinder = KinderSkills(shelf=self._skills, backend=self.backend)
        self.backend.grip = 0.0

    def on_objects_moved(self) -> None:
        """A human moved objects; whatever they took out of the hand is no longer held."""
        if self._kinder is None or self._skills is None:
            return
        self._kinder.sync()
        self.backend.grip = self._skills.grip

    def _run(self, *, skill: str, **kwargs: Any) -> None:
        skills, kinder = self._skills, self._kinder
        if skills is None or kinder is None:
            raise ControllerFailure("no scene: reset the environment before acting")
        skills.ticks = 0
        try:
            kinder.run(skill=getattr(kinder, skill), label=skill, **kwargs)
        except SkillFailed as failure:
            raise ControllerFailure(f"{skill}: {failure}") from failure
        finally:
            self.steps = skills.ticks
            self.backend.grip = skills.grip

    def pick_wiper(self) -> None:
        self._run(skill="pick_wiper")

    def sweep_shelf(self, *, cube: str, offset: float, angle: float) -> None:
        self._run(skill="sweep_shelf", cube=cube, offset=offset, angle=angle)

    def return_wiper(self) -> None:
        self._run(skill="return_wiper")

    def pick_cube_from_shelf(self, *, cube: str) -> None:
        self._run(skill="pick_cube_from_shelf", cube=cube)

    def pick_cube_from_floor(self, *, cube: str) -> None:
        self._run(skill="pick_cube_from_floor", cube=cube)

    def place_cube_in_box(self, *, cube: str) -> None:
        self._run(skill="place_cube_in_box", cube=cube)

    def return_cube_to_shelf(self, *, cube: str) -> None:
        self._run(skill="return_cube_to_shelf", cube=cube)
