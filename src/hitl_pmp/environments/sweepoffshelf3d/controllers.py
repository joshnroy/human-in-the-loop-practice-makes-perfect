"""SweepOffShelf3D's skill controllers: the scripted skills, run on the backend's env.

`ShelfSkills` (scripted_skills.py) drives the TidyBot through KINDER's 11-D action. This
wrapper gives each planner skill its own step budget, turns `SkillFailed` into
`ControllerFailure`, rebuilds the skills whenever the scene is rebuilt, and keeps their
grasp bookkeeping honest after a human moves things.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .backend import SweepOffShelf3DBackend
from .scripted_skills import ShelfSkills, SkillFailed


class ControllerFailure(Exception):
    """A skill gave up: unreachable target, failed convergence or step budget."""


class SweepOffShelf3DControllers(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: SweepOffShelf3DBackend
    step_limit: int = 12000
    steps: int = 0

    _skills: ShelfSkills | None = PrivateAttr(default=None)

    def reset_counter(self) -> None:
        self.steps = 0

    def on_scene_reset(self) -> None:
        """The scene was rebuilt: the skills re-read it from scratch."""
        self._skills = ShelfSkills(env=self.backend.env(), tick_limit=self.step_limit)
        self.backend.grip = 0.0

    def on_objects_moved(self) -> None:
        """A human moved objects; whatever they took out of the hand is no longer held."""
        skills = self._skills
        if skills is None:
            return
        if skills.held_cube is not None and not self.backend.held(name=skills.held_cube):
            skills.held_cube = None
            skills.grip = 0.0
        if skills.in_hand is not None and not self.backend.held(name=self.backend.wiper_name):
            skills.in_hand = None
            skills.grasp_kind = None
            skills.grip = 0.0
        self.backend.grip = skills.grip

    def _run(self, *, skill: str, **kwargs: Any) -> None:
        skills = self._skills
        if skills is None:
            raise ControllerFailure("no scene: reset the environment before acting")
        skills.ticks = 0
        try:
            getattr(skills, skill)(**kwargs)
        except SkillFailed as failure:
            raise ControllerFailure(f"{skill}: {failure}") from failure
        finally:
            self.steps = skills.ticks
            self.backend.grip = skills.grip

    def pick_wiper(self) -> None:
        self._run(skill="pick_wiper")

    def sweep_shelf(self, *, height: float, speed: float) -> None:
        self._run(skill="sweep_shelf", height=height, speed=speed)

    def return_wiper(self) -> None:
        self._run(skill="return_wiper")

    def pick_cube_from_shelf(self, *, cube: str) -> None:
        self._run(skill="pick_cube_from_shelf", cube=cube)

    def pick_cube_from_floor(self, *, cube: str) -> None:
        self._run(skill="pick_cube_from_floor", cube=cube)

    def place_cube_in_box(self, *, cube: str) -> None:
        self._run(skill="place_cube_in_box", cube=cube)
