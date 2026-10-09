"""Fresh interface-only bootstrap; contains no learned controller or robot geometry."""

# ruff: noqa: PLR0917 -- public generated-controller interface

from typing import Any


class GeneratedSkills:
    def __init__(self, action_space: Any, observation_space: Any, primitives: Any) -> None:
        self.action_space = action_space

    def reset(self, observation: Any, skill: str) -> None:
        self.skill = skill

    def get_action(self, observation: Any) -> Any:
        return None
